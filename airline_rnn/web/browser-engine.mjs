// Frozen RNN inference: no fitting, random weights, or external inference service.
const f32 = Math.fround;
const word = '[\\p{L}\\p{N}_]';
const now = () => performance.now();
const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

function transpose(matrix, rows, cols) {
  const result = new Float32Array(matrix.length);
  for (let row = 0; row < rows; row++)
    for (let col = 0; col < cols; col++) result[col * rows + row] = matrix[row * cols + col];
  return result;
}

function multiply(vector, transposed, columns) {
  const result = new Float32Array(columns), rows = vector.length;
  for (let col = 0; col < columns; col++) {
    let sum = 0;
    const offset = col * rows;
    for (let row = 0; row < rows; row++) sum += vector[row] * transposed[offset + row];
    result[col] = sum;
  }
  return result;
}

export class BrowserRNN {
  constructor(manifest, tokenizer, preprocessing, binary) {
    if (manifest.format !== 'airline-rnn-browser-v1' || manifest.dtype !== 'float32-le')
      throw Error('Định dạng mô hình RNN không được hỗ trợ.');
    this.manifest = manifest;
    this.config = manifest.config;
    this.representation = manifest.representation;
    this.tokenizer = tokenizer;
    this.preprocessing = preprocessing;
    this.invalidCodepoints = new Set(preprocessing.invalid_codepoints);
    const chars = preprocessing.whitespace_codepoints.map(code => `\\u{${code.toString(16)}}`).join('');
    this.whitespaceClass = `[${chars}]`;
    this.whitespace = new RegExp(this.whitespaceClass + '+', 'gu');
    this.nonText = new RegExp(`[^a-z0-9${chars}]`, 'gu');
    if (manifest.representation.legacy || !manifest.config.mask_zero || tokenizer.legacy)
      throw Error('Cấu hình phải là RNN P0 có masking.');
    if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1)
      throw Error('Thiết bị không hỗ trợ định dạng trọng số hiện tại.');
    if (binary.byteLength !== manifest.parameters * 4)
      throw Error('Kích thước trọng số RNN không hợp lệ.');
    this.weights = {};
    for (const tensor of manifest.tensors) {
      const product = tensor.shape.reduce((a, b) => a * b, 1);
      if (tensor.length !== product) throw Error('Shape trọng số không hợp lệ.');
      this.weights[tensor.name] = new Float32Array(binary, tensor.offset * 4, tensor.length);
    }
    const c = this.config, w = this.weights;
    this.inputKernel = transpose(w.input_kernel, c.embedding_dimension, c.hidden_units);
    this.recurrentKernel = transpose(w.recurrent_kernel, c.hidden_units, c.hidden_units);
    this.denseKernel = transpose(w.dense_kernel, c.hidden_units, c.dense_units);
    this.outputKernel = transpose(w.output_kernel, c.dense_units, manifest.labels.length);
  }

  unescape(text) {
    const p = this.preprocessing;
    return text.replace(/&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)/g, (match, reference) => {
      if (reference[0] === '#') {
        const hex = reference[1].toLowerCase() === 'x';
        const code = parseInt(reference.slice(hex ? 2 : 1).replace(/;$/, ''), hex ? 16 : 10);
        if (own(p.invalid_charrefs, code)) return p.invalid_charrefs[code];
        if (code >= 0xD800 && code <= 0xDFFF || code > 0x10FFFF) return '\uFFFD';
        if (this.invalidCodepoints.has(code)) return '';
        return String.fromCodePoint(code);
      }
      if (own(p.html_entities, reference)) return p.html_entities[reference];
      for (let length = reference.length - 1; length > 1; length--)
        if (own(p.html_entities, reference.slice(0, length)))
          return p.html_entities[reference.slice(0, length)] + reference.slice(length);
      return match;
    });
  }

  clean(text) {
    let cleaned = this.unescape(text).normalize('NFKC').replace(/[’‘]/g, "'").toLowerCase();
    cleaned = cleaned.replace(new RegExp(`https?://[^${this.whitespaceClass.slice(1, -1)}]+|www\\.[^${this.whitespaceClass.slice(1, -1)}]+`, 'gu'), ' urlmarker ');
    cleaned = cleaned.replace(/@[a-z0-9_]+/g, ' usermarker ');
    for (const [key, value] of Object.entries(this.preprocessing.contractions))
      cleaned = cleaned.replace(new RegExp(`(?<!${word})${key}(?!${word})`, 'gu'), value);
    cleaned = cleaned.replace(new RegExp(`(?<!${word})(${word}+)n't(?!${word})`, 'gu'), '$1 not');
    for (const [suffix, expansion] of [["'re", ' are'], ["'ve", ' have'], ["'ll", ' will'], ["'d", ' would'], ["'m", ' am']])
      cleaned = cleaned.replace(new RegExp(`${suffix}(?!${word})`, 'gu'), expansion);
    return cleaned.replace(this.nonText, ' ').replace(this.whitespace, ' ').trim() || 'emptymarker';
  }

  validate(text) {
    if (typeof text !== 'string' || !text.replace(this.whitespace, '').length) throw Error('Hãy nhập một câu hoặc tweet tiếng Anh.');
    if (Array.from(text).length > 4000) throw Error('Câu nhập quá dài; tối đa 4.000 ký tự.');
  }

  encode(cleaned) {
    const words = cleaned.split(' ').filter(Boolean), ids = [], tokens = [];
    let unknown = 0;
    for (const token of words) {
      const known = own(this.tokenizer.word_index, token) ? this.tokenizer.word_index[token] : undefined;
      const oov = known === undefined || known >= this.tokenizer.vocabulary_size;
      const id = oov ? 1 : known;
      if (oov) unknown++;
      ids.push(id); tokens.push({word: token, id, oov});
    }
    return {words, ids, tokens, unknown};
  }

  forward(ids) {
    const c = this.config, w = this.weights;
    let hidden = new Float32Array(c.hidden_units), firstEmbedding = [];
    for (const id of ids) {
      if (id === 0) continue; // Keras Embedding(mask_zero=True): carry state through PAD.
      const embedding = w.embedding.subarray(id * c.embedding_dimension, (id + 1) * c.embedding_dimension);
      if (!firstEmbedding.length) firstEmbedding = Array.from(embedding.slice(0, 8));
      const input = multiply(embedding, this.inputKernel, c.hidden_units);
      const recurrent = multiply(hidden, this.recurrentKernel, c.hidden_units);
      for (let unit = 0; unit < c.hidden_units; unit++)
        hidden[unit] = Math.tanh(f32(f32(input[unit] + w.rnn_bias[unit]) + recurrent[unit]));
    }
    const dense = multiply(hidden, this.denseKernel, c.dense_units);
    for (let unit = 0; unit < dense.length; unit++) dense[unit] = Math.max(0, f32(dense[unit] + w.dense_bias[unit]));
    const logits = multiply(dense, this.outputKernel, this.manifest.labels.length);
    for (let unit = 0; unit < logits.length; unit++) logits[unit] = f32(logits[unit] + w.output_bias[unit]);
    const maximum = Math.max(...logits), exponentials = Array.from(logits, x => Math.exp(x - maximum));
    const total = exponentials.reduce((a, b) => a + b, 0);
    return {hidden, dense, firstEmbedding, probabilities: exponentials.map(x => x / total)};
  }

  *events(text) {
    this.validate(text);
    const began = now(), requestId = crypto.randomUUID(), c = this.config, m = this.manifest;
    const start = step => ({type: 'step_start', step, request_id: requestId});
    const done = (step, data, started) => ({type: 'step_done', step, data, request_id: requestId, elapsed_ms: now() - started});
    yield {type: 'start', request_id: requestId, steps: 8};
    yield start(1); let started = now();
    yield done(1, {original_text: text, characters: Array.from(text).length, max_characters: 4000}, started);
    yield start(2); started = now(); const cleaned = this.clean(text);
    yield done(2, {clean_text: cleaned, changed: cleaned !== text, empty_marker: cleaned === 'emptymarker', preprocessing: 'P0'}, started);
    yield start(3); started = now(); const encoded = this.encode(cleaned);
    yield done(3, {words: encoded.words}, started);
    yield start(4); started = now();
    yield done(4, {tokens: encoded.tokens, ids_before_padding: encoded.ids, token_length: encoded.words.length,
      sequence_length_before_padding: encoded.ids.length, unknown_tokens: encoded.unknown,
      oov_rate: encoded.words.length ? encoded.unknown / encoded.words.length : 0,
      vocabulary_size: c.vocabulary_size, vocabulary_fit_scope: this.representation.fit_scope}, started);
    yield start(5); started = now();
    const length = this.representation.sequence_length, ids = encoded.ids.slice(0, length), retained = ids.length;
    while (ids.length < length) ids.push(0);
    yield done(5, {ids, mask: ids.map(id => id !== 0), shape: [1, length], sequence_length: length,
      retained_tokens: retained, padding_tokens: length - retained, removed_tokens: Math.max(0, encoded.ids.length - length),
      mask_zero: true, padding: 'post', truncating: 'post'}, started);
    yield start(6); started = now(); const trace = this.forward(ids);
    yield done(6, {embedding_shape: [1, length, c.embedding_dimension], hidden_shape: [1, c.hidden_units],
      dense_shape: [1, c.dense_units], output_shape: [1, m.labels.length],
      first_token_embedding_preview: trace.firstEmbedding, last_hidden_state_preview: Array.from(trace.hidden.slice(0, 8)),
      hidden_l2_norm: Math.sqrt(trace.hidden.reduce((sum, x) => sum + x * x, 0)), parameters: m.parameters,
      hidden_units: c.hidden_units, embedding_dimension: c.embedding_dimension, activation: 'tanh', dense_activation: 'relu',
      dropout_active: false, config_id: c.id, seed: m.seed,
      note: 'Các vector là giá trị thật trong forward pass; không phải giải thích mức đóng góp của từng từ.'}, started);
    yield start(7); started = now();
    const scores = Object.fromEntries(m.labels.map((name, index) => [name, trace.probabilities[index]]));
    yield done(7, {probabilities: scores, probability_sum: trace.probabilities.reduce((a, b) => a + b, 0)}, started);
    yield start(8); started = now();
    const index = trace.probabilities.indexOf(Math.max(...trace.probabilities));
    const result = {sentiment: m.labels[index], probabilities: scores, original_text: text, sequence_length: length,
      confidence: trace.probabilities[index], clean_text: cleaned, token_length: encoded.ids.length,
      unknown_tokens: encoded.unknown, truncated: encoded.ids.length > length, config_id: c.id, seed: m.seed,
      note: 'Softmax chưa được calibration; không diễn giải score như độ tin cậy đã kiểm chứng.'};
    yield done(8, {sentiment: result.sentiment, confidence: result.confidence, rule: 'argmax'}, started);
    yield {type: 'result', request_id: requestId, total_elapsed_ms: now() - began, result};
  }
}
