import {BrowserRNN} from './browser-engine.mjs';

async function checkedAsset(base, name, expected) {
  const response = await fetch(new URL(name, base));
  if (!response.ok) throw Error(`Không tải được ${name}.`);
  const bytes = await response.arrayBuffer();
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
  if (hash !== expected) throw Error(`Checksum mô hình không khớp: ${name}.`);
  return bytes;
}

const initialized = (async () => {
  const base = new URL('../model/', import.meta.url);
  const response = await fetch(new URL('manifest.json', base));
  if (!response.ok) throw Error('Không tải được manifest RNN.');
  const manifest = await response.json();
  const [weights, tokenizer, preprocessing] = await Promise.all(
    ['weights.bin', 'tokenizer.json', 'preprocessing.json'].map(name => checkedAsset(base, name, manifest.assets[name])));
  const decoder = new TextDecoder();
  return new BrowserRNN(manifest, JSON.parse(decoder.decode(tokenizer)), JSON.parse(decoder.decode(preprocessing)), weights);
})();

self.onmessage = async ({data, ports}) => {
  const port = ports[0];
  try {
    const model = await initialized;
    if (data.type === 'health') {
      const m = model.manifest, c = model.config;
      port.postMessage({status: 'ready', label_names: m.labels, vocabulary_size: c.vocabulary_size,
        sequence_length: model.representation.sequence_length, config_id: c.id, seed: m.seed,
        runtime: 'browser-rnn', source_model_sha256: m.source_model_sha256});
    } else if (data.type === 'analyze') {
      for (const event of model.events(data.text)) port.postMessage(event);
    } else throw Error('Yêu cầu không hợp lệ.');
  } catch (error) {
    port.postMessage({type: 'error', error: error.message});
  } finally {
    port.postMessage({type: 'end'});
    port.close();
  }
};
