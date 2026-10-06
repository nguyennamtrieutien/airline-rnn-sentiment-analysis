import fs from 'node:fs';
import assert from 'node:assert/strict';
import {BrowserRNN} from '../airline_rnn/web/browser-engine.mjs';

const root = new URL('../', import.meta.url);
const read = name => JSON.parse(fs.readFileSync(new URL(name, root), 'utf8'));
const buffer = fs.readFileSync(new URL('deploy/browser-model/weights.bin', root));
const model = new BrowserRNN(read('deploy/browser-model/manifest.json'), read('deploy/browser-model/tokenizer.json'),
  read('deploy/browser-model/preprocessing.json'), buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
const fixtures = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const began = performance.now();
let maximumError = 0, maximumHiddenError = 0, mismatches = 0, cases = 0;
for (const sample of fixtures.cases) {
  const events = Array.from(model.events(sample.text));
  assert.equal(events.length, 18);
  assert.deepEqual(events.filter(x => x.type === 'step_done').map(x => x.step), [1,2,3,4,5,6,7,8]);
  const result = events.at(-1).result;
  assert.equal(result.clean_text, sample.clean_text, `P0 mismatch: ${sample.text}`);
  if (sample.ids) assert.deepEqual(events.find(x => x.type === 'step_done' && x.step === 5).data.ids, sample.ids);
  assert.equal(result.token_length, sample.token_length);
  assert.equal(result.unknown_tokens, sample.unknown_tokens);
  assert.equal(result.truncated, sample.truncated);
  if (result.sentiment !== sample.sentiment) mismatches++;
  for (const name of model.manifest.labels)
    maximumError = Math.max(maximumError, Math.abs(result.probabilities[name] - sample.probabilities[name]));
  if (sample.hidden_preview) {
    const hidden = events.find(x => x.type === 'step_done' && x.step === 6).data.last_hidden_state_preview;
    for (let i = 0; i < hidden.length; i++) maximumHiddenError = Math.max(maximumHiddenError, Math.abs(hidden[i] - sample.hidden_preview[i]));
  }
  cases++;
}
for (const text of ['', '  ', '\u001c', null, 42, 'x'.repeat(4001)]) assert.throws(() => Array.from(model.events(text)));
assert.equal(mismatches, 0, 'Converted model changed predicted classes');
assert.ok(maximumError < 0.0001, `Probability deviation too large: ${maximumError}`);
assert.ok(maximumHiddenError < 0.0001, `Hidden-state deviation too large: ${maximumHiddenError}`);
console.log(JSON.stringify({passed: true, cases, original_test_cases: fixtures.original_test_cases,
  predicted_class_mismatches: mismatches, max_probability_absolute_error: maximumError,
  max_hidden_preview_absolute_error: maximumHiddenError, invalid_inputs_rejected: 6,
  elapsed_seconds: (performance.now() - began) / 1000,
  source_model_sha256: model.manifest.source_model_sha256,
  scope: 'Inference conversion parity only; no fitting, tuning, or new scientific scores.'}, null, 2));
