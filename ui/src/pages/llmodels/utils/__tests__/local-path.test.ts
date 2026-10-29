import assert from 'node:assert/strict';
import test from 'node:test';
import { listTarget, toPathOptions } from '../local-path';

test('the typed text decides which directory is listed', () => {
  assert.deepEqual(listTarget(''), { dir: '/', prefix: '' });
  assert.deepEqual(listTarget('/'), { dir: '/', prefix: '' });
  assert.deepEqual(listTarget('/data'), { dir: '/', prefix: 'data' });
  assert.deepEqual(listTarget('/data/'), { dir: '/data/', prefix: '' });
  assert.deepEqual(listTarget(' /data/mod '), { dir: '/data/', prefix: 'mod' });
  assert.deepEqual(listTarget('models'), { dir: '/', prefix: 'models' });
});

test('options continue the typed text and open directories further', () => {
  const entries = [
    { name: 'Qwen', path: '/m/Qwen', is_dir: true, is_model: true },
    { name: 'qwen-old', path: '/m/qwen-old', is_dir: true, is_model: false },
    { name: 'llama', path: '/m/llama', is_dir: true, is_model: true },
    { name: 'tiny.gguf', path: '/m/tiny.gguf', is_dir: false, is_model: true }
  ];
  assert.deepEqual(
    toPathOptions(entries, 'qw').map((o) => [o.value, o.isFinal]),
    [
      ['/m/Qwen', true],
      ['/m/qwen-old/', false]
    ]
  );
  const all = toPathOptions(entries, '');
  assert.deepEqual(
    all.map((o) => o.name),
    ['Qwen/', 'qwen-old/', 'llama/', 'tiny.gguf']
  );
  assert.equal(all[3].value, '/m/tiny.gguf');
  assert.equal(all[3].isFinal, true);
});
