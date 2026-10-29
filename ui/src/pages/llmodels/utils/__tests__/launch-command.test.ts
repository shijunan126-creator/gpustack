import assert from 'node:assert/strict';
import test from 'node:test';
import {
  mergeLaunchCommand,
  parseLaunchCommand,
  tokenize
} from '../launch-command';

const RECIPE = `export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"

# Ensure the model path matches the directory recorded during download
vllm serve "$MODEL_PATH" \\
    --host 0.0.0.0 \\
    --port 8900 \\
    --max-model-len 150000 \\
    --served-model-name dsv41 \\
    --data-parallel-size 4 \\
    --enable-expert-parallel \\
    --enable-auto-tool-choice \\
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \\
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_flashcomm1":true
    }'
`;

test('a recipe becomes environment variables and parameter entries', () => {
  const result = parseLaunchCommand(RECIPE);
  assert.equal(result.error, undefined);
  assert.deepEqual(result.env, {
    VLLM_USE_V2_MODEL_RUNNER: '0',
    VLLM_ENGINE_READY_TIMEOUT_S: '36000',
    PYTORCH_NPU_ALLOC_CONF: 'expandable_segments:True',
    HCCL_BUFFSIZE: '1024',
    HCCL_OP_EXPANSION_MODE: 'AIV'
  });
  assert.deepEqual(result.parameters, [
    '--max-model-len=150000',
    '--data-parallel-size=4',
    '--enable-expert-parallel',
    '--enable-auto-tool-choice',
    '--compilation-config={"cudagraph_mode":"FULL_DECODE_ONLY"}',
    '--additional-config={"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_flashcomm1":true}'
  ]);
});

test('flags GPUStack sets per instance and the model are left out', () => {
  const result = parseLaunchCommand(RECIPE);
  assert.deepEqual(
    result.dropped.map(({ flag, reason }) => [flag, reason]),
    [
      ['--host', 'managed'],
      ['--port', 'managed'],
      ['--served-model-name', 'managed']
    ]
  );
  assert.equal(result.model, '$MODEL_PATH');
  assert.deepEqual(result.variables, []);
});

test('--model and an unquoted positional model are both the model', () => {
  const positional = parseLaunchCommand(
    'vllm serve Qwen/Qwen3-8B --dtype bf16'
  );
  assert.equal(positional.model, 'Qwen/Qwen3-8B');
  assert.deepEqual(positional.parameters, ['--dtype=bf16']);

  const flagged = parseLaunchCommand(
    'python3 -m vllm.entrypoints.openai.api_server --model /models/a --tp 2'
  );
  assert.equal(flagged.model, '/models/a');
  assert.deepEqual(flagged.parameters, ['--tp=2']);
  assert.deepEqual(flagged.dropped, [
    { flag: '--model', value: '/models/a', reason: 'model' }
  ]);
});

test('sglang launchers are recognised', () => {
  const module = parseLaunchCommand(
    'python -m sglang.launch_server --model-path /m --tp-size 4 --enable-dp-attention'
  );
  assert.deepEqual(module.parameters, ['--tp-size=4', '--enable-dp-attention']);
  const cli = parseLaunchCommand(
    'sglang serve --model-path /m --mem-fraction-static 0.8'
  );
  assert.deepEqual(cli.parameters, ['--mem-fraction-static=0.8']);
});

test('a flag with several values keeps them as separate entries', () => {
  const result = parseLaunchCommand(
    'vllm serve m --lora-modules a=/x b=/y --max-model-len=4096 --seed -1'
  );
  assert.deepEqual(result.parameters, [
    '--lora-modules',
    'a=/x',
    'b=/y',
    '--max-model-len=4096',
    '--seed=-1'
  ]);
});

test('variables written in front of the command and after && are read', () => {
  const result = parseLaunchCommand(
    'cd /opt && A=1 B="x y" vllm serve m --dtype bf16 && echo done'
  );
  assert.deepEqual(result.env, { A: '1', B: 'x y' });
  assert.deepEqual(result.parameters, ['--dtype=bf16']);
});

test('shell variables in values are reported rather than expanded', () => {
  const result = parseLaunchCommand(
    'export DIR="$HOME/cache"\nvllm serve m --download-dir "$DIR" --dtype bf16'
  );
  assert.equal(result.env.DIR, '$HOME/cache');
  assert.deepEqual(result.variables, ['DIR', '--download-dir']);
  assert.ok(result.parameters.includes('--download-dir=$DIR'));
});

test('single quotes keep a dollar sign literal', () => {
  const result = parseLaunchCommand("vllm serve m --chat-template '$x'");
  assert.deepEqual(result.variables, []);
});

test('text without an engine command is refused with a reason', () => {
  assert.equal(parseLaunchCommand('   \n').error, 'empty');
  assert.equal(parseLaunchCommand('export A=1\nls -l').error, 'no_command');
  assert.equal(
    parseLaunchCommand('vllm serve m --x "oops').error,
    'unterminated_quote'
  );
});

test('tokenize drops comments and joins continuations', () => {
  const statements = tokenize('a b # note\nc \\\n d');
  assert.deepEqual(
    statements?.map((words) => words.map((word) => word.text)),
    [
      ['a', 'b'],
      ['c', 'd']
    ]
  );
});

test('merging replaces a flag already set and keeps the rest', () => {
  const merged = mergeLaunchCommand(
    {
      parameters: ['--max-model-len=4096', '--seed=1', 'positional'],
      env: { KEEP: '1', HCCL_BUFFSIZE: '200' }
    },
    {
      parameters: ['--max-model-len=150000', '--enable-expert-parallel'],
      env: { HCCL_BUFFSIZE: '1024' }
    }
  );
  assert.deepEqual(merged.parameters, [
    '--seed=1',
    'positional',
    '--max-model-len=150000',
    '--enable-expert-parallel'
  ]);
  assert.deepEqual(merged.env, { KEEP: '1', HCCL_BUFFSIZE: '1024' });
});

test('merging into an empty form just adds', () => {
  assert.deepEqual(
    mergeLaunchCommand({}, { parameters: ['--a'], env: { X: '1' } }),
    { parameters: ['--a'], env: { X: '1' } }
  );
});
