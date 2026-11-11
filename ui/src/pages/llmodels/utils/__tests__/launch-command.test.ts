import assert from 'node:assert/strict';
import test from 'node:test';
import {
  mergeLaunchCommand,
  parseLaunchCommand,
  sanitizeForPD,
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

// The published PD launcher scripts: a guard line, exports with shell
// arithmetic, and a serve command whose values are positional parameters.
const PD_DECODE = `#!/usr/bin/env bash
: "\${LOCAL_IP:?Set LOCAL_IP in the D0 or D1 wrapper}"
export HCCL_IF_IP="$LOCAL_IP"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_BUFFSIZE=1024
export HCCL_HOST_SOCKET_PORT_RANGE="$((60000 + $4 * 100))-$((60099 + $4 * 100))"
export VLLM_RPC_TIMEOUT=3600000
exec vllm serve "$MODEL_PATH" \\
  --device-ids "$1" \\
  --port "$2" \\
  --data-parallel-size "$3" \\
  --data-parallel-rank "$4" \\
  --tensor-parallel-size "$7" \\
  --prefill-context-parallel-size "$8" \\
  --served-model-name glm \\
  --max-num-seqs 40 \\
  --gpu-memory-utilization 0.92 \\
  --async-scheduling \\
  --enable-auto-tool-choice \\
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer"}' \\
  --additional-config '{"enable_flashcomm1": false}'
`;

test('a PD role keeps the engine flags and refuses what the form owns', () => {
  const result = parseLaunchCommand(PD_DECODE, { pd: true });
  assert.equal(result.error, undefined);
  assert.deepEqual(result.parameters, [
    '--max-num-seqs=40',
    '--gpu-memory-utilization=0.92',
    '--async-scheduling',
    '--enable-auto-tool-choice',
    '--additional-config={"enable_flashcomm1":false}'
  ]);
  // Nothing positional survives: `$1`..`$8` have no launcher here.
  assert.ok(!result.parameters.some((entry) => entry.includes('$')));
  assert.deepEqual(
    result.dropped
      .filter(({ flag }) => flag.startsWith('-'))
      .map(({ flag, reason }) => [flag, reason]),
    [
      ['--device-ids', 'managed'],
      ['--port', 'managed'],
      ['--data-parallel-size', 'managed'],
      ['--data-parallel-rank', 'managed'],
      ['--tensor-parallel-size', 'managed'],
      ['--prefill-context-parallel-size', 'managed'],
      ['--served-model-name', 'managed'],
      ['--kv-transfer-config', 'managed']
    ]
  );
});

test('a PD role keeps its own environment and refuses the connection variables', () => {
  const result = parseLaunchCommand(PD_DECODE, { pd: true });
  assert.deepEqual(result.env, {
    HCCL_BUFFSIZE: '1024',
    VLLM_RPC_TIMEOUT: '3600000'
  });
  assert.deepEqual(
    result.dropped
      .filter(({ reason }) => reason === 'managed')
      .map(({ flag }) => flag)
      .filter((flag) => !flag.startsWith('-'))
      .sort(),
    ['HCCL_HOST_SOCKET_PORT_RANGE', 'HCCL_IF_IP', 'HCCL_SOCKET_IFNAME']
  );
  // Those values name this node's interface or another rank's ports; there is
  // no launcher here to supply them, so none may reach the form.
  assert.deepEqual(result.variables, []);
});

test('without PD a script keeps its shell variables as before', () => {
  const result = parseLaunchCommand(PD_DECODE);
  assert.ok(result.parameters.some((entry) => entry.includes('$7')));
  assert.equal(result.variables.includes('--tensor-parallel-size'), true);
});

test('switching PD on drops what a member cannot take from an earlier paste', () => {
  const cleaned = sanitizeForPD(
    [
      '--device-ids=$1',
      '--tensor-parallel-size=$7',
      '--max-num-seqs=40',
      '--seed=1024'
    ],
    { HCCL_IF_IP: '$LOCAL_IP', HCCL_BUFFSIZE: '1024' }
  );
  assert.deepEqual(cleaned.parameters, ['--max-num-seqs=40', '--seed=1024']);
  assert.deepEqual(cleaned.env, { HCCL_BUFFSIZE: '1024' });
});

