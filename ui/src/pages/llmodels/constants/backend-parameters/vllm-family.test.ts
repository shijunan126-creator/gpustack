import assert from 'node:assert/strict';
import test from 'node:test';
import useCheckBackend from '../../hooks/use-check-backend';
import params, {
  backendOptionsMap,
  BuiltInBackendOptions,
  defaultAscendBackend,
  isVllmFamily
} from './index';

test('vLLM-Ascend belongs to the vLLM family, other engines do not', () => {
  assert.equal(isVllmFamily(backendOptionsMap.vllm), true);
  assert.equal(isVllmFamily(backendOptionsMap.vllmAscend), true);
  assert.equal(isVllmFamily(backendOptionsMap.SGLang), false);
  assert.equal(isVllmFamily(backendOptionsMap.ascendMindie), false);
  assert.equal(isVllmFamily(backendOptionsMap.custom), false);
  assert.equal(isVllmFamily(undefined), false);
  assert.equal(isVllmFamily(null), false);
  assert.equal(isVllmFamily(''), false);
});

test('the frontend names the backend exactly as the server registers it', () => {
  assert.equal(backendOptionsMap.vllmAscend, 'vLLM-Ascend');
});

test('vLLM-Ascend is a built-in backend offering the vLLM parameter catalog', () => {
  assert.ok(BuiltInBackendOptions.includes(backendOptionsMap.vllmAscend));
  const catalog = params as Record<string, Array<{ value: string }>>;
  assert.deepEqual(
    catalog[backendOptionsMap.vllmAscend].map((p) => p.value),
    catalog[backendOptionsMap.vllm].map((p) => p.value)
  );
  assert.ok(catalog[backendOptionsMap.vllmAscend].length > 0);
});

const device = (vendor: string) => ({ children: [{ vendor }] });

test('a selection of only Ascend NPUs defaults to vLLM-Ascend', () => {
  const { checkCurrentbackend, checkOnlyAscendNPU } = useCheckBackend();
  const gpuOptions = [device('huawei'), device('Huawei')];
  assert.equal(checkOnlyAscendNPU(gpuOptions), true);
  assert.equal(defaultAscendBackend, backendOptionsMap.vllmAscend);
  assert.equal(
    checkCurrentbackend({
      isAudio: false,
      isGGUF: false,
      gpuOptions,
      defaultBackend: backendOptionsMap.vllm
    }),
    backendOptionsMap.vllmAscend
  );
});

test('mixed or non-Ascend devices keep the caller default', () => {
  const { checkCurrentbackend } = useCheckBackend();
  for (const gpuOptions of [
    [device('nvidia')],
    [device('huawei'), device('nvidia')],
    []
  ]) {
    assert.equal(
      checkCurrentbackend({
        isAudio: false,
        isGGUF: false,
        gpuOptions,
        defaultBackend: backendOptionsMap.vllm
      }),
      backendOptionsMap.vllm
    );
  }
});

test('an audio model stays on vLLM even on Ascend NPUs', () => {
  const { checkCurrentbackend } = useCheckBackend();
  assert.equal(
    checkCurrentbackend({
      isAudio: true,
      isGGUF: false,
      gpuOptions: [device('huawei')],
      defaultBackend: backendOptionsMap.vllmAscend
    }),
    backendOptionsMap.vllm
  );
});
