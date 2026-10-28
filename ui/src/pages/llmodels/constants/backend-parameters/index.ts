// preset backend parameters for built-in backends
import llamaParameters from './llama';
import mindieParameters from './mindie';
import sglangParameters from './sglang';
import vllmParameters from './vllm';

/**
 * backend label map
 */
export const backendOptionsMap = {
  llamaBox: 'llama-box',
  vllm: 'vLLM',
  vllmAscend: 'vLLM-Ascend',
  voxBox: 'VoxBox',
  ascendMindie: 'MindIE',
  custom: 'Custom',
  SGLang: 'SGLang'
};

// for checking built-in backends when selecting a gguf
export const BuiltInBackendOptions = [
  backendOptionsMap.vllm,
  backendOptionsMap.vllmAscend,
  backendOptionsMap.ascendMindie,
  backendOptionsMap.SGLang,
  backendOptionsMap.voxBox
];

// Engines that run the same vLLM server and accept the same command line,
// disaggregation recipes and speculative decoding. They differ in where their
// images come from and which devices they run on.
export const vllmFamily: string[] = [
  backendOptionsMap.vllm,
  backendOptionsMap.vllmAscend
];

export const isVllmFamily = (backend?: string | null): boolean => {
  return !!backend && vllmFamily.includes(backend);
};

// The backend a deployment starts on when every selected device is an Ascend
// NPU. MindIE stays selectable from the dropdown.
export const defaultAscendBackend = backendOptionsMap.vllmAscend;

export interface BackendParameter {
  label: string;
  value: string;
  options?: Array<string | number>;
}

export const generateBackendParameters = (options: BackendParameter[]) => {
  return options.map((option) => {
    return {
      label: option.label,
      value: option.value,
      opts: option.options?.map((opt) => {
        return {
          label: opt,
          value: opt
        };
      })
    };
  });
};

export default {
  [backendOptionsMap.llamaBox]: generateBackendParameters(llamaParameters),
  [backendOptionsMap.vllm]: generateBackendParameters(vllmParameters),
  [backendOptionsMap.vllmAscend]: generateBackendParameters(vllmParameters),
  [backendOptionsMap.ascendMindie]: generateBackendParameters(mindieParameters),
  [backendOptionsMap.voxBox]: [],
  [backendOptionsMap.custom]: [],
  [backendOptionsMap.SGLang]: generateBackendParameters(sglangParameters)
};
