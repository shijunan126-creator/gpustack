import { JSDOM } from 'jsdom';
import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import { act, createElement } from 'react';

let autoProps: any;
let selectProps: any;
mock.module('@gpustack/core-ui', {
  namedExports: {
    AutoComplete: (props: any) => {
      autoProps = props;
      return createElement('div');
    },
    Input: { Input: () => createElement('div') },
    Select: (props: any) => {
      selectProps = props;
      return createElement('div');
    },
    TooltipList: () => null,
    useAppUtils: () => ({ getRuleMessage: () => 'required' })
  }
});
mock.module('@umijs/max', {
  namedExports: {
    useIntl: () => ({ formatMessage: ({ id }: { id: string }) => id })
  }
});
const directories: Record<string, any[]> = {
  '1:/': [{ name: 'data', path: '/data', is_dir: true, is_model: false }],
  '1:/data/': [
    { name: 'qwen', path: '/data/qwen', is_dir: true, is_model: true },
    { name: 'misc', path: '/data/misc', is_dir: true, is_model: false }
  ],
  '2:/': [{ name: 'models', path: '/models', is_dir: true, is_model: false }]
};
const listCalls: string[] = [];
mock.module(new URL('../../apis/index.ts', import.meta.url).href, {
  namedExports: {
    queryWorkerDirectory: async (workerId: number, path: string) => {
      listCalls.push(`${workerId}:${path}`);
      return { entries: directories[`${workerId}:${path}`] || [] };
    }
  }
});
mock.module(new URL('../../../resources/apis/index.ts', import.meta.url).href, {
  namedExports: {
    queryWorkersList: async () => ({
      items: [
        { id: 1, name: 'w1', state: 'ready' },
        { id: 2, name: 'w2', state: 'ready' },
        { id: 3, name: 'w3', state: 'not_ready' }
      ]
    })
  }
});

const dom = new JSDOM('<div id="root"></div>');
Object.defineProperties(globalThis, {
  window: { configurable: true, value: dom.window },
  document: { configurable: true, value: dom.window.document },
  HTMLElement: { configurable: true, value: dom.window.HTMLElement },
  Element: { configurable: true, value: dom.window.Element },
  navigator: { configurable: true, value: dom.window.navigator },
  IS_REACT_ACT_ENVIRONMENT: { configurable: true, value: true }
});
dom.window.matchMedia = (media) => ({
  matches: false,
  media,
  onchange: null,
  addListener() {},
  removeListener() {},
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent: () => false
});
// react-dom decides whether the DOM supports `input` events when it loads, so
// it has to load after the globals above exist.
const { createRoot } = await import('react-dom/client');
const { Form } = await import('antd');
const { FormContext } = await import('../../config/form-context.ts');
const { modelSourceMap } = await import('../../config/index.ts');
const { default: LocalPathForm } = await import('../local-path-source.tsx');

test('the path field offers the directories of a ready worker as you go', async () => {
  const root = createRoot(dom.window.document.getElementById('root')!);
  const onValuesChange = mock.fn();
  let form: any;
  const Capture = () => {
    form = Form.useFormInstance();
    return null;
  };
  await act(async () => {
    root.render(
      createElement(
        FormContext.Provider,
        { value: { onValuesChange, flatBackendOptions: [] } as any },
        createElement(
          Form,
          {
            initialValues: {
              source: modelSourceMap.local_path_value,
              cluster_id: 1,
              local_path: ''
            }
          },
          createElement(
            Form.Item,
            { name: 'source', hidden: true },
            createElement('input')
          ),
          createElement(
            Form.Item,
            { name: 'cluster_id', hidden: true },
            createElement('input')
          ),
          createElement(Capture),
          createElement(LocalPathForm)
        )
      )
    );
  });
  const values = () => autoProps.options.map((o: any) => o.value);

  await act(async () => {
    await autoProps.onFocus();
  });
  assert.deepEqual(listCalls, ['1:/']);
  assert.deepEqual(values(), ['/data/']);
  assert.equal(selectProps.options.length, 2, 'only ready workers are offered');

  // A directory that is not a model is entered, not chosen.
  await act(async () => {
    autoProps.onSelect('/data/', { isFinal: false });
  });
  assert.equal(form.getFieldValue('local_path'), '/data/');
  assert.deepEqual(values(), ['/data/qwen', '/data/misc/']);
  assert.equal(onValuesChange.mock.callCount(), 0);

  // Typing within a listed directory narrows it without another request.
  await act(async () => {
    autoProps.onSearch('/data/qw');
  });
  assert.deepEqual(values(), ['/data/qwen']);
  assert.deepEqual(listCalls, ['1:/', '1:/data/']);

  // A model is a place to stop.
  await act(async () => {
    autoProps.onSelect('/data/qwen', { isFinal: true });
  });
  assert.equal(form.getFieldValue('local_path'), '/data/qwen');
  assert.equal(onValuesChange.mock.callCount(), 1);
  assert.equal(autoProps.open, false);

  // Another worker lists the same typed path on its own filesystem.
  await act(async () => {
    selectProps.onChange(2);
  });
  assert.deepEqual(listCalls.at(-1), '2:/data/');
  root.unmount();
});
