import { JSDOM } from 'jsdom';
import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import { act, createElement } from 'react';

let listProps: any;
mock.module('@gpustack/core-ui', {
  namedExports: {
    ListInput: (props: any) => {
      listProps = props;
      return createElement('div', null, props.labelExtra);
    }
  }
});
mock.module('@umijs/max', {
  namedExports: {
    useIntl: () => ({
      locale: 'en-US',
      formatMessage: ({ id }: { id: string }, values?: any) =>
        values ? `${id} ${JSON.stringify(values)}` : id
    })
  }
});
mock.module(new URL('../../config/index.ts', import.meta.url).href, {
  namedExports: {
    backendParamsHolderTips: {},
    getBackendParamsTips: (backend: string) => ({ backend, link: '' })
  }
});

const dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true });
Object.defineProperties(globalThis, {
  window: { configurable: true, value: dom.window },
  document: { configurable: true, value: dom.window.document },
  getComputedStyle: {
    configurable: true,
    // jsdom does not implement the pseudo-element form antd's scroll lock uses.
    value: (element: Element) => dom.window.getComputedStyle(element)
  },
  HTMLElement: { configurable: true, value: dom.window.HTMLElement },
  HTMLTextAreaElement: {
    configurable: true,
    value: dom.window.HTMLTextAreaElement
  },
  Element: { configurable: true, value: dom.window.Element },
  SVGElement: { configurable: true, value: dom.window.SVGElement },
  ResizeObserver: {
    configurable: true,
    value: class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  },
  Node: { configurable: true, value: dom.window.Node },
  ShadowRoot: { configurable: true, value: dom.window.ShadowRoot },
  MutationObserver: { configurable: true, value: dom.window.MutationObserver },
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
const { default: PasteCommandEntry } =
  await import('../paste-command-entry.tsx');
const { default: BackendParametersList } =
  await import('../backend-parameters-list.tsx');

const SCRIPT = `export HCCL_BUFFSIZE=1024
vllm serve "$MODEL_PATH" \\
  --port 8900 \\
  --max-model-len 150000 \\
  --additional-config '{
    "enable_cpu_binding": true
  }'`;

test('applying a pasted script fills parameters and environment together', async () => {
  const container = dom.window.document.getElementById('root')!;
  const root = createRoot(container);
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
        { value: { flatBackendOptions: [], onValuesChange } as any },
        createElement(
          Form,
          {
            initialValues: {
              backend: 'vLLM',
              backend_parameters: ['--max-model-len=4096', '--seed=1'],
              env: { KEEP: '1' }
            }
          },
          createElement(
            Form.Item,
            { name: 'backend', hidden: true },
            createElement('input')
          ),
          createElement(
            Form.Item,
            { name: 'env', hidden: true },
            createElement('input')
          ),
          createElement(Capture),
          createElement(BackendParametersList)
        )
      )
    );
  });

  await act(async () => {
    listProps.labelExtra.props.onClick();
  });
  const textarea = dom.window.document.querySelector('textarea')!;
  assert.ok(textarea, 'the modal opens a text area');
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      dom.window.HTMLTextAreaElement.prototype,
      'value'
    )!.set!;
    setter.call(textarea, SCRIPT);
    textarea.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
  });
  const body = dom.window.document.body.textContent!;
  assert.match(body, /HCCL_BUFFSIZE=1024/);
  assert.match(body, /--port 8900/);

  const apply = [...dom.window.document.querySelectorAll('button')].find(
    (b) => b.textContent === 'models.form.pasteCommand.apply'
  )!;
  await act(async () => {
    apply.click();
  });

  assert.deepEqual(form.getFieldValue('backend_parameters'), [
    '--seed=1',
    '--max-model-len=150000',
    '--additional-config={"enable_cpu_binding":true}'
  ]);
  assert.deepEqual(form.getFieldValue('env'), {
    KEEP: '1',
    HCCL_BUFFSIZE: '1024'
  });
  assert.equal(onValuesChange.mock.callCount(), 1);
  root.unmount();
});

test('the entry under the replica count fills the same fields', async () => {
  const container = dom.window.document.createElement('div');
  dom.window.document.body.appendChild(container);
  const root = createRoot(container);
  let form: any;
  const Capture = () => {
    form = Form.useFormInstance();
    return null;
  };
  await act(async () => {
    root.render(
      createElement(
        FormContext.Provider,
        { value: { flatBackendOptions: [], onValuesChange: mock.fn() } as any },
        createElement(
          Form,
          { initialValues: { backend: 'vLLM' } },
          createElement(Capture),
          createElement(PasteCommandEntry)
        )
      )
    );
  });

  const buttons = () => [...dom.window.document.querySelectorAll('button')];
  const label = 'models.form.pasteCommand';
  await act(async () => {
    buttons()
      .find((b) => b.textContent === label)!
      .click();
  });
  const textarea = dom.window.document.querySelector('textarea')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(
      dom.window.HTMLTextAreaElement.prototype,
      'value'
    )!.set!.call(textarea, 'export A=1\nvllm serve m --tensor-parallel-size 4');
    textarea.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
  });
  await act(async () => {
    buttons()
      .find((b) => b.textContent === `${label}.apply`)!
      .click();
  });

  assert.deepEqual(form.getFieldValue('backend_parameters'), [
    '--tensor-parallel-size=4'
  ]);
  assert.deepEqual(form.getFieldValue('env'), { A: '1' });
  root.unmount();
});
