import { queryWorkersList } from '@/pages/resources/apis';
import {
  AutoComplete,
  Select as SealSelect,
  TooltipList,
  useAppUtils
} from '@gpustack/core-ui';
import { useIntl } from '@umijs/max';
import { Flex, Form, Tag } from 'antd';
import _ from 'lodash';
import React, { useRef, useState } from 'react';
import { queryWorkerDirectory } from '../apis';
import { DeployFormKeyMap, localPathTipsList, modelSourceMap } from '../config';
import { useFormContext } from '../config/form-context';
import { DirectoryEntry, FormData } from '../config/types';
import {
  backendOptionsMap,
  defaultAscendBackend
} from '../constants/backend-parameters';
import useCheckBackend from '../hooks/use-check-backend';
import { listTarget, PathOption, toPathOptions } from '../utils/local-path';

interface BrowseState {
  workers: { label: string; value: number }[];
  workerId?: number;
  options: PathOption[];
  open: boolean;
}

const LocalPathForm: React.FC = () => {
  const { checkOnlyAscendNPU } = useCheckBackend();
  const form = Form.useFormInstance();
  const source = Form.useWatch('source', form);
  const {
    formKey,
    gpuOptions,
    flatBackendOptions,
    onValuesChange,
    onBackendChange
  } = useFormContext();
  const { getRuleMessage } = useAppUtils();
  const intl = useIntl();
  const localPathCache = useRef<string>(form.getFieldValue('local_path') || '');
  const clusterId = Form.useWatch('cluster_id', form);
  const [browse, setBrowse] = useState<BrowseState>({
    workers: [],
    options: [],
    open: false
  });
  // Only the newest listing may write state: they are fired as the user types
  // and the last one sent is not always the last one back.
  const listSeq = useRef(0);
  const listed = useRef<{ key: string; entries: DirectoryEntry[] } | null>(
    null
  );
  const workersLoaded = useRef(false);

  if (
    ![modelSourceMap.local_path_value].includes(source) ||
    formKey === DeployFormKeyMap.CATALOG
  ) {
    return null;
  }

  const handleOnBlur = (e: any) => {
    const value = e.target.value;
    onValuesChange?.({ local_path: value }, form.getFieldsValue());
  };

  const handleLocalPathBlur = async (e: any) => {
    const value = e.target.value;
    if (value === localPathCache.current || !value) {
      return;
    }
    const isEndwithGGUF = _.endsWith(value, '.gguf');
    const isBlobFile = value.split('/').pop().includes('sha256');
    let backend = form.getFieldValue('backend');
    const oldBackend = backend;

    if (isEndwithGGUF || isBlobFile) {
      backend = backendOptionsMap.llamaBox;
    } else if (
      !isEndwithGGUF &&
      !isBlobFile &&
      backend === backendOptionsMap.llamaBox
    ) {
      backend = checkOnlyAscendNPU(gpuOptions || [])
        ? defaultAscendBackend
        : backendOptionsMap.vllm;
    }
    form.setFieldValue('backend', backend);

    await new Promise((resolve) => {
      setTimeout(() => {
        resolve(true);
      }, 0);
    });

    if (oldBackend !== backend) {
      const option = flatBackendOptions.find((item) => item.value === backend);
      onBackendChange?.(backend, option);
    } else {
      onValuesChange?.({ local_path: value }, form.getFieldsValue());
    }
  };

  const handleOnFocus = () => {
    localPathCache.current = form.getFieldValue('local_path');
  };

  // Shows what continues the typed text. A directory is fetched once; typing
  // within it only narrows what was fetched.
  const listDirectory = async (value: string, workerId?: number) => {
    if (workerId === undefined) {
      return;
    }
    const seq = ++listSeq.current;
    const { dir, prefix } = listTarget(value);
    const key = `${workerId}:${dir}`;
    const show = (entries: DirectoryEntry[]) => {
      setBrowse((prev) => ({
        ...prev,
        options: toPathOptions(entries, prefix)
      }));
    };
    if (listed.current?.key === key) {
      show(listed.current.entries);
      return;
    }
    try {
      const listing = await queryWorkerDirectory(workerId, dir);
      if (seq === listSeq.current) {
        listed.current = { key, entries: listing.entries };
        show(listing.entries);
      }
    } catch (error) {
      if (seq === listSeq.current) {
        setBrowse((prev) => ({ ...prev, options: [] }));
      }
    }
  };

  // The workers are read when the field is first focused, not when the form
  // opens: most deployments never use a local path.
  const handleFocus = async () => {
    handleOnFocus();
    let workerId = browse.workerId;
    if (!workersLoaded.current) {
      workersLoaded.current = true;
      try {
        const res = await queryWorkersList(
          { cluster_id: clusterId, perPage: 100 } as any,
          { skipErrorHandler: true }
        );
        const workers = (res.items || [])
          .filter((item) => item.state === 'ready')
          .map((item) => ({ label: item.name, value: item.id as number }));
        workerId = workers[0]?.value;
        setBrowse((prev) => ({ ...prev, workers, workerId }));
      } catch (error) {
        return;
      }
    }
    listDirectory(form.getFieldValue('local_path') || '', workerId);
  };

  const handleSearch = (value: string) => {
    listDirectory(value, browse.workerId);
  };

  const handleWorkerChange = (workerId: number) => {
    setBrowse((prev) => ({ ...prev, workerId, options: [] }));
    listDirectory(form.getFieldValue('local_path') || '', workerId);
  };

  const handleSelect = (value: string, option: any) => {
    form.setFieldValue('local_path', value);
    if (option?.isFinal) {
      localPathCache.current = value;
      setBrowse((prev) => ({ ...prev, open: false }));
      onValuesChange?.({ local_path: value }, form.getFieldsValue());
      return;
    }
    // A directory to go into: show what is inside it.
    listDirectory(value, browse.workerId);
    setTimeout(() => setBrowse((prev) => ({ ...prev, open: true })), 0);
  };

  const pathOptions = browse.options.map((option) => ({
    value: option.value,
    isFinal: option.isFinal,
    label: (
      <Flex justify="space-between" align="center" gap={8}>
        <span>{option.name}</span>
        {option.isModel && (
          <Tag color="success" variant="filled" style={{ marginInlineEnd: 0 }}>
            {intl.formatMessage({ id: 'models.localpath.modelTag' })}
          </Tag>
        )}
      </Flex>
    )
  }));

  return (
    <>
      {browse.workers.length > 1 && (
        <SealSelect
          label={intl.formatMessage({ id: 'models.localpath.browseWorker' })}
          value={browse.workerId}
          options={browse.workers}
          onChange={handleWorkerChange}
        ></SealSelect>
      )}
      <Form.Item<FormData>
        name="local_path"
        key="local_path"
        rules={[
          {
            required: true,
            message: getRuleMessage('input', 'models.form.filePath')
          }
        ]}
      >
        <AutoComplete
          allowClear
          required
          open={browse.open}
          options={pathOptions}
          onFocus={handleFocus}
          onBlur={handleOnBlur}
          onSearch={handleSearch}
          onSelect={handleSelect}
          onOpenChange={(open: boolean) =>
            setBrowse((prev) => ({ ...prev, open }))
          }
          label={intl.formatMessage({ id: 'models.form.filePath' })}
          description={<TooltipList list={localPathTipsList}></TooltipList>}
        ></AutoComplete>
      </Form.Item>
    </>
  );
};

export default LocalPathForm;
