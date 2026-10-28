import { queryModelsList } from '@/pages/llmodels/apis';
import { ListItem as ModelListItem } from '@/pages/llmodels/config/types';
import { useIntl } from '@umijs/max';
import _ from 'lodash';
import React, { useState } from 'react';
import styled from 'styled-components';

const OptionWrapper = styled.span`
  display: flex;
  align-items: center;
  gap: 8px;
`;

type EmptyObject = Record<never, never>;
type CascaderOption<T extends object = EmptyObject> = {
  label: React.ReactNode;
  value: string | number;
  parent?: boolean;
  disabled?: boolean;
  index?: number;
  data?: {
    [key: string]: any;
    parentId?: string | number;
  };
  children?: CascaderOption<T>[];
} & Partial<T>;

const useTargetSourceModels = () => {
  const intl = useIntl();
  const [sourceModels, setSourceModels] = useState<CascaderOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchSourceModels = async (params?: any) => {
    setLoading(true);
    try {
      const models = await queryModelsList({ page: -1, ...params });

      const modelsList = [
        {
          label: intl.formatMessage({ id: 'menu.models.deployment' }),
          value: 'deployments',
          parent: false,
          isParent: true,
          children: _.uniqBy(
            models.items?.flatMap?.((model: ModelListItem) => {
              const items: any[] = [
                {
                  label: model.name,
                  value: model.id,
                  data: {
                    model_id: model.id,
                    parentId: 'deployments'
                  },
                  source: 'deployment'
                }
              ];
              model.lora_list?.forEach?.((lora) => {
                const loraName = lora.lora_name || lora.lora_repo_name;
                if (!loraName) {
                  return;
                }
                items.push({
                  label: loraName,
                  value: `${model.id}_lora_${loraName}`,
                  data: {
                    model_id: model.id,
                    parentId: 'deployments',
                    overridden_model_name: loraName
                  },
                  source: 'deployment',
                  isLora: true
                });
              });
              return items;
            }) || [],
            'value'
          )
        }
      ].filter((group) => group.children && group.children.length > 0);

      setSourceModels(modelsList);
    } catch (error) {
      setSourceModels([]);
    } finally {
      setLoading(false);
    }
  };

  return {
    sourceModels,
    loading,
    fetchSourceModels
  };
};

export default useTargetSourceModels;
