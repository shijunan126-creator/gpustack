import { useIntl } from '@umijs/max';
import { Flex, Statistic, Switch, Table, Tag, Tooltip } from 'antd';
import React, { useEffect, useState } from 'react';
import { queryBenchmarkAccuracy } from '../../apis';
import { useDetailContext } from '../../config/detail-context';
import { AccuracySampleItem } from '../../config/types';

// The verdict view of an accuracy run: one score, then the questions behind it.
// Wrong answers are the first thing anyone wants to read, so they can be
// filtered to without loading a different page.
const AccuracySummary: React.FC = () => {
  const intl = useIntl();
  const { id, detailData, refreshToken } = useDetailContext();
  const [samples, setSamples] = useState<AccuracySampleItem[]>([]);
  const [total, setTotal] = useState(0);
  const [onlyIncorrect, setOnlyIncorrect] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!id) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    queryBenchmarkAccuracy(id, {
      only_incorrect: onlyIncorrect,
      limit: 200,
      offset: 0
    })
      .then((res) => {
        if (cancelled) return;
        setSamples(res?.items ?? []);
        setTotal(res?.total ?? 0);
      })
      .catch(() => {
        if (!cancelled) {
          setSamples([]);
          setTotal(0);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id, onlyIncorrect, refreshToken]);

  const scored = detailData?.accuracy_total ?? 0;
  const columns = [
    {
      title: '#',
      dataIndex: 'seq',
      width: 64
    },
    {
      title: intl.formatMessage({ id: 'benchmark.accuracy.question' }),
      dataIndex: 'question',
      ellipsis: true,
      render: (value: string) => (
        <Tooltip title={value}>
          <span>{value}</span>
        </Tooltip>
      )
    },
    {
      title: intl.formatMessage({ id: 'benchmark.accuracy.answer' }),
      dataIndex: 'extracted',
      width: 120,
      render: (value: string, row: AccuracySampleItem) =>
        row.error ? (
          <Tag color="red">
            {intl.formatMessage({ id: 'benchmark.accuracy.errored' })}
          </Tag>
        ) : (
          value || '-'
        )
    },
    {
      title: intl.formatMessage({ id: 'benchmark.accuracy.reference' }),
      dataIndex: 'reference',
      width: 120
    },
    {
      title: intl.formatMessage({ id: 'benchmark.accuracy.verdict' }),
      dataIndex: 'correct',
      width: 110,
      render: (value: boolean) =>
        value ? (
          <Tag color="success">
            {intl.formatMessage({ id: 'benchmark.accuracy.correct' })}
          </Tag>
        ) : (
          <Tag color="error">
            {intl.formatMessage({ id: 'benchmark.accuracy.wrong' })}
          </Tag>
        )
    },
    {
      title: intl.formatMessage({ id: 'benchmark.accuracy.latency' }),
      dataIndex: 'latency_ms',
      width: 110,
      render: (value: number) => (value == null ? '-' : `${value} ms`)
    }
  ];

  return (
    <Flex vertical gap={16}>
      <Flex gap={40} wrap="wrap">
        <Statistic
          title={intl.formatMessage({ id: 'benchmark.accuracy.score' })}
          value={detailData?.accuracy_score ?? 0}
          precision={2}
          suffix="%"
        />
        <Statistic
          title={intl.formatMessage({ id: 'benchmark.accuracy.correctCount' })}
          value={detailData?.accuracy_correct ?? 0}
          suffix={`/ ${scored}`}
        />
        <Statistic
          title={intl.formatMessage({ id: 'benchmark.accuracy.invalid' })}
          value={detailData?.accuracy_invalid ?? 0}
        />
        <Statistic
          title={intl.formatMessage({ id: 'benchmark.accuracy.errored' })}
          value={detailData?.accuracy_errored ?? 0}
        />
        <Statistic
          title={intl.formatMessage({ id: 'benchmark.accuracy.elapsed' })}
          value={detailData?.accuracy_elapsed_seconds ?? 0}
          precision={0}
          suffix="s"
        />
      </Flex>
      <Flex align="center" gap={8}>
        <Switch
          size="small"
          checked={onlyIncorrect}
          onChange={setOnlyIncorrect}
        />
        <span>{intl.formatMessage({ id: 'benchmark.accuracy.onlyWrong' })}</span>
        <span style={{ color: 'var(--ant-color-text-tertiary)' }}>
          {intl.formatMessage(
            { id: 'benchmark.accuracy.showing' },
            { count: samples.length, total }
          )}
        </span>
      </Flex>
      <Table<AccuracySampleItem>
        rowKey="id"
        size="small"
        loading={loading}
        dataSource={samples}
        columns={columns as any}
        pagination={false}
        scroll={{ x: true }}
      />
    </Flex>
  );
};

export default AccuracySummary;
