import { useIntl } from '@umijs/max';
import { Alert, Button, Flex, Input, Modal, Typography } from 'antd';
import _ from 'lodash';
import React, { useState } from 'react';
import {
  LaunchCommandResult,
  parseLaunchCommand
} from '../utils/launch-command';

interface PasteCommandModalProps {
  open: boolean;
  onCancel: () => void;
  onApply: (parsed: LaunchCommandResult) => void;
}

const PasteCommandModal: React.FC<PasteCommandModalProps> = ({
  open,
  onCancel,
  onApply
}) => {
  const intl = useIntl();
  const [text, setText] = useState('');
  const [parsed, setParsed] = useState<LaunchCommandResult>(() =>
    parseLaunchCommand('')
  );

  const handleChange = (value: string) => {
    setText(value);
    setParsed(parseLaunchCommand(value));
  };

  const handleCancel = () => {
    setText('');
    setParsed(parseLaunchCommand(''));
    onCancel();
  };

  const handleApply = () => {
    onApply(parsed);
    setText('');
    setParsed(parseLaunchCommand(''));
  };

  const envEntries = Object.entries(parsed.env);
  const t = (id: string, values?: Record<string, any>) =>
    intl.formatMessage({ id: `models.form.pasteCommand.${id}` }, values);

  const renderLines = (lines: string[]) => (
    <pre
      style={{
        margin: '4px 0 0',
        padding: '8px 12px',
        maxHeight: 160,
        overflow: 'auto',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-all',
        background: 'var(--ant-color-fill-quaternary)',
        borderRadius: 'var(--ant-border-radius)'
      }}
    >
      {lines.join('\n')}
    </pre>
  );

  return (
    <Modal
      open={open}
      title={t('title')}
      width={720}
      onCancel={handleCancel}
      destroyOnHidden
      footer={
        <Flex justify="flex-end" gap={8}>
          <Button onClick={handleCancel}>
            {intl.formatMessage({ id: 'common.button.cancel' })}
          </Button>
          <Button
            type="primary"
            disabled={!!parsed.error}
            onClick={handleApply}
          >
            {t('apply')}
          </Button>
        </Flex>
      }
    >
      <Flex vertical gap={12}>
        <Typography.Text type="secondary">{t('tips')}</Typography.Text>
        <Input.TextArea
          autoFocus
          rows={10}
          value={text}
          spellCheck={false}
          placeholder={t('placeholder')}
          style={{ fontFamily: 'var(--ant-font-family-code, monospace)' }}
          onChange={(e) => handleChange(e.target.value)}
        />
        {parsed.error && parsed.error !== 'empty' && (
          <Alert type="error" showIcon message={t(`error.${parsed.error}`)} />
        )}
        {!parsed.error && (
          <Flex vertical gap={8}>
            {parsed.model && (
              <Typography.Text type="secondary">
                {t('model', { model: parsed.model })}
              </Typography.Text>
            )}
            {!!envEntries.length && (
              <div>
                <Typography.Text strong>
                  {t('env', { count: envEntries.length })}
                </Typography.Text>
                {renderLines(envEntries.map(([k, v]) => `${k}=${v}`))}
              </div>
            )}
            {!!parsed.parameters.length && (
              <div>
                <Typography.Text strong>
                  {t('params', { count: parsed.parameters.length })}
                </Typography.Text>
                {renderLines(parsed.parameters)}
              </div>
            )}
            {!!parsed.dropped.length && (
              <div>
                <Typography.Text strong>{t('skipped')}</Typography.Text>
                {renderLines(
                  parsed.dropped.map(
                    (d) =>
                      `${d.flag}${d.value ? ` ${d.value}` : ''}  — ${t(
                        `skipped.${d.reason}`
                      )}`
                  )
                )}
              </div>
            )}
            {!!parsed.variables.length && (
              <Alert
                type="warning"
                showIcon
                message={t('variables', {
                  names: _.uniq(parsed.variables).join(', ')
                })}
              />
            )}
          </Flex>
        )}
        {parsed.error === 'empty' && (
          <Typography.Text type="secondary">{t('error.empty')}</Typography.Text>
        )}
      </Flex>
    </Modal>
  );
};

export default PasteCommandModal;
