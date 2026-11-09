import { InboxOutlined } from '@ant-design/icons';
import { useIntl } from '@umijs/max';
import { Form, Input, Modal, Upload, message } from 'antd';
import React, { useState } from 'react';
import { createBenchmarkDataset } from '../apis';

type AddDatasetModalProps = {
  open: boolean;
  onCancel: () => void;
  onOk?: () => void;
};

// Upload a corpus from the benchmark form. The server validates the file and
// derives the format, the row count and whether it carries answers, so this
// dialog only has to send the bytes and the name.
const AddDatasetModal: React.FC<AddDatasetModalProps> = ({
  open,
  onCancel,
  onOk
}) => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleCancel = () => {
    form.resetFields();
    setFile(null);
    onCancel();
  };

  const handleOk = async () => {
    const values = await form.validateFields();
    if (!file) {
      message.error(intl.formatMessage({ id: 'benchmark.dataset.fileRequired' }));
      return;
    }
    setSubmitting(true);
    try {
      const content = await file.text();
      await createBenchmarkDataset({
        data: {
          name: values.name,
          description: values.description,
          content
        }
      });
      message.success(intl.formatMessage({ id: 'benchmark.dataset.added' }));
      form.resetFields();
      setFile(null);
      onOk?.();
      onCancel();
    } catch (error: any) {
      message.error(
        error?.response?.data?.message ||
          error?.message ||
          intl.formatMessage({ id: 'benchmark.dataset.addFailed' })
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title={intl.formatMessage({ id: 'benchmark.dataset.add' })}
      open={open}
      onCancel={handleCancel}
      onOk={handleOk}
      confirmLoading={submitting}
      destroyOnClose
      width={520}
    >
      <Form form={form} layout="vertical" preserve={false}>
        <Form.Item
          name="name"
          label={intl.formatMessage({ id: 'common.table.name' })}
          rules={[{ required: true }]}
        >
          <Input maxLength={64} />
        </Form.Item>
        <Form.Item
          name="description"
          label={intl.formatMessage({ id: 'common.table.description' })}
        >
          <Input.TextArea rows={2} maxLength={200} />
        </Form.Item>
        <Form.Item
          label={intl.formatMessage({ id: 'benchmark.dataset.file' })}
          required
          extra={intl.formatMessage({ id: 'benchmark.dataset.fileHint' })}
        >
          <Upload.Dragger
            accept=".json,.jsonl"
            maxCount={1}
            beforeUpload={(next) => {
              setFile(next as unknown as File);
              return false;
            }}
            onRemove={() => setFile(null)}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              {intl.formatMessage({ id: 'benchmark.dataset.fileDrop' })}
            </p>
          </Upload.Dragger>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddDatasetModal;
