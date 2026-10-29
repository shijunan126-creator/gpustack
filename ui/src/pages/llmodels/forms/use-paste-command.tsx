import { Form } from 'antd';
import { useState } from 'react';
import { useFormContext } from '../config/form-context';
import {
  LaunchCommandResult,
  mergeLaunchCommand
} from '../utils/launch-command';
import PasteCommandModal from './paste-command-modal';

/**
 * The paste-a-launch-command flow for one parameter/environment pair: the
 * modal, and what applying it does to the form. The model's own pair lives at
 * the top level; a PD role's under `['roles', index]`.
 *
 * Writes parameters and environment together, since a launch script carries
 * both and one without the other is half a configuration.
 */
const usePasteCommand = (namePrefix?: (string | number)[]) => {
  const form = Form.useFormInstance();
  const { onValuesChange } = useFormContext();
  const [open, setOpen] = useState(false);
  const path = (field: string) => (namePrefix ? [...namePrefix, field] : field);

  const handleApply = (parsed: LaunchCommandResult) => {
    const merged = mergeLaunchCommand(
      {
        parameters: form.getFieldValue(path('backend_parameters')),
        env: form.getFieldValue(path('env'))
      },
      parsed
    );
    form.setFieldValue(path('backend_parameters'), merged.parameters);
    form.setFieldValue(path('env'), merged.env);
    setOpen(false);
    onValuesChange?.({}, form.getFieldsValue());
  };

  return {
    openModal: () => setOpen(true),
    modal: (
      <PasteCommandModal
        open={open}
        onCancel={() => setOpen(false)}
        onApply={handleApply}
      />
    )
  };
};

export default usePasteCommand;
