import { Form } from 'antd';
import { useState } from 'react';
import { useFormContext } from '../config/form-context';
import {
  LaunchCommandResult,
  ParseOptions,
  mergeLaunchCommand,
  sanitizeForPD
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
const usePasteCommand = (
  namePrefix?: (string | number)[],
  parseOptions?: ParseOptions
) => {
  const form = Form.useFormInstance();
  const { onValuesChange } = useFormContext();
  const [open, setOpen] = useState(false);
  // A deployment with roles is disaggregated, whichever list the paste lands
  // in: the model's parameters reach every role through the merge, so the same
  // refusals apply at both levels. Derived here rather than at each call site,
  // because the model-level entry and the role entries reach the same modal.
  //
  // ⚠️ Read the store, not only the watch. `Form.useWatch` publishes its first
  // value a render after mount, and this entry mounts inside a panel that opens
  // long after the switch seeded `roles` — a fresh mount therefore sees the
  // watch's placeholder, not the roles, and would parse as non-PD. The store
  // read is synchronously correct in both cases.
  const watchedRoles = Form.useWatch('roles', { form, preserve: true });
  const roles = Array.isArray(watchedRoles) && watchedRoles.length
    ? watchedRoles
    : form.getFieldValue('roles');
  const pd = !!parseOptions?.pd || (Array.isArray(roles) && roles.length > 0);
  const effectiveOptions: ParseOptions = { ...parseOptions, pd };
  const path = (field: string) => (namePrefix ? [...namePrefix, field] : field);

  const handleApply = (parsed: LaunchCommandResult) => {
    const merged = mergeLaunchCommand(
      {
        parameters: form.getFieldValue(path('backend_parameters')),
        env: form.getFieldValue(path('env'))
      },
      parsed
    );
    // The modal previews under the rules it was parsed with; this is the
    // guarantee. A PD member never gets a connector, a port or a placement
    // from a paste, whatever reached the modal.
    const applied = pd ? sanitizeForPD(merged.parameters, merged.env) : merged;
    form.setFieldValue(path('backend_parameters'), applied.parameters);
    form.setFieldValue(path('env'), applied.env);
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
        parseOptions={effectiveOptions}
      />
    )
  };
};

export default usePasteCommand;
