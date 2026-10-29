import { useIntl } from '@umijs/max';
import { Alert, Button } from 'antd';
import React from 'react';
import usePasteCommand from './use-paste-command';

/**
 * A visible way in to the paste flow, for a form that would otherwise leave it
 * to a small link beside the parameters list further down.
 */
const PasteCommandEntry: React.FC = () => {
  const intl = useIntl();
  const { openModal, modal } = usePasteCommand();

  return (
    <>
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        title={intl.formatMessage({ id: 'models.form.pasteCommand.entry' })}
        action={
          <Button size="small" type="primary" onClick={openModal}>
            {intl.formatMessage({ id: 'models.form.pasteCommand' })}
          </Button>
        }
      />
      {modal}
    </>
  );
};

export default PasteCommandEntry;
