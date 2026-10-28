import GPUStackLogo from '@/assets/images/small-logo-200x200.png';
import React from 'react';

/** The mark shown beside a deployment in route-target and usage selectors. */
const DeploymentLogo: React.FC<{ style?: React.CSSProperties }> = ({
  style
}) => (
  <img
    src={GPUStackLogo}
    alt="deployment logo"
    style={{ width: 16, height: 16, ...style }}
  />
);

export default DeploymentLogo;
