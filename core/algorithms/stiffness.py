"""Joint stiffness conversion between MDH and controller coordinates."""
import numpy as np

def robot_from_config(cfg, *, historical_raw_replay=False):
    from .stiffness_demo.robot_model import RobotModel
    values=np.asarray(cfg['joint_stiffness_nm_per_rad'],dtype=float)
    if values.shape!=(6,) or not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Expected six positive finite joint stiffness values')
    basis=cfg.get('joint_stiffness_basis')
    if basis=='model_mdh':
        return RobotModel(model_joint_stiffness=np.diag(values))
    if basis=='raw_controller' or (basis is None and historical_raw_replay):
        return RobotModel(raw_joint_stiffness=np.diag(values))
    raise ValueError('joint_stiffness_basis must be model_mdh or raw_controller')


def work_compliance(jacobian_base, joint_stiffness, rotation_base_from_work):
    """J [m/rad], joint stiffness [N m/rad] -> work compliance [mm/N]."""
    J=np.asarray(rotation_base_from_work).T@np.asarray(jacobian_base)[:3]
    C=J@np.linalg.solve(joint_stiffness,J.T)*1000.
    return C,np.linalg.inv(C[:2,:2]),C[:2,2]
