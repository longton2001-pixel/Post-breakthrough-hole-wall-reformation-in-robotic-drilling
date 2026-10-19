"""Cartesian parameters evaluated at the supplied reference configurations."""
from pathlib import Path
import csv
import numpy as np
from .stiffness import work_compliance
from .stiffness_demo.robot_model import RobotModel,fanuc_wpr_deg_to_rotation

def reference_parameters(configurations_path=None,joint_stiffness_path=None):
    if configurations_path is None:
        configurations_path=Path(__file__).resolve().parents[1]/'parameters/reference_configurations.csv'
    with Path(configurations_path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if joint_stiffness_path is None:
        robot=RobotModel()
    else:
        with Path(joint_stiffness_path).open(encoding='utf-8-sig',newline='') as f:stiffness=list(csv.DictReader(f))
        ordered=sorted(stiffness,key=lambda r:int(r['joint']))
        if [int(r['joint']) for r in ordered]!=list(range(1,7)):raise ValueError('Joint stiffness rows 1 through 6 required')
        robot=RobotModel(model_joint_stiffness=np.diag([float(r['stiffness_nm_per_rad']) for r in ordered]))
    result={}
    for row in rows:
        posture=int(row['posture'])
        if posture in result:raise ValueError('Unique reference posture identifiers required')
        q=robot.raw_deg_to_model_rad([float(row[f'joint_{j}_deg']) for j in range(1,7)])
        W=fanuc_wpr_deg_to_rotation([float(row[f'work_{a}_deg']) for a in ['w','p','r']])
        C,K,c=work_compliance(robot.compute_jacobian(q),robot.joint_stiffness_model,W)
        coupling=float(np.linalg.norm(c));stiffness=float(np.linalg.norm(K@c)/coupling)
        result[posture]=dict(C=C,K=K,c=c,coupling_mm_n=coupling,recovery_stiffness_n_mm=stiffness)
    return result
