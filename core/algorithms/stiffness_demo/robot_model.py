from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .coupling import compute_axis_radial_coupling


MDH_PARAMS = np.array(
    [
        [0.0, 0.0, 0.0, 0.670],
        [-np.pi / 2, 0.312, -np.pi / 2, 0.0],
        [0.0, 1.075, 0.0, 0.0],
        [-np.pi / 2, 0.225, 0.0, 1.280],
        [np.pi / 2, 0.0, 0.0, 0.0],
        [-np.pi / 2, 0.0, 0.0, 0.215],
    ],
    dtype=float,
)

JOINT_LIMITS_MODEL_RAD = np.array(
    [
        [-3.2, 3.2],
        [-1.0, 1.3],
        [-2.3, 3.0],
        [-6.2, 6.2],
        [-2.0, 2.0],
        [-6.2, 6.2],
    ],
    dtype=float,
)

JOINT_STIFFNESS_MODEL = np.diag([933941.7645122126, 4129256.136278848, 1907534.0914646569, 233204.60264753376, 404358.8275329604, 79618.15940385307])
# Joint stiffness in the MDH model basis [N m/rad].


def fanuc_wpr_deg_to_rotation(wpr_deg: Iterable[float]) -> np.ndarray:
    """Convert FANUC W/P/R degrees to a reference-to-tool rotation matrix.

    FANUC WPR uses fixed-axis rotations W about X, then P about Y, then R
    about Z.  For column vectors the composed matrix is Rz(R) Ry(P) Rx(W).
    Its columns are the Tool X/Y/Z axes expressed in the reference frame.
    """
    wpr = np.asarray(list(wpr_deg), dtype=float)
    if wpr.shape != (3,) or not np.isfinite(wpr).all():
        raise ValueError("wpr_deg must contain 3 finite values")
    w, p, r = np.deg2rad(wpr)
    cw, sw = np.cos(w), np.sin(w)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cw, -sw], [0.0, sw, cw]], dtype=float
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float
    )
    rotation_z = np.array(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )
    return rotation_z @ rotation_y @ rotation_x


TOOL_WPR_DEG = np.array([-90.784, 0.204, -1.167], dtype=float)
TOOL_ROTATION_MATRIX = fanuc_wpr_deg_to_rotation(TOOL_WPR_DEG)
TOOL_CENTER_OFFSET_M = np.array([0.006875, 0.255313, 0.319324], dtype=float)


@dataclass(frozen=True)
class IkSolution:
    converged: bool
    model_joints_rad: np.ndarray
    raw_joints_deg: np.ndarray
    position_error_m: float
    direction_error_rad: float
    iterations: int
    status: str


@dataclass(frozen=True)
class AxisHeightSolution:
    """Solution of a TCP-height plus one tool-axis direction constraint."""

    converged: bool
    model_joints_rad: np.ndarray
    raw_joints_deg: np.ndarray
    height_error_m: float
    direction_error_rad: float
    iterations: int
    status: str


def raw_to_model_joint_transform() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def transform_raw_stiffness_to_model(raw_joint_stiffness: np.ndarray) -> np.ndarray:
    transform = raw_to_model_joint_transform()
    inverse = np.linalg.inv(transform)
    return inverse.T @ np.asarray(raw_joint_stiffness, dtype=float) @ inverse


JOINT_STIFFNESS_RAW = raw_to_model_joint_transform().T @ JOINT_STIFFNESS_MODEL @ raw_to_model_joint_transform()


class RobotModel:
    def __init__(
        self,
        *,
        mdh_params: np.ndarray = MDH_PARAMS,
        raw_joint_stiffness: np.ndarray | None = None,
        model_joint_stiffness: np.ndarray | None = None,
        joint_limits_model_rad: np.ndarray = JOINT_LIMITS_MODEL_RAD,
        tool_rotation_matrix: np.ndarray = TOOL_ROTATION_MATRIX,
        tool_center_offset_m: np.ndarray = TOOL_CENTER_OFFSET_M,
    ) -> None:
        self.mdh = np.asarray(mdh_params, dtype=float)
        if raw_joint_stiffness is not None and model_joint_stiffness is not None:
            raise ValueError("Specify exactly one stiffness coordinate basis")
        stiffness = (transform_raw_stiffness_to_model(raw_joint_stiffness) if raw_joint_stiffness is not None
                     else np.asarray(JOINT_STIFFNESS_MODEL if model_joint_stiffness is None else model_joint_stiffness, dtype=float))
        if stiffness.shape != (6, 6) or not np.isfinite(stiffness).all() or not np.allclose(stiffness, stiffness.T) or np.linalg.eigvalsh(stiffness).min() <= 0:
            raise ValueError("Joint stiffness must be a finite symmetric positive definite 6x6 matrix")
        self.joint_stiffness_model = stiffness.copy()
        self.joint_limits_model_rad = np.asarray(joint_limits_model_rad, dtype=float)
        self.tool_rotation_matrix = np.asarray(tool_rotation_matrix, dtype=float)
        self.tool_center_offset_m = np.asarray(tool_center_offset_m, dtype=float)

    def raw_deg_to_model_rad(self, raw_joints_deg: Iterable[float]) -> np.ndarray:
        raw_rad = np.deg2rad(_six_values(raw_joints_deg, "raw_joints_deg"))
        model = raw_to_model_joint_transform() @ raw_rad
        return np.asarray(model, dtype=float)

    def model_rad_to_raw_deg(self, model_joints_rad: Iterable[float]) -> list[float]:
        model_rad = _six_values(model_joints_rad, "model_joints_rad")
        raw_rad = np.linalg.inv(raw_to_model_joint_transform()) @ model_rad
        return [float(value) for value in np.rad2deg(raw_rad)]

    def forward_kinematics_all(self, model_joints_rad: Iterable[float]) -> list[np.ndarray]:
        q = _six_values(model_joints_rad, "model_joints_rad")
        transforms: list[np.ndarray] = []
        current = np.eye(4, dtype=float)
        for index in range(6):
            alpha, a, theta_offset, d = self.mdh[index]
            current = current @ _mdh_transform(alpha, a, q[index] + theta_offset, d)
            transforms.append(current)
        return transforms

    def forward_kinematics(self, model_joints_rad: Iterable[float]) -> np.ndarray:
        flange = self.forward_kinematics_all(model_joints_rad)[-1]
        rotation_flange = flange[:3, :3]
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = rotation_flange @ self.tool_rotation_matrix
        pose[:3, 3] = flange[:3, 3] + rotation_flange @ self.tool_center_offset_m
        return pose

    def compute_jacobian(self, model_joints_rad: Iterable[float]) -> np.ndarray:
        transforms = self.forward_kinematics_all(model_joints_rad)
        flange = transforms[-1]
        p_tcp = flange[:3, 3] + flange[:3, :3] @ self.tool_center_offset_m
        jacobian = np.zeros((6, 6), dtype=float)
        for index, transform in enumerate(transforms):
            z_axis = transform[:3, 2]
            p_joint = transform[:3, 3]
            jacobian[:3, index] = np.cross(z_axis, p_tcp - p_joint)
            jacobian[3:, index] = z_axis
        return jacobian

    def cartesian_stiffness_tool(self, model_joints_rad: Iterable[float]) -> np.ndarray:
        q = _six_values(model_joints_rad, "model_joints_rad")
        jacobian = self.compute_jacobian(q)
        jv = jacobian[:3, :]
        compliance_world = jv @ np.linalg.pinv(self.joint_stiffness_model) @ jv.T
        stiffness_world = _symmetrize_positive(np.linalg.pinv(compliance_world))
        rotation_tool = self.forward_kinematics(q)[:3, :3]
        stiffness_tool = rotation_tool.T @ stiffness_world @ rotation_tool
        return _symmetrize_positive(stiffness_tool)

    def evaluate_raw_joints(self, raw_joints_deg: Iterable[float]) -> dict:
        raw = _six_values(raw_joints_deg, "raw_joints_deg")
        model = self.raw_deg_to_model_rad(raw)
        pose = self.forward_kinematics(model)
        stiffness_tool = self.cartesian_stiffness_tool(model)
        coupling = compute_axis_radial_coupling(stiffness_tool)
        return {
            "pose": {
                "raw_joints_deg": [float(value) for value in raw],
                "model_joints_deg": [float(value) for value in np.rad2deg(model)],
                "tcp_position_base_mm": [float(value) for value in pose[:3, 3] * 1000.0],
                "tool_wpr_deg": TOOL_WPR_DEG.tolist(),
                "tool_x_base": [float(value) for value in pose[:3, 0]],
                "tool_y_base": [float(value) for value in pose[:3, 1]],
                "tool_z_base": [float(value) for value in pose[:3, 2]],
                "base_T_tool": pose.tolist(),
            },
            "stiffness_tool_n_per_m": stiffness_tool.tolist(),
            "coupling": coupling,
        }

    def solve_ik(
        self,
        target_position_m: Iterable[float],
        target_tool_z_base: Iterable[float],
        *,
        initial_model_rad: Iterable[float] | None = None,
        max_iter: int = 120,
        tol_pos_m: float = 1.0e-4,
        tol_dir_rad: float = 1.0e-3,
        damping: float = 0.01,
        step_scale: float = 0.55,
        max_step_rad: float = 0.35,
    ) -> IkSolution:
        target_pos = np.asarray(list(target_position_m), dtype=float)
        if target_pos.shape != (3,):
            raise ValueError("target_position_m must contain 3 values")
        target_dir = _unit_vector(target_tool_z_base, "target_tool_z_base")
        if initial_model_rad is None:
            q = self._default_ik_seed(target_pos)
        else:
            q = _six_values(initial_model_rad, "initial_model_rad").astype(float)
        q = self._clip_model_joints(q)

        pos_error_norm = float("inf")
        dir_error_rad = float("inf")
        for iteration in range(1, int(max_iter) + 1):
            pose = self.forward_kinematics(q)
            current_pos = pose[:3, 3]
            current_dir = _unit_vector(pose[:3, 2], "current_tool_z")
            err_pos = target_pos - current_pos
            err_rot = np.cross(current_dir, target_dir)
            pos_error_norm = float(np.linalg.norm(err_pos))
            dir_dot = float(np.clip(np.dot(current_dir, target_dir), -1.0, 1.0))
            dir_error_rad = float(np.arccos(dir_dot))
            if pos_error_norm < tol_pos_m and dir_error_rad < tol_dir_rad:
                return self._ik_solution(True, q, pos_error_norm, dir_error_rad, iteration, "converged")

            error = np.hstack([err_pos, err_rot])
            jacobian = self.compute_jacobian(q)
            system = jacobian @ jacobian.T + (float(damping) ** 2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(system, error)
            norm = float(np.linalg.norm(delta))
            if norm > max_step_rad:
                delta = delta / norm * max_step_rad
            q = self._clip_model_joints(q + step_scale * delta)

        return self._ik_solution(False, q, pos_error_norm, dir_error_rad, int(max_iter), "not_converged")

    def solve_pose_ik(
        self,
        target_position_m: Iterable[float],
        target_rotation_base: np.ndarray,
        *,
        initial_model_rad: Iterable[float] | None = None,
        max_iter: int = 160,
        tol_pos_m: float = 1.0e-8,
        tol_rot_rad: float = 1.0e-7,
        damping: float = 0.002,
        step_scale: float = 0.5,
        max_step_rad: float = 0.08,
    ) -> IkSolution:
        """Solve a complete TCP position and orientation target.

        ``direction_error_rad`` in the returned ``IkSolution`` contains the
        complete orientation error angle for this method.
        """
        target_pos = np.asarray(list(target_position_m), dtype=float)
        target_rotation = np.asarray(target_rotation_base, dtype=float)
        if target_pos.shape != (3,) or not np.isfinite(target_pos).all():
            raise ValueError("target_position_m must contain 3 finite values")
        if target_rotation.shape != (3, 3) or not np.isfinite(target_rotation).all():
            raise ValueError("target_rotation_base must be a finite 3x3 matrix")
        if not np.allclose(target_rotation.T @ target_rotation, np.eye(3), atol=1.0e-7):
            raise ValueError("target_rotation_base must be orthonormal")
        if np.linalg.det(target_rotation) < 0.0:
            raise ValueError("target_rotation_base must be a proper rotation")
        if initial_model_rad is None:
            q = self._default_ik_seed(target_pos)
        else:
            q = _six_values(initial_model_rad, "initial_model_rad").astype(float)
        q = self._clip_model_joints(q)

        pos_error_norm = float("inf")
        rot_error_rad = float("inf")
        for iteration in range(1, int(max_iter) + 1):
            pose = self.forward_kinematics(q)
            err_pos = target_pos - pose[:3, 3]
            err_rot = _rotation_vector(target_rotation @ pose[:3, :3].T)
            pos_error_norm = float(np.linalg.norm(err_pos))
            rot_error_rad = float(np.linalg.norm(err_rot))
            if pos_error_norm < tol_pos_m and rot_error_rad < tol_rot_rad:
                return self._ik_solution(
                    True,
                    q,
                    pos_error_norm,
                    rot_error_rad,
                    iteration,
                    "converged",
                )

            error = np.hstack([err_pos, err_rot])
            jacobian = self.compute_jacobian(q)
            system = jacobian @ jacobian.T + (float(damping) ** 2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(system, error)
            norm = float(np.linalg.norm(delta))
            if norm > max_step_rad:
                delta = delta / norm * max_step_rad
            q = self._clip_model_joints(q + step_scale * delta)

        return self._ik_solution(
            False,
            q,
            pos_error_norm,
            rot_error_rad,
            int(max_iter),
            "not_converged",
        )

    def solve_axis_height_constraint(
        self,
        target_height_m: float,
        target_tool_axis_base: Iterable[float],
        *,
        tool_axis_index: int = 0,
        initial_model_rad: Iterable[float] | None = None,
        max_iter: int = 120,
        tol_height_m: float = 5.0e-4,
        tol_dir_rad: float = np.deg2rad(0.2),
        damping: float = 0.01,
        step_scale: float = 0.65,
        max_step_rad: float = 0.35,
    ) -> AxisHeightSolution:
        """Keep TCP Base-Z and one tool-axis direction while other DOFs vary.

        The constraint has three independent components: one TCP-height
        constraint and two direction constraints.  Starting from different
        seeds therefore explores the remaining posture freedom instead of
        fixing TCP X/Y or the rotation about the locked tool axis.
        """
        axis_index = int(tool_axis_index)
        if axis_index not in (0, 1, 2):
            raise ValueError("tool_axis_index must be 0 (X), 1 (Y), or 2 (Z)")
        target_height = float(target_height_m)
        if not np.isfinite(target_height):
            raise ValueError("target_height_m must be finite")
        target_dir = _unit_vector(target_tool_axis_base, "target_tool_axis_base")
        if initial_model_rad is None:
            q = np.zeros(6, dtype=float)
        else:
            q = _six_values(initial_model_rad, "initial_model_rad").astype(float)
        q = self._clip_model_joints(q)

        height_error = float("inf")
        dir_error_rad = float("inf")
        for iteration in range(1, int(max_iter) + 1):
            pose = self.forward_kinematics(q)
            current_dir = _unit_vector(
                pose[:3, axis_index], f"current_tool_axis_{axis_index}"
            )
            err_height = target_height - float(pose[2, 3])
            err_rot = np.cross(current_dir, target_dir)
            height_error = abs(float(err_height))
            dir_dot = float(np.clip(np.dot(current_dir, target_dir), -1.0, 1.0))
            dir_error_rad = float(np.arccos(dir_dot))
            if height_error < tol_height_m and dir_error_rad < tol_dir_rad:
                return self._axis_height_solution(
                    True,
                    q,
                    height_error,
                    dir_error_rad,
                    iteration,
                    "converged",
                )

            jacobian = self.compute_jacobian(q)
            constraint_jacobian = np.vstack([jacobian[2:3, :], jacobian[3:, :]])
            error = np.hstack([err_height, err_rot])
            system = (
                constraint_jacobian @ constraint_jacobian.T
                + (float(damping) ** 2) * np.eye(4)
            )
            delta = constraint_jacobian.T @ np.linalg.solve(system, error)
            norm = float(np.linalg.norm(delta))
            if norm > max_step_rad:
                delta = delta / norm * max_step_rad
            q = self._clip_model_joints(q + step_scale * delta)

        return self._axis_height_solution(
            False,
            q,
            height_error,
            dir_error_rad,
            int(max_iter),
            "not_converged",
        )

    def _default_ik_seed(self, target_position_m: np.ndarray) -> np.ndarray:
        j1 = float(np.arctan2(target_position_m[1], target_position_m[0]))
        return np.array([j1, 0.1, 0.1, 0.0, 0.1, 0.0], dtype=float)

    def _clip_model_joints(self, model_joints_rad: np.ndarray) -> np.ndarray:
        return np.clip(model_joints_rad, self.joint_limits_model_rad[:, 0], self.joint_limits_model_rad[:, 1])

    def _ik_solution(
        self,
        converged: bool,
        model_joints_rad: np.ndarray,
        position_error_m: float,
        direction_error_rad: float,
        iterations: int,
        status: str,
    ) -> IkSolution:
        return IkSolution(
            converged=bool(converged),
            model_joints_rad=np.asarray(model_joints_rad, dtype=float).copy(),
            raw_joints_deg=np.asarray(self.model_rad_to_raw_deg(model_joints_rad), dtype=float),
            position_error_m=float(position_error_m),
            direction_error_rad=float(direction_error_rad),
            iterations=int(iterations),
            status=status,
        )

    def _axis_height_solution(
        self,
        converged: bool,
        model_joints_rad: np.ndarray,
        height_error_m: float,
        direction_error_rad: float,
        iterations: int,
        status: str,
    ) -> AxisHeightSolution:
        return AxisHeightSolution(
            converged=bool(converged),
            model_joints_rad=np.asarray(model_joints_rad, dtype=float).copy(),
            raw_joints_deg=np.asarray(
                self.model_rad_to_raw_deg(model_joints_rad), dtype=float
            ),
            height_error_m=float(height_error_m),
            direction_error_rad=float(direction_error_rad),
            iterations=int(iterations),
            status=status,
        )


def _mdh_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _six_values(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.shape != (6,):
        raise ValueError(f"{name} must contain 6 values")
    return array


def _unit_vector(values: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(list(values), dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain 3 values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector of a proper 3x3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    sine_vector = 0.5 * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    )
    sine = float(np.linalg.norm(sine_vector))
    angle = float(np.arctan2(sine, cosine))
    if sine > 1.0e-10:
        return sine_vector * (angle / sine)
    if cosine > 0.0:
        return sine_vector
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
    axis /= np.linalg.norm(axis)
    return axis * angle


def _symmetrize_positive(matrix: np.ndarray) -> np.ndarray:
    sym = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigval, eigvec = np.linalg.eigh(sym)
    eigval = np.clip(eigval, 1.0e-12, None)
    return eigvec @ np.diag(eigval) @ eigvec.T
