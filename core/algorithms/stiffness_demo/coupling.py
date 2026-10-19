from __future__ import annotations

import math
from typing import Any

import numpy as np


WEAK_THRESHOLD = 0.1
HIGH_THRESHOLD = 0.3


def classify_coupling(
    *,
    max_normalized_stiffness: float,
    rho_radial_per_axial: float,
    weak_threshold: float = WEAK_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
) -> str:
    strongest = max(float(max_normalized_stiffness), float(rho_radial_per_axial))
    if strongest >= float(high_threshold):
        return "high"
    if float(max_normalized_stiffness) < float(weak_threshold) and float(rho_radial_per_axial) < float(weak_threshold):
        return "weak"
    return "moderate"


def compute_axis_radial_coupling(stiffness_matrix: np.ndarray) -> dict[str, Any]:
    stiffness = np.asarray(stiffness_matrix, dtype=float)
    if stiffness.shape != (3, 3):
        raise ValueError("stiffness_matrix must have shape (3, 3)")
    compliance = np.linalg.inv(stiffness)

    unit_response = compliance @ np.array([0.0, 0.0, 1.0], dtype=float)
    radial_response = float(np.linalg.norm(unit_response[:2]))
    axial_response = abs(float(unit_response[2]))
    rho = radial_response / axial_response if axial_response > 0.0 else float("inf")

    c_xz = _normalized_off_diagonal(stiffness, 0, 2)
    c_yz = _normalized_off_diagonal(stiffness, 1, 2)
    denom_radial = math.sqrt(abs(float(stiffness[2, 2] * (stiffness[0, 0] + stiffness[1, 1]))))
    c_radial_z = (
        math.sqrt(float(stiffness[0, 2] ** 2 + stiffness[1, 2] ** 2)) / denom_radial
        if denom_radial > 0.0
        else float("inf")
    )
    max_normalized = max(c_xz, c_yz)

    return {
        "c_xz": float(c_xz),
        "c_yz": float(c_yz),
        "c_radial_z": float(c_radial_z),
        "max_normalized_stiffness": float(max_normalized),
        "compliance_matrix_m_per_n": compliance.tolist(),
        "unit_axial_force_response_m_per_n": unit_response.tolist(),
        "unit_axial_radial_response_m_per_n": radial_response,
        "unit_axial_axial_response_m_per_n": float(unit_response[2]),
        "rho_radial_per_axial_displacement": float(rho),
        "classification": classify_coupling(
            max_normalized_stiffness=max_normalized,
            rho_radial_per_axial=rho,
        ),
    }


def _normalized_off_diagonal(stiffness: np.ndarray, row: int, col: int) -> float:
    denom = math.sqrt(abs(float(stiffness[row, row] * stiffness[col, col])))
    return abs(float(stiffness[row, col])) / denom if denom > 0.0 else float("inf")
