"""Rotational cutting-envelope union and radial boundary evaluation."""
import numpy as np
from numba import njit

@njit(cache=False, nogil=True)
def _disk_union(centers, radii, directions, nominal):
    out = np.full(len(directions), nominal)
    for j in range(len(centers)):
        x, y, r = (centers[j, 0], centers[j, 1], radii[j])
        norm2 = x * x + y * y
        upper = np.sqrt(norm2) + r
        if upper < nominal - 1e-13:
            continue
        origin_inside = norm2 <= r * r
        for k in range(len(directions)):
            current = out[k]
            if upper < current - 1e-13:
                continue
            along = x * directions[k, 0] + y * directions[k, 1]
            if origin_inside and current * current - 2 * current * along + norm2 > r * r + 1e-12:
                continue
            disc = r * r - (norm2 - along * along)
            if disc >= 0:
                candidate = along + np.sqrt(disc)
                if candidate > current:
                    out[k] = candidate
    return out

def outline_fast(result, depths, angles=4608):
    """Accumulate swept disks and evaluate the final radial profile (Eqs. 23-26)."""
    depths = np.asarray(depths, dtype=float)
    if depths.ndim != 1 or not np.isfinite(depths).all() or len(depths) == 0 or (depths < 0).any() or (depths > result['geometry']['H']).any():
        raise ValueError('finite section depths inside the plate required')
    if angles < 3 or int(angles) != angles:
        raise ValueError('at least three integer angular samples required')
    g = result['geometry']
    u = result['u']
    zc = result['z_corner']
    radius = g['radius']
    tangent = np.tan(np.deg2rad(g['point_angle'] / 2))
    tip = radius / tangent
    theta = np.linspace(0, 2 * np.pi, int(angles), endpoint=False)
    directions = np.c_[np.cos(theta), np.sin(theta)]
    rows = []
    for depth in depths:
        radii = np.where(zc >= depth, radius, np.where((zc < depth) & (zc + tip >= depth), (zc + tip - depth) * tangent, np.nan))
        keep = np.isfinite(radii) & (radii >= 0)
        centers, rr = (-u[keep], radii[keep])
        cross = np.flatnonzero((zc[:-1] > depth) & (zc[1:] < depth))
        if len(cross):
            j = cross[0]
            fraction = (zc[j] - depth) / (zc[j] - zc[j + 1])
            centers = np.vstack([centers, -(u[j] + fraction * (u[j + 1] - u[j]))])
            rr = np.r_[rr, radius]
        disks = np.unique(np.c_[centers, rr], axis=0)
        rows.append(_disk_union(np.ascontiguousarray(disks[:, :2]), np.ascontiguousarray(disks[:, 2]), directions, radius))
    rho = np.asarray(rows)
    return (theta, rho)
