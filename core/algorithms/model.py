"""Elastic recovery, withdrawal reaction and cutting-envelope trajectory.

The CSV force is a positive magnitude: the signed axial reaction is -F0.
target is d0; u=d0-d; the cutting-envelope center is therefore -u.
"""
import numpy as np
from numba import njit
from .envelope import outline_fast

@njit(cache=False, nogil=True)
def _step(previous, target, hold, eigen):
    """Solve the incremental energy minimization (paper Eq. 21)."""
    w = eigen * (target - previous)
    norm = np.sqrt(w[0] * w[0] + w[1] * w[1])
    if norm <= hold + 1e-12:
        return previous.copy()
    if hold <= 1e-12:
        return target.copy()
    lo = 0.0
    hi = norm / max(hold, 1e-15) / min(eigen[0], eigen[1])
    for _ in range(36):
        a = (lo + hi) / 2
        r = w / (1 + a * eigen)
        if np.sqrt(r[0] * r[0] + r[1] * r[1]) > hold:
            lo = a
        else:
            hi = a
    a = (lo + hi) / 2
    return previous + a * w / (1 + a * eigen)

@njit(cache=False, nogil=True)
def _history(eigen, target, qc, z, recovery_steps, H):
    history = np.zeros((len(z), 2))
    for j in range(1, len(z)):
        if j <= recovery_steps:
            desired = target * (j / recovery_steps)
            hold = H * qc
        else:
            desired = target
            hold = qc * min(H, max(0.0, z[j]))
        history[j] = _step(history[j - 1], desired, hold, eigen)
    return history

def simulate(K, target, qc=63.0, *, overtravel=3.0, recovery_steps=512, withdrawal_steps=2400, H=3.0, radius=5.0, point_angle=140.0):
    """Quasi-static unloading at full support, followed by axial withdrawal.

    Initialization steps discretize the quasi-static unloading path.
    K is in N/mm, target and geometry in mm, qc in N/mm, point_angle in degrees.
    """
    K = np.asarray(K, float)
    target = np.asarray(target, float)
    if K.shape != (2, 2) or target.shape != (2,) or (not np.isfinite(K).all()) or (not np.isfinite(target).all()):
        raise ValueError('finite two-dimensional stiffness and target required')
    if not np.allclose(K, K.T, rtol=0, atol=1e-09):
        raise ValueError('symmetric stiffness required')
    eigen, vectors = np.linalg.eigh(K)
    if eigen.min() <= 0 or qc < 0 or not np.isfinite(qc):
        raise ValueError('positive stiffness and nonnegative finite qc required')
    if not np.isfinite([H,radius,point_angle,overtravel]).all() or H<=0 or radius<=0 or not 0<point_angle<180 or overtravel<0:
        raise ValueError('positive finite geometry and a point angle between 0 and 180 required')
    if recovery_steps < 1 or withdrawal_steps < 1 or int(recovery_steps)!=recovery_steps or int(withdrawal_steps)!=withdrawal_steps:
        raise ValueError('positive integer step counts required')
    recovery_steps,withdrawal_steps=int(recovery_steps),int(withdrawal_steps)
    tip = radius / np.tan(np.deg2rad(point_angle / 2))
    start = H + overtravel - tip
    if start < H:
        raise ValueError('initialization requires full cylindrical support')
    z = np.r_[np.full(recovery_steps + 1, start), np.linspace(start, -tip, withdrawal_steps + 1)[1:]]
    u = _history(eigen, vectors.T @ target, float(qc), z, recovery_steps, H) @ vectors.T
    desired = np.broadcast_to(target, u.shape).copy()
    desired[:recovery_steps + 1] = np.linspace(0, 1, recovery_steps + 1)[:, None] * target
    phase = np.array(['loaded'] + ['fixed_recovery'] * recovery_steps + ['withdrawal'] * withdrawal_steps)
    return dict(K=K, target=target, u=u, z_corner=z, phase=phase,
                reaction=(desired-u)@K.T, target_history=desired,
                recovery_steps=recovery_steps, withdrawal_steps=withdrawal_steps,
                qc_n_per_mm=float(qc),
                geometry=dict(H=H,radius=radius,point_angle=point_angle,tip_height=tip,overtravel=overtravel))

def query(tr, zquery):
    first = tr['recovery_steps']
    z = tr['z_corner'][first:][::-1]
    u = np.column_stack([np.interp(zquery, z, tr['u'][first:, j][::-1]) for j in range(2)])
    return (u, (tr['target'] - u) @ tr['K'].T)


def predict(K, coupling_mm_per_n, axial_force_n, depths_mm, qc=63., *,
            overtravel=3., H=3., radius=5., point_angle=140.,
            recovery_steps=512, withdrawal_steps=2400, angles=4608):
    """Return trajectory and radial envelope from a positive axial-force magnitude."""
    if not np.isfinite(axial_force_n) or axial_force_n<0:
        raise ValueError('axial_force_n is a finite nonnegative force magnitude')
    target = -np.asarray(coupling_mm_per_n, float) * float(axial_force_n)
    trace = simulate(K, target, qc, overtravel=overtravel,H=H,radius=radius,point_angle=point_angle,
                     recovery_steps=recovery_steps,withdrawal_steps=withdrawal_steps)
    theta, rho = outline_fast(trace, depths_mm, angles)
    trace.update(theta=theta, rho=rho, depths_mm=np.asarray(depths_mm, float))
    return trace
