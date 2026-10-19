"""Whole-hole rigid registration, depth weights and radial error."""
import numpy as np
from scipy.optimize import least_squares

def rotation(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def depth_weights(z):
    z = np.array(z)
    return np.diff(np.r_[z[0], (z[:-1] + z[1:]) / 2, z[-1]]) / (z[-1] - z[0])

def align_wall(points, theta, rho, weights, seed=None):
    """Fit one in-plane translation and rotation shared by all hole sections."""
    origin = np.mean([p.mean(0) for p in points], axis=0)
    centered = [p - origin for p in points]
    directions = np.c_[np.cos(theta), np.sin(theta)]
    pc = (rho[..., None] * directions).mean(axis=(0, 1))

    def residual(x):
        rot = rotation(x[2])
        result = []
        for p, rad, w in zip(centered, rho, weights):
            xy = (p - x[:2]) @ rot
            angle = np.arctan2(xy[:, 1], xy[:, 0])
            e = np.linalg.norm(xy, axis=1) - np.interp(angle, theta, rad, period=2 * np.pi)
            result.append(e * np.sqrt(w / len(p)))
        return np.concatenate(result)
    starts = []
    if seed is not None:
        starts.append(np.r_[seed[:2] - origin, seed[2]])
    _, vm = np.linalg.eigh(np.cov(centered[0].T))
    ma = np.arctan2(vm[1, -1], vm[0, -1])
    xy = rho[0, :, None] * directions
    _, vp = np.linalg.eigh(np.cov(xy.T))
    pa = np.arctan2(vp[1, -1], vp[0, -1])
    for phi in [ma - pa, ma - pa + np.pi]:
        starts.append(np.r_[-pc @ rotation(phi).T, phi])
    best = None
    for start in starts:
        fit = least_squares(residual, start, max_nfev=65, ftol=1e-08, xtol=1e-08, gtol=1e-08)
        loss = float(np.dot(fit.fun, fit.fun))
        if best is None or loss < best[0]:
            best = (loss, fit.x, fit.success)
    return (np.sqrt(best[0]), np.r_[origin + best[1][:2], best[1][2]], bool(best[2]))

def fit_circle(xy):
    xy = np.asarray(xy, float)
    if len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError('circle requires >=3 finite points')
    origin = xy.mean(axis=0)
    p = xy - origin
    if np.linalg.matrix_rank(p) < 2:
        raise ValueError('collinear points')
    a = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
    b = np.sum(p * p, axis=1)
    c = np.linalg.lstsq(a, b, rcond=None)[0][:2]
    opt = least_squares(lambda c: np.linalg.norm(p - c, axis=1) - np.linalg.norm(p - c, axis=1).mean(), c, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    center = opt.x + origin
    r = np.linalg.norm(xy - center, axis=1)
    radius = float(r.mean())
    return (center, radius, r - radius)


def radial_errors(points, theta, rho, transform):
    """One rigid transform for all layers, with periodic angular interpolation."""
    rot = rotation(transform[2])
    return [(lambda xy, rad: np.linalg.norm(xy,axis=1) - np.interp(
        np.arctan2(xy[:,1],xy[:,0]),theta,rad,period=2*np.pi))(
            (np.asarray(p)-transform[:2])@rot, rad)
        for p,rad in zip(points,rho)]

def error_metrics(residuals, weights):
    flat=np.concatenate(residuals)
    mse=float(sum(w*np.mean(e*e) for w,e in zip(weights,residuals)))
    return dict(rms_mm=float(np.sqrt(mse)), max_abs_mm=float(np.max(abs(flat))),
                p95_abs_mm=float(np.quantile(abs(flat),.95)))
