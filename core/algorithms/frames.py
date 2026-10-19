"""Force coordinate transformations using the sensor installation axes."""
import numpy as np

def rotation(value, name):
    r = np.asarray(value, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError(f'{name}: expected a finite 3 by 3 rotation')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-08, rtol=0):
        raise ValueError(f'{name}: axes must be orthonormal')
    if not np.isclose(np.linalg.det(r), 1, atol=1e-08, rtol=0):
        raise ValueError(f'{name}: rotation determinant must equal +1')
    return r

def work_to_sensor_reference(work_force, work_from_sensor):
    """Transform three-component force vectors from work to sensor coordinates [N]."""
    force = np.asarray(work_force, dtype=float)
    if force.shape[-1] != 3:
        raise ValueError('Expected three work-frame force components')
    return force @ rotation(work_from_sensor, 'work_from_sensor')


def sensor_to_work(force_sensor, rotation_base_from_work, rotation_base_from_sensor):
    R=rotation(rotation_base_from_work,'work').T@rotation(rotation_base_from_sensor,'sensor')
    return np.asarray(force_sensor)@R.T
