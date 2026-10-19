"""Compute recovery trajectories, reaction forces and hole-wall profiles."""
from pathlib import Path
import argparse
import csv
import numpy as np

from algorithms.model import predict

ROOT = Path(__file__).resolve().parents[1]


def run(input_path, out, angles=4608):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    depths = np.linspace(.1, 2.9, 8)
    with Path(input_path).open(encoding='utf-8-sig', newline='') as handle:
        cases = list(csv.DictReader(handle))
    names = [row['case'] for row in cases]
    if len(set(names)) != len(names):
        raise ValueError('Case names must be unique')
    history_columns = ['case', 'phase', 'z_corner_mm', 'recovery_x_mm',
                       'recovery_y_mm', 'reaction_x_n', 'reaction_y_n', 'lateral_force_n']
    contour_columns = ['case', 'depth_mm', 'angle_deg', 'predicted_radius_mm']
    with (out/'history.csv').open('w', encoding='utf-8', newline='') as hf, \
            (out/'contours.csv').open('w', encoding='utf-8', newline='') as cf:
        history = csv.writer(hf)
        contours = csv.writer(cf)
        history.writerow(history_columns)
        contours.writerow(contour_columns)
        for row in cases:
            values = {key: float(value) for key, value in row.items() if key != 'case'}
            K = np.array([[values['kxx_n_mm'], values['kxy_n_mm']],
                          [values['kxy_n_mm'], values['kyy_n_mm']]])
            trace = predict(K, [values['cxz_mm_n'], values['cyz_mm_n']],
                            values['F0_n'], depths, qc=values['qc_n_mm'],
                            overtravel=values['overtravel_mm'],H=values['thickness_mm'],
                            radius=values['radius_mm'],point_angle=values['point_angle_deg'],angles=angles)
            for phase, z, u, reaction in zip(trace['phase'], trace['z_corner'],
                                            trace['u'], trace['reaction']):
                history.writerow([row['case'], phase, z, *u, *reaction,
                                  np.linalg.norm(reaction)])
            for depth, radii in zip(depths, trace['rho']):
                for angle, radius in zip(np.degrees(trace['theta']), radii):
                    contours.writerow([row['case'], depth, angle, radius])
            print(row['case'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs')
    parser.add_argument('--angles', type=int, default=4608)
    args = parser.parse_args()
    run(args.input, args.out, args.angles)


if __name__ == '__main__':
    main()
