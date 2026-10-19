"""Compute withdrawal reaction forces at supplied axial positions."""
from pathlib import Path
import argparse,csv
from collections import defaultdict
import numpy as np
from algorithms.model import simulate,query

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    path=Path(path)
    if path.is_dir():
        files=sorted(path.glob('*.csv'))
        if not files:raise ValueError(f'No CSV files in {path}')
        return [row for file in files for row in read(file)]
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--sampling',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/force_prediction.csv')
    args=p.parse_args();samples=defaultdict(list)
    for row in read(args.sampling):samples[row['case']].append(row)
    cases=read(args.cases)
    if len({row['case'] for row in cases})!=len(cases) or {row['case'] for row in cases}!=set(samples):
        raise ValueError('Unique cases and matching position sampling required')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['case','time_s','z_corner_mm','recovery_x_mm','recovery_y_mm','reaction_x_n','reaction_y_n','predicted_lateral_n'])
        for row in cases:
            v={k:float(x) for k,x in row.items() if k!='case'}
            K=np.array([[v['kxx_n_mm'],v['kxy_n_mm']],[v['kxy_n_mm'],v['kyy_n_mm']]])
            target=-np.array([v['cxz_mm_n'],v['cyz_mm_n']])*v['F0_n']
            tr=simulate(K,target,qc=v['qc_n_mm'],overtravel=v['overtravel_mm'],H=v['thickness_mm'],radius=v['radius_mm'],point_angle=v['point_angle_deg'])
            a=samples[row['case']];z=np.array([float(x['z_corner_mm']) for x in a]);u,R=query(tr,z)
            w.writerows([row['case'],x['time_s'],zi,*ui,*ri,np.linalg.norm(ri)] for x,zi,ui,ri in zip(a,z,u,R))

if __name__=='__main__':main()
