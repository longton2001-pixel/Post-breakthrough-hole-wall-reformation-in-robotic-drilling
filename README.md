# Post-breakthrough hole-wall reformation in robotic drilling

Core algorithms for hole-wall formation in robotic drilling, with lateral-force and coordinate measuring machine (CMM) data for 108 holes in each dataset. The algorithms cover stiffness and compliance conversion, elastic recovery, withdrawal reactions, the cutting envelope, rigid registration, geometric error calculation, and morphology classification.

## Model parameters

`core/algorithms/stiffness_demo/robot_model.py` contains the modified Denavit–Hartenberg (MDH) parameters, joint stiffnesses, tool installation parameters, and controller-to-model joint coordinate transformations. `core/parameters/reference_configurations.csv` contains the joint angles and workpiece coordinate frames for six reference configurations.

`algorithms.reference.reference_parameters()` reads these configurations and uses the joint stiffnesses defined in the code to calculate compliance, lateral stiffness, and axial-to-lateral coupling.

## Data

Each dataset contains six robot postures and six drilling feed rates, with three holes per condition. Case identifiers use `FORCE` or `CMM` to identify the dataset, `P` for the paper posture number, `V` for the drilling feed rate, and `R` for repeat 01, 02, or 03. Records within each dataset are linked by `case`.

| Files | Contents |
|---|---|
| `data/cmm/CMM_Pxx_Vxxxx_Rxx.csv` | One standardized eight-section CMM profile per hole, with coordinates and radii at 8 depths × 24 angular positions; 108 files |
| `data/force/FORCE_Pxx_Vxxxx_Rxx.csv` | One measured lateral-force time series per hole; 108 files |

The section depths are 0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, and 2.9 mm, with an angular spacing of 15°. CMM coordinates are expressed in the model frame after whole-hole registration. In the force data, `time_s` is measured from the start of withdrawal.

## Usage

Python 3.12:

```bash
python -m pip install -r requirements.txt
python -B core/predict.py --help
python -B core/force.py --help
```

`predict.py` accepts a parameter table through `--input` and exports force and displacement histories and predicted hole profiles. `force.py` accepts a mechanical parameter table through `--cases` and an axial position sampling table through `--sampling`, and exports withdrawal reaction forces. Each entry point provides its full argument list through `--help`.

The parameter table contains `case`, `posture`, `speed_mm_min`, `F0_n`, `kxx_n_mm`, `kxy_n_mm`, `kyy_n_mm`, `cxz_mm_n`, `cyz_mm_n`, `overtravel_mm`, `qc_n_mm`, `thickness_mm`, `radius_mm`, and `point_angle_deg`. The position sampling table contains `case`, `time_s`, and `z_corner_mm`.

Generated results are written to `outputs/`.

## Solver conventions

Lengths are in mm, forces in N, stiffnesses in N/mm, compliances in mm/N, and angles in degrees. `F0_n` is the axial force magnitude. The initial offset is `d0 = -c_z * F0_n`, the recovery displacement is `u`, the reaction force is `K @ (d0 - u)`, and the tool center is at `-u`.

The nominal post-breakthrough overtravel is 3 mm. The numerical starting range `overtravel_mm` is set to `max(3, z_max + tip_height - H + 0.01)` to cover the supplied withdrawal positions.
