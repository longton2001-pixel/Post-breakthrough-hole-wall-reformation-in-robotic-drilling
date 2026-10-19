"""Depth-profile classification with the adopted tail-window rule."""
import numpy as np
THRESHOLD=.12
SPLIT=ENTRY_SPLIT=.7
DOMAIN=(.1,2.9)
CLASS_ORDER=("A","C","B")

def cell_fractions(z, n, threshold=THRESHOLD, split=SPLIT):
    """Nearest measured layer represents its midpoint-bounded depth cell.

    Both windows are clipped to the same actually covered interval. No
    extrapolation beyond measured endpoints is permitted. Exactly 50% is round.
    """
    z, n = (np.asarray(z, float), np.asarray(n, float))
    lo, hi = DOMAIN
    if len(z) != len(n) or len(z) < 2 or (not np.isfinite(z).all()) or (not np.isfinite(n).all()) or np.any(np.diff(z) <= 0) or np.any(n < 0):
        raise ValueError('Finite amplitudes and at least two strictly ordered depth planes required')
    if not np.isfinite([threshold, split]).all() or threshold <= 0 or (not lo < split < hi):
        raise ValueError('Positive threshold and interior depth split required')
    if z[0] > lo + 1e-08 or z[-1] < hi - 1e-08:
        raise ValueError('Incomplete common-depth coverage: cannot extrapolate')
    edges = np.r_[z[0], (z[:-1] + z[1:]) / 2.0, z[-1]]
    state = n > threshold
    fractions = []
    for start, end in ((lo, split), (split, hi)):
        width = np.maximum(0.0, np.minimum(edges[1:], end) - np.maximum(edges[:-1], start))
        assert abs(width.sum() - (end - start)) < 1e-07
        fractions.append(float(width[state].sum() / (end - start)))
    return tuple(fractions)

def assign(entry_fraction, body_fraction):
    if body_fraction > 0.5 + 1e-12:
        return 'B'
    return 'C' if entry_fraction > 0.5 + 1e-12 else 'A'

def condition_label(labels):
    if len(labels) != 3 or any((v not in CLASS_ORDER for v in labels)):
        raise ValueError('Exactly three unique hard labels required per condition')
    return sorted(labels, key=CLASS_ORDER.index)[1]
def depth_mean(z, n, lower, upper):
    """Mean continuous linear N(z), without extrapolation or 0/1 conversion."""
    z, n = (np.asarray(z, float), np.asarray(n, float))
    if len(z) != len(n) or len(z) < 2 or np.any(np.diff(z) <= 0) or (not np.isfinite(z).all()) or (not np.isfinite(n).all()) or np.any(n < 0):
        raise ValueError('Finite nonnegative amplitudes on strictly ordered planes required')
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('Finite, increasing integration limits required')
    if lower < z[0] - 1e-08 or upper > z[-1] + 1e-08:
        raise ValueError('No extrapolation beyond observed depth coverage')
    grid = np.r_[lower, z[(z > lower + 1e-08) & (z < upper - 1e-08)], upper]
    values = np.interp(grid, z, n)
    return float(np.sum(np.diff(grid) * (values[:-1] + values[1:]) / 2.0) / (upper - lower))

def profile(z, n, threshold=THRESHOLD, entry_split=ENTRY_SPLIT, tail_start=None, tail_threshold=None):
    lo, hi = DOMAIN
    tail_start = hi - (entry_split - lo) if tail_start is None else tail_start
    tail_threshold = threshold if tail_threshold is None else tail_threshold
    if not lo < tail_start < hi or tail_threshold <= 0 or (not np.isfinite(tail_threshold)):
        raise ValueError('Finite positive tail threshold and interior start required')
    entry_fraction, body_fraction = cell_fractions(z, n, threshold, entry_split)
    prior_label = assign(entry_fraction, body_fraction)
    entry_mean = depth_mean(z, n, lo, entry_split)
    tail_mean = depth_mean(z, n, tail_start, hi)
    tail_near_round = tail_mean <= tail_threshold + 1e-12
    label = 'A' if prior_label == 'A' else 'C' if tail_near_round else 'B'
    n = np.asarray(n, float)
    z = np.asarray(z, float)
    flags = []
    if len(z) == 4:
        flags.append('four_depth_planes')
    if label == 'A' and (not tail_near_round):
        flags.append('unchanged_A_gate_with_tail_above_threshold')
    if label == 'C' and entry_mean <= tail_mean:
        flags.append('C_requires_midbody_evidence_not_entry_mean_decline')
    if label == 'B' and n[-1] <= threshold:
        flags.append('tail_mean_noncircular_despite_near_round_last_plane')
    if np.min(abs(n - threshold)) < 0.01:
        flags.append('some_plane_within_0p01mm_of_threshold')
    if abs(tail_mean - tail_threshold) < 0.01:
        flags.append('tail_mean_within_0p01mm_of_threshold')
    return dict(label=label, previous_label=prior_label, entry_fraction=entry_fraction, body_fraction=body_fraction, entry_mean_N_mm=entry_mean, tail_mean_N_mm=tail_mean, tail_margin_mm=tail_mean - tail_threshold, relative_entry_to_tail_reduction=1.0 - tail_mean / entry_mean if entry_mean > 1e-12 else None, exit_plane_N_mm=float(n[-1]), tail_start_mm=float(tail_start), amplitude_threshold_mm=float(threshold), tail_threshold_mm=float(tail_threshold), state_sequence=''.join(('N' if v > threshold else 'R' for v in n)), flags=';'.join(flags))


def paper_label(source_label):
    """Convert the internal A/C/B ordering to the paper A/B/C ordering once."""
    return {'A':'A','C':'B','B':'C'}[source_label]
