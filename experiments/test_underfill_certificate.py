"""Test whether deliberately under-filling the transported count repairs C-RM power.

Diagnosis this addresses
------------------------
On COCO-117k the conservative gate certifies a positive source TPR gap in 22.7%
of comparisons (fully paired: 39.4%), yet only 0.26% (0.56%) actually switch.
The cost term  g_R_upper * max(0, dmu_upper)  kills 99% of the candidates.

That cost is charged for the action-volume change dmu = mu_S - mu_R. But RM sets
the receiver threshold to MATCH the reference count, so the realized change is
~0 by construction. The gate nevertheless pays two Clopper-Pearson interval
widths, amplified by g_R/mu_R (median 22x, up to 93x on rare classes).

Idea under test
---------------
Deliberately under-fill: give the receiver slightly FEWER slots than the
reference. If mu_S <= mu_R then max(0, dmu) = 0 and the amplified cost term
vanishes entirely. The price is a smaller decision set, which lowers the
receiver's own TPR and therefore shrinks the certified gap. Whether this trade
is favourable is an empirical question -- this script answers it.

Protocol
--------
Reference and receiver are two independently trained GoEmotions heads whose
score files are archived locally. Sample splits are disjoint: source validation
is halved into threshold-setting and source-certificate parts; every target draw
is split into threshold-matching, rate-certificate, and evaluation parts. Target
labels are used only for evaluation and never by the gate.

This is a mechanism test on the certificate, not a replication of the paper's
main protocol.
"""

from pathlib import Path
import numpy as np
from scipy.stats import beta as Beta

RESULTS = Path(__file__).resolve().parent / "results_a800_20260819"
DELTA = 0.05
ALPHA = DELTA / 5.0
MAGNITUDES = [1.0, 2.0, 3.0, 5.0, 8.0]
TARGET_SEEDS = 40
TARGET_SIZE = 1500
MARGINS = [0.0, 0.01, 0.02, 0.05, 0.10]


def cp_upper(k, n, alpha):
    if n <= 0:
        return 1.0
    return 1.0 if k >= n else float(Beta.ppf(1 - alpha, k + 1, n - k))


def cp_lower(k, n, alpha):
    if n <= 0:
        return 0.0
    return 0.0 if k <= 0 else float(Beta.ppf(alpha, k, n - k + 1))


def f1(labels, pred):
    tp = int(np.sum(labels & pred))
    denom = int(pred.sum()) + int(labels.sum())
    return 2.0 * tp / max(1, denom)


def best_threshold(labels, scores):
    qs = np.quantile(scores, np.linspace(0.5, 0.9995, 150))
    return max(qs, key=lambda t: f1(labels, scores >= t))


def threshold_for_count(scores, k):
    """Smallest threshold selecting at most k examples."""
    if k <= 0:
        return np.inf
    k = min(k, len(scores))
    return float(np.sort(scores)[::-1][k - 1])


def evaluate(margin, ref_v, rcv_v, ref_t, rcv_t, yv, yt, rng_seed):
    """Run the gate for one class at one under-fill margin."""
    n_v = len(yv)
    order = np.random.RandomState(7).permutation(n_v)
    thr_idx, cert_idx = order[: n_v // 2], order[n_v // 2 :]

    tau = best_threshold(yv[thr_idx], ref_v[thr_idx])

    src_pos = cert_idx[yv[cert_idx] == 1]
    if len(src_pos) < 15:
        return None

    pos_pool = np.where(yt == 1)[0]
    neg_pool = np.where(yt == 0)[0]
    if len(pos_pool) < 20:
        return None
    base_prior = len(pos_pool) / len(yt)

    out = []
    for mag in MAGNITUDES:
        prior = min(0.5, base_prior * mag)
        n_pos = max(1, int(TARGET_SIZE * prior))
        for seed in range(TARGET_SEEDS):
            rng = np.random.RandomState(10_000 * rng_seed + seed)
            idx = np.concatenate([
                rng.choice(pos_pool, n_pos, replace=len(pos_pool) < n_pos),
                rng.choice(neg_pool, TARGET_SIZE - n_pos, replace=False),
            ])
            y = yt[idx]
            rs, vs = ref_t[idx], rcv_t[idx]

            a, b, c = np.array_split(np.arange(TARGET_SIZE), 3)

            # --- receiver threshold matched on the threshold subset, under-filled
            k_ref = int(np.sum(rs[a] >= tau))
            k_target = int(np.floor(k_ref * (1.0 - margin)))
            if k_target <= 0:
                continue
            t_rcv = threshold_for_count(vs[a], k_target)

            # --- source certificate (labels are SOURCE labels only)
            ref_hit = ref_v[src_pos] >= tau
            rcv_hit = rcv_v[src_pos] >= t_rcv
            m_src = len(src_pos)
            g_ref_up = cp_upper(int(ref_hit.sum()), m_src, ALPHA)
            g_rcv_lo = cp_lower(int(rcv_hit.sum()), m_src, ALPHA)
            delta_lo = g_rcv_lo - g_ref_up

            # --- target rate certificate (NO labels)
            rd, vd = rs[b] >= tau, vs[b] >= t_rcv
            n_b = len(b)
            mu_ref_lo = cp_lower(int(rd.sum()), n_b, ALPHA)
            n10 = int(np.sum(vd & ~rd))
            n01 = int(np.sum(~vd & rd))
            dmu_up = cp_upper(n10, n_b, ALPHA) - cp_lower(n01, n_b, ALPHA)

            gamma = -1.0
            if delta_lo > 0:
                gamma = mu_ref_lo * delta_lo - g_ref_up * max(0.0, dmu_up)

            # --- evaluation (labels used here only, gate never sees this)
            k_eval = int(np.sum(rs[c] >= tau))
            ref_pred = rs[c] >= tau
            rcv_pred = vs[c] >= t_rcv
            f_ref, f_rcv = f1(y[c], ref_pred), f1(y[c], rcv_pred)

            out.append((gamma, delta_lo, dmu_up, f_rcv - f_ref, k_ref, k_target))
    return out


def main():
    ref = np.load(RESULTS / "goemo_roberta_pw6_scores.npz")
    rcv = np.load(RESULTS / "goemo_roberta_pw20_scores.npz")
    yv_all, yt_all = ref["valid_labels"], ref["test_labels"]
    rv, rt = ref["valid_logits"].astype(np.float64), ref["test_logits"].astype(np.float64)
    vv, vt = rcv["valid_logits"].astype(np.float64), rcv["test_logits"].astype(np.float64)

    prev = yv_all.mean(axis=0)
    classes = [c for c in range(yv_all.shape[1]) if 0.005 < prev[c] < 0.30]
    print(f"reference = RoBERTa pos_weight_cap=6, receiver = cap=20")
    print(f"classes evaluated: {len(classes)}")
    print()

    print(f"{'margin':>7} | {'switches':>9} {'rate':>7} | {'losses':>7} | "
          f"{'cost=0':>7} | {'mean gain':>10}")
    print("-" * 66)

    for margin in MARGINS:
        rows = []
        for ci, c in enumerate(classes):
            r = evaluate(margin, rv[:, c], vv[:, c], rt[:, c], vt[:, c],
                         yv_all[:, c], yt_all[:, c], ci)
            if r:
                rows.extend(r)
        if not rows:
            print(f"{margin:7.0%} | no usable comparisons")
            continue
        arr = np.array([(g, d, m, gap) for g, d, m, gap, _, _ in rows])
        gamma, dlo, dmu, gap = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        sw = gamma > 0
        cost_zero = np.maximum(0.0, dmu) <= 0
        losses = int((gap[sw] < 0).sum()) if sw.any() else 0
        mg = gap[sw].mean() if sw.any() else float("nan")
        print(f"{margin:7.0%} | {sw.sum():9d} {sw.mean():7.2%} | {losses:7d} | "
              f"{cost_zero.mean():7.1%} | {mg:+10.4f}")

    print()
    print(f"total comparisons per margin: {len(rows)}")
    print("'cost=0' = share of comparisons where the amplified cost term vanished.")
    print("'losses' = certified switches whose realized F1 was worse (oracle check).")


if __name__ == "__main__":
    main()
