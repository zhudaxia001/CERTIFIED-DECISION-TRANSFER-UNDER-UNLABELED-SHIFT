"""CPU-only replay of the released, complete GoEmotions and COCO records."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_factorized_certificate import (
    archived_gamma_audit, factorized_gamma, reconstruct_counts, summarize, verify_primary,
)
from summarize_goemo_strict_upgrade import compare, summarize as summarize_goemo
from verify_theory_strengthening import verify_theory_strengthening


ROOT = Path(__file__).resolve().parents[1]


def read(stem):
    with gzip.open(ROOT / "data" / f"{stem}.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def audit_goemo():
    expected = {
        ("tfidf", "qwen"): (3578, 225, 834),
        ("tfidf", "modernbert"): (7358, 86, 256),
        ("roberta", "qwen"): (11605, 0, 0),
        ("roberta", "modernbert"): (17649, 0, 0),
    }
    for (reference, candidate), totals in expected.items():
        all_rows = []
        for seed in (42, 43, 44):
            payload = read(f"goemo_{reference}_to_{candidate}_strict_s{seed}_t50")
            rows = payload["records"]
            assert len(rows) == 8750
            assert all(r["reference_count"] == r["receiver_count"] for r in rows)
            # This archive stores bound endpoints, not the source logits. Audit
            # the decision margin here; retraining scripts recompute the bounds.
            for prefix in ("paired_cp", "fully_paired_cp"):
                lower = np.array([r[f"{prefix}_delta_lower"] for r in rows])
                rate = np.array([r[f"{prefix}_reference_rate_lower"] for r in rows])
                tpr = np.array([r[f"{prefix}_reference_tpr_upper"] for r in rows])
                excess = np.array([r[f"{prefix}_rate_difference_upper"] for r in rows])
                recomputed = np.where(lower > 0, rate * lower - tpr * np.maximum(excess, 0), -1.)
                assert np.allclose(recomputed, [r[f"{prefix}_gamma"] for r in rows], atol=1e-12, rtol=0)
            for gate, key in (("paired", "paired_cp_gamma"), ("fully_paired", "fully_paired_cp_gamma")):
                compare(payload["summary"][gate], summarize_goemo(rows, key), f"{reference}/{candidate}/{seed}/{gate}")
            all_rows.extend(rows)
        paired = summarize_goemo(all_rows, "paired_cp_gamma")
        full = summarize_goemo(all_rows, "fully_paired_cp_gamma")
        observed = (full["naive_harmful_population_comparisons"], paired["switches"], full["switches"])
        assert observed == totals, (reference, candidate, observed)
        assert full["comparisons"] == 26250
        assert full["unsafe_population_switches"] == full["unsafe_batch_switches"] == 0
        print(f"GoEmotions {reference}->{candidate}: " + json.dumps(full))


def audit_coco():
    primary = read("coco_crm_dinov2_full117k_m25_fixed_iid_s100_d05_pop")
    rows = primary["records"]
    counts = reconstruct_counts(rows, primary["summary"]["protocol"]["target_size"])
    archived_gamma_audit(rows, counts)
    full_summary = summarize(rows, counts)
    verify_primary(full_summary)
    print("COCO full archive:", json.dumps(full_summary["delta_curve"]["0.05"]))
    splits = [[r for r in rows if r["target_seed"] < 50]]
    splits += [read(f"coco_crm_dinov2_full117k_m25_fixed_iid_s50_d05_pop_split{i}")["records"] for i in (1, 2)]
    for split, (records, expected) in enumerate(zip(splits, ((153, 616), (190, 846), (95, 439))), 1):
        assert len(records) == 31500
        counts = reconstruct_counts(records, 5000)
        archived_gamma_audit(records, counts)
        direct = np.array([r["fully_paired_cp_gamma"] > 0 for r in records])
        factorized = factorized_gamma(counts, .05) > 0
        pop = np.array([r["receiver_population_f1"] + 1e-12 < r["reference_population_f1"] for r in records])
        batch = np.array([r["receiver_f1"] + 1e-12 < r["reference_f1"] for r in records])
        assert (int(direct.sum()), int(factorized.sum())) == expected
        assert not np.any(factorized & pop)
        print(f"COCO split {split}: direct={direct.sum()}, factorized={factorized.sum()}, "
              f"population_losses={np.sum(factorized & pop)}, batch_losses={np.sum(factorized & batch)}")


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for relative, expected in manifest["code"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    for relative, metadata in manifest["archives"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == metadata["sha256"], relative
    verify_theory_strengthening()
    audit_goemo()
    audit_coco()
    print("PASS: archive integrity, decision margins, approval counts, both loss endpoints, and theory checks.")


if __name__ == "__main__":
    main()
