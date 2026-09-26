"""Render the GoEmotions AUC-versus-deployed-depth crossover diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np


MAGNITUDES = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
MODELS = {
    "RoBERTa": {
        "scores": (
            "goemo_roberta_pw20_scores.json",
            "goemo_roberta_pw20_s43_scores.json",
            "goemo_roberta_pw20_s44_scores.json",
        ),
        "transfer": (
            "goemo_roberta_pw20_transfer.json",
            "goemo_roberta_pw20_s43_transfer.json",
            "goemo_roberta_pw20_s44_transfer.json",
        ),
        "color": "#19796F",
        "marker": "o",
    },
    "ModernBERT": {
        "scores": (
            "goemo_modernbert_pw20_scores.json",
            "goemo_modernbert_pw20_s43_scores.json",
            "goemo_modernbert_pw20_s44_scores.json",
        ),
        "transfer": (
            "goemo_modernbert_pw20_transfer.json",
            "goemo_modernbert_pw20_s43_transfer.json",
            "goemo_modernbert_pw20_s44_transfer.json",
        ),
        "color": "#BD5C43",
        "marker": "s",
    },
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def collect(results_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    collected: dict[str, dict[str, np.ndarray]] = {}
    for model, config in MODELS.items():
        score_rows = [load_json(results_dir / name) for name in config["scores"]]
        transfer_rows = [load_json(results_dir / name) for name in config["transfer"]]
        auc = np.asarray([row["test_macro_auc"] for row in score_rows], dtype=float)
        fixed_count_f1 = np.asarray(
            [
                [row["magnitudes"][str(magnitude)]["receiver"] for magnitude in MAGNITUDES]
                for row in transfer_rows
            ],
            dtype=float,
        )
        collected[model] = {"auc": auc, "fixed_count_f1": fixed_count_f1}
    return collected


def write_csv(path: Path, data: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "test_macro_auc_mean", "test_macro_auc_sd", "prior_multiplier",
             "fixed_count_f1_mean", "fixed_count_f1_sd", "n_seeds"]
        )
        for model, values in data.items():
            auc = values["auc"]
            f1 = values["fixed_count_f1"]
            for column, magnitude in enumerate(MAGNITUDES):
                writer.writerow(
                    [
                        model,
                        float(auc.mean()),
                        float(auc.std(ddof=1)),
                        magnitude,
                        float(f1[:, column].mean()),
                        float(f1[:, column].std(ddof=1)),
                        len(auc),
                    ]
                )
    differences = 100 * (
        data["RoBERTa"]["fixed_count_f1"] - data["ModernBERT"]["fixed_count_f1"]
    )
    paired_path = path.with_name(path.stem + "_paired_differences.csv")
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "prior_multiplier", "mean_f1_points", "sample_sd_f1_points",
            "paired_seed_1", "paired_seed_2", "paired_seed_3",
        ])
        for column, magnitude in enumerate(MAGNITUDES):
            values = differences[:, column]
            writer.writerow([magnitude, values.mean(), values.std(ddof=1), *values])


def render(path: Path, data: dict[str, dict[str, np.ndarray]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 7.5, "axes.titlesize": 8.3,
        "axes.titleweight": "semibold", "text.color": "#28333C",
        "axes.labelcolor": "#28333C", "xtick.color": "#58636B",
        "ytick.color": "#58636B", "axes.edgecolor": "#BAC3C8",
        "axes.linewidth": 0.65, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure, (axis_auc, axis_f1, axis_gap) = plt.subplots(
        1, 3, figsize=(7.2, 2.55),
        gridspec_kw={"width_ratios": (0.98, 1.58, 1.0)},
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.77, wspace=0.57)

    names = list(MODELS)
    seed_offsets = np.linspace(-0.075, 0.075, len(data[names[0]]["auc"]))
    for index, name in enumerate(names):
        values = data[name]["auc"]
        config = MODELS[name]
        axis_auc.errorbar(
            index, values.mean(), yerr=values.std(ddof=1),
            fmt=config["marker"], color=config["color"],
            markersize=5.5, elinewidth=1.25, capsize=3, zorder=4,
            markeredgecolor="white", markeredgewidth=0.65,
        )
        axis_auc.scatter(
            index + 0.22 + seed_offsets, values, s=9,
            color=config["color"], alpha=0.48, linewidths=0, zorder=3,
        )
        axis_auc.annotate(
            f"{values.mean():.4f}", (index, values.mean() + values.std(ddof=1)),
            xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
            color=config["color"], fontsize=8, fontweight="semibold",
        )
    axis_auc.set_xticks(range(len(names)), names)
    axis_auc.set_xlim(-0.35, 1.48)
    axis_auc.set_ylim(0.925, 0.9324)
    axis_auc.set_yticks([0.926, 0.928, 0.930, 0.932])
    axis_auc.set_ylabel("Test macro-AUC")
    axis_auc.set_title("(a) Global ranking", loc="left", pad=12)

    x = np.asarray(MAGNITUDES)
    for name, config in MODELS.items():
        values = data[name]["fixed_count_f1"]
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        axis_f1.plot(
            x, mean, color=config["color"], marker=config["marker"], linewidth=1.5,
            markersize=3.6, markeredgecolor="white", markeredgewidth=0.45,
            label=name, zorder=4,
        )
        axis_f1.fill_between(
            x, mean - sd, mean + sd, color=config["color"], alpha=0.13,
            linewidth=0, zorder=2,
        )
    axis_f1.set_xscale("log", base=2)
    axis_f1.xaxis.set_major_locator(FixedLocator(x))
    axis_f1.xaxis.set_minor_locator(NullLocator())
    axis_f1.set_xticks(x, [f"{value:g}" for value in MAGNITUDES])
    axis_f1.set_xlim(0.92, 8.7)
    axis_f1.set_ylim(0.44, 0.635)
    axis_f1.set_yticks([0.45, 0.50, 0.55, 0.60])
    axis_f1.set_xlabel("Target-prior multiplier (log scale)", labelpad=7)
    axis_f1.set_ylabel("Fixed-count macro-F1", labelpad=5)
    axis_f1.set_title("(b) Matched workload", loc="left", pad=12)

    # Pair matching training seeds rather than propagating marginal SDs.
    differences = 100 * (
        data["RoBERTa"]["fixed_count_f1"] - data["ModernBERT"]["fixed_count_f1"]
    )
    y = np.arange(len(MAGNITUDES))
    gap_color = MODELS["RoBERTa"]["color"]
    axis_gap.axvline(0, color="#737F86", linewidth=0.85, linestyle=(0, (3, 3)))
    axis_gap.errorbar(
        differences.mean(axis=0), y, xerr=differences.std(axis=0, ddof=1),
        fmt="o", color=gap_color, markersize=4, capsize=2.5, elinewidth=1.1,
        markeredgecolor="white", markeredgewidth=0.45, zorder=4,
    )
    for seed, offset in enumerate(seed_offsets):
        axis_gap.scatter(
            differences[seed], y + 0.22 + offset, color=gap_color, alpha=0.4,
            s=7, linewidths=0, zorder=3,
        )
    axis_gap.set_yticks(y, [f"{value:g}x" for value in MAGNITUDES])
    axis_gap.set_ylim(len(y) - 0.5, -0.65)
    axis_gap.set_xlim(-0.35, 3.15)
    axis_gap.set_xticks([0, 1, 2, 3], ["0", "+1", "+2", "+3"])
    axis_gap.set_xlabel("RoBERTa - ModernBERT\nF1 difference (points)", labelpad=7)
    axis_gap.set_title("(c) Local advantage", loc="left", pad=12)

    for axis in (axis_auc, axis_f1, axis_gap):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=2.5, width=0.6, pad=3)
        axis.set_axisbelow(True)
        axis.grid(axis="x" if axis is axis_gap else "y", color="#E6E9EB", linewidth=0.5)

    handles = [
        Line2D([0], [0], color=config["color"], marker=config["marker"],
               markersize=4.5, linewidth=1.5, label=name)
        for name, config in MODELS.items()
    ]
    figure.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.06, 1.02),
        ncol=2, frameon=False, handlelength=1.6, columnspacing=1.6,
    )
    figure.text(
        0.985, 0.963, "3 training seeds  |  mean +/- 1 SD",
        ha="right", va="center", fontsize=7, color="#667078",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        figure.savefig(path.with_suffix(suffix), bbox_inches="tight", dpi=320)
    plt.close(figure)


def verify_crossover(data: dict[str, dict[str, np.ndarray]]) -> None:
    roberta = data["RoBERTa"]
    modern = data["ModernBERT"]
    if not modern["auc"].mean() > roberta["auc"].mean():
        raise AssertionError("Expected ModernBERT to have higher mean macro-AUC")
    if not np.all(
        roberta["fixed_count_f1"].mean(axis=0) > modern["fixed_count_f1"].mean(axis=0)
    ):
        raise AssertionError("Expected RoBERTa to win at every transported count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent / "results_a800_20260819"
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path(__file__).resolve().parents[1] / "submission" / "iclr2027" / "figures" /
        "auc_depth_crossover.pdf"
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "generated" /
        "goemotions_auc_depth_crossover.csv"
    )
    args = parser.parse_args()

    data = collect(args.results_dir)
    verify_crossover(data)
    write_csv(args.csv, data)
    render(args.figure, data)
    for model, values in data.items():
        print(
            f"{model}: AUC={values['auc'].mean():.4f}+/-{values['auc'].std(ddof=1):.4f}; "
            f"F1@1x={values['fixed_count_f1'][:, 0].mean():.4f}; "
            f"F1@8x={values['fixed_count_f1'][:, -1].mean():.4f}"
        )
    print(f"saved figure: {args.figure}")
    print(f"saved csv: {args.csv}")


if __name__ == "__main__":
    main()
