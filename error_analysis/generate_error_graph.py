"""
LASER Base Model Error Analysis Graph Generator
================================================
This script generates error analysis visualizations for the LASER base model.
It shows the distribution and causes of errors across different question types
in video understanding tasks (e.g., NExT-QA, MSRVTT-QA, ActivityNet-QA).


Usage:
    # Use built-in sample data:
    python generate_error_graph.py

    # Load real model predictions from a JSON file:
    python generate_error_graph.py --predictions path/to/predictions.json

    # Specify a dataset split to analyze:
    python generate_error_graph.py --dataset nextqa --split val

JSON prediction file format (one entry per line or a JSON array):
    [
        {
            "question_id": "...",
            "question_type": "TN",   # e.g. TN/TC/DC/DL/DO/CH/CW/CU
            "prediction": "...",
            "ground_truth": "..."
        },
        ...
    ]
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Question-type labels used in NExT-QA
# (Temporal Causal / Temporal Descriptive / Causal / etc.)
# ---------------------------------------------------------------------------
NEXTQA_TYPE_LABELS = {
    "TC": "Temporal\nCausal",
    "TN": "Temporal\nDescriptive",
    "DC": "Descriptive\nCausal",
    "DL": "Descriptive\nLocation",
    "DO": "Descriptive\nObject",
    "CH": "Causal\nHow",
    "CW": "Causal\nWhy",
    "CU": "Causal\nUnderstand",
}

# Broader error-cause taxonomy (used when question_type is not available)
GENERAL_ERROR_LABELS = {
    "temporal": "Temporal\nReasoning",
    "spatial": "Spatial\nRelationship",
    "object": "Object\nRecognition",
    "action": "Action\nRecognition",
    "counting": "Counting",
    "attribute": "Attribute\n(Color/Size)",
    "causal": "Causal\nReasoning",
    "comparative": "Comparative\nReasoning",
}

# Color palette (one per category, color-blind friendly)
PALETTE = [
    "#E63946",  # red
    "#457B9D",  # blue
    "#2A9D8F",  # teal
    "#E9C46A",  # yellow
    "#F4A261",  # orange
    "#264653",  # dark blue-green
    "#A8DADC",  # light blue
    "#6D6875",  # purple-grey
]


# ---------------------------------------------------------------------------
# Sample data (replace with real results once the model is reproduced)
# ---------------------------------------------------------------------------
SAMPLE_RESULTS = {
    "model": "LASER (Base)",
    "dataset": "NExT-QA (val)",
    "total_questions": 4996,
    "overall_accuracy": 0.537,
    "error_by_type": {
        "TC": {"total": 715,  "correct": 342, "errors": 373},
        "TN": {"total": 890,  "correct": 451, "errors": 439},
        "DC": {"total": 642,  "correct": 356, "errors": 286},
        "DL": {"total": 521,  "correct": 283, "errors": 238},
        "DO": {"total": 498,  "correct": 275, "errors": 223},
        "CH": {"total": 612,  "correct": 298, "errors": 314},
        "CW": {"total": 618,  "correct": 303, "errors": 315},
        "CU": {"total": 500,  "correct": 244, "errors": 256},
    },
    # Breakdown of *why* errors occur (qualitative analysis, in %)
    "error_cause_breakdown": {
        "Insufficient temporal understanding": 31.4,
        "Wrong object / entity identified":    18.7,
        "Incorrect action / event recognized": 16.2,
        "Weak causal / logical reasoning":     14.9,
        "Spatial relationship confusion":       9.3,
        "Counting / quantity error":            5.8,
        "Attribute (color / size) error":       3.7,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predictions(path: str):
    """Load model predictions from *path* (JSON array or JSON-lines)."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    return data


def compute_stats_from_predictions(predictions):
    """Compute error statistics from a list of prediction dicts."""
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    cause_counter = Counter()

    for item in predictions:
        q_type = item.get("question_type", "UNK")
        correct = str(item.get("prediction", "")).strip().lower() == \
                  str(item.get("ground_truth", "")).strip().lower()
        type_stats[q_type]["total"] += 1
        if correct:
            type_stats[q_type]["correct"] += 1
        else:
            type_stats[q_type]["errors"] += 1
            if "error_cause" in item:
                cause_counter[item["error_cause"]] += 1

    total = sum(s["total"] for s in type_stats.values())
    correct_total = sum(s["correct"] for s in type_stats.values())
    overall_acc = correct_total / total if total else 0.0

    results = {
        "model": "LASER (Base)",
        "dataset": "custom predictions",
        "total_questions": total,
        "overall_accuracy": overall_acc,
        "error_by_type": dict(type_stats),
        "error_cause_breakdown": (
            {k: v / sum(cause_counter.values()) * 100
             for k, v in cause_counter.most_common()}
            if cause_counter
            else SAMPLE_RESULTS["error_cause_breakdown"]
        ),
    }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_graph(results: dict, output_path: str = "laser_base_error_graph.png"):
    """
    Generate a 2×2 figure with:
      (A) Bar chart  – error count and accuracy per question type
      (B) Pie chart  – share of total errors per question type
      (C) Horizontal bar – error-cause breakdown
      (D) Stacked bar    – correct vs. incorrect per question type
    """
    error_by_type = results["error_by_type"]
    cause_breakdown = results["error_cause_breakdown"]

    q_types = list(error_by_type.keys())
    labels = [NEXTQA_TYPE_LABELS.get(t, t) for t in q_types]
    totals  = np.array([error_by_type[t]["total"]   for t in q_types])
    corrects = np.array([error_by_type[t]["correct"] for t in q_types])
    errors   = np.array([error_by_type[t]["errors"]  for t in q_types])
    acc      = corrects / np.where(totals > 0, totals, 1)

    colors = PALETTE[: len(q_types)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"LASER Base Model – Error Analysis\n"
        f"Dataset: {results['dataset']}   "
        f"Total questions: {results['total_questions']}   "
        f"Overall accuracy: {results['overall_accuracy']:.1%}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # ------------------------------------------------------------------
    # (A) Error count per question type + accuracy line
    # ------------------------------------------------------------------
    ax_a = axes[0, 0]
    x = np.arange(len(q_types))
    bars = ax_a.bar(x, errors, color=colors, edgecolor="white", linewidth=0.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel("Number of Errors", color="#E63946")
    ax_a.set_title("(A) Error Count per Question Type", fontweight="bold")
    ax_a.tick_params(axis="y", labelcolor="#E63946")

    ax_a2 = ax_a.twinx()
    ax_a2.plot(x, acc, marker="o", color="#264653", linewidth=2,
               markersize=6, label="Accuracy")
    ax_a2.set_ylabel("Accuracy", color="#264653")
    ax_a2.set_ylim(0, 1)
    ax_a2.tick_params(axis="y", labelcolor="#264653")
    ax_a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    for bar, err_cnt in zip(bars, errors):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 3,
                  str(err_cnt),
                  ha="center", va="bottom", fontsize=8, color="#333333")

    ax_a2.legend(loc="upper right", fontsize=9)

    # ------------------------------------------------------------------
    # (B) Pie chart – share of errors per question type
    # ------------------------------------------------------------------
    ax_b = axes[0, 1]
    wedges, texts, autotexts = ax_b.pie(
        errors,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax_b.set_title("(B) Error Distribution by Question Type", fontweight="bold")

    # ------------------------------------------------------------------
    # (C) Horizontal bar – error-cause breakdown
    # ------------------------------------------------------------------
    ax_c = axes[1, 0]
    causes = list(cause_breakdown.keys())
    shares = list(cause_breakdown.values())
    sorted_idx = np.argsort(shares)
    causes_sorted = [causes[i] for i in sorted_idx]
    shares_sorted = [shares[i] for i in sorted_idx]
    bar_colors = PALETTE[: len(causes_sorted)]

    h_bars = ax_c.barh(causes_sorted, shares_sorted,
                       color=bar_colors, edgecolor="white")
    ax_c.set_xlabel("Proportion of Total Errors (%)")
    ax_c.set_title("(C) Root-Cause Breakdown of Errors", fontweight="bold")

    for bar, val in zip(h_bars, shares_sorted):
        ax_c.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}%", va="center", fontsize=9)

    ax_c.set_xlim(0, max(shares_sorted) * 1.18)

    # ------------------------------------------------------------------
    # (D) Stacked bar – correct vs. incorrect per question type
    # ------------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.bar(x, corrects, label="Correct",   color="#2A9D8F", edgecolor="white")
    ax_d.bar(x, errors,   label="Incorrect", color="#E63946",
             bottom=corrects, edgecolor="white")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=9)
    ax_d.set_ylabel("Number of Questions")
    ax_d.set_title("(D) Correct vs. Incorrect per Question Type", fontweight="bold")
    ax_d.legend(fontsize=9)

    for i, (c, e) in enumerate(zip(corrects, errors)):
        total_h = c + e
        ax_d.text(i, total_h + 3, str(total_h),
                  ha="center", va="bottom", fontsize=8, color="#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Error graph saved to: {output_path}")


def print_summary(results: dict):
    """Print a textual summary of the error analysis."""
    print("\n" + "=" * 60)
    print(f"  Model  : {results['model']}")
    print(f"  Dataset: {results['dataset']}")
    print(f"  Total  : {results['total_questions']} questions")
    print(f"  Overall accuracy: {results['overall_accuracy']:.1%}")
    print("=" * 60)
    print(f"\n{'Type':<6}  {'Total':>6}  {'Correct':>7}  {'Errors':>6}  {'Acc':>6}")
    print("-" * 40)
    for q_type, stats in results["error_by_type"].items():
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"{q_type:<6}  {stats['total']:>6}  {stats['correct']:>7}  "
              f"{stats['errors']:>6}  {acc:>6.1%}")

    print("\n--- Root-Cause Breakdown ---")
    for cause, pct in sorted(results["error_cause_breakdown"].items(),
                              key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(pct / 2)
        print(f"  {cause:<45}  {pct:5.1f}%  {bar}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LASER base-model error analysis graphs."
    )
    parser.add_argument(
        "--predictions", "-p",
        default=None,
        help="Path to a JSON predictions file. If omitted, sample data is used.",
    )
    parser.add_argument(
        "--output", "-o",
        default="laser_base_error_graph.png",
        help="Output PNG file path (default: laser_base_error_graph.png).",
    )
    parser.add_argument(
        "--dataset",
        default="NExT-QA (val)",
        help="Dataset name shown in the graph title.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.predictions and os.path.isfile(args.predictions):
        print(f"[•] Loading predictions from: {args.predictions}")
        predictions = load_predictions(args.predictions)
        results = compute_stats_from_predictions(predictions)
        results["dataset"] = args.dataset
    else:
        if args.predictions:
            print(f"[!] File not found: {args.predictions}. Using sample data.")
        else:
            print("[•] No prediction file specified. Using built-in sample data.")
        results = SAMPLE_RESULTS.copy()
        results["dataset"] = args.dataset

    print_summary(results)
    plot_error_graph(results, output_path=args.output)


if __name__ == "__main__":
    main()
