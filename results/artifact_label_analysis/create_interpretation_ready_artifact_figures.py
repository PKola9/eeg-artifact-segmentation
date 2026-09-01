from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "ARTIFACT_SPECIFIC_METRIC_BARS" / "00_source_per_artifact_label_metrics.csv"
OUT = ROOT / "INTERPRETATION_READY_ARTIFACT_RESULTS"


def performance_band(dice: float) -> str:
    if dice >= 0.60:
        return "strong"
    if dice >= 0.35:
        return "moderate"
    return "weak"


def band_color(band: str) -> str:
    return {"strong": "#16a34a", "moderate": "#f59e0b", "weak": "#dc2626"}[band]


def nice_label(label: str) -> str:
    return label.replace("_", " ")


def save_ranked_dice(df: pd.DataFrame) -> None:
    plot = df.sort_values("dice", ascending=True).copy()
    colors = [band_color(b) for b in plot["performance_band"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot["display_label"], plot["dice"], color=colors, edgecolor="#111827", linewidth=0.7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Dice / F1 score", fontsize=13)
    ax.set_title("Artifact-wise segmentation performance for the selected M3 model", fontsize=17, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    for y, value in enumerate(plot["dice"]):
        ax.text(value + 0.015, y, f"{value:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.text(0.03, -0.9, "Green ≥ 0.60: strong   |   Amber 0.35–0.60: moderate   |   Red < 0.35: weak",
            fontsize=11, color="#475569")
    fig.tight_layout()
    fig.savefig(OUT / "01_artifact_labels_ranked_by_dice_interpretation.png", dpi=220)
    plt.close(fig)


def save_precision_recall(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [band_color(b) for b in df["performance_band"]]
    sizes = 280 * (df["artifact_prevalence"] / df["artifact_prevalence"].max()).clip(lower=0.12) + 60
    ax.scatter(df["recall_sensitivity"], df["precision"], s=sizes, c=colors, edgecolor="#111827", linewidth=0.7, alpha=0.9)

    for _, r in df.iterrows():
        ax.text(r["recall_sensitivity"] + 0.012, r["precision"] + 0.012, r["display_label"], fontsize=9)

    ax.axhline(0.5, color="#94a3b8", linewidth=1, linestyle="--")
    ax.axvline(0.5, color="#94a3b8", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall: fraction of true artifact samples detected", fontsize=13)
    ax.set_ylabel("Precision: fraction of predicted artifact samples that were correct", fontsize=13)
    ax.set_title("Artifact labels reveal different false-positive / missed-detection behavior", fontsize=16, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.text(0.03, 0.96, "Upper-right is best", fontsize=11, color="#475569")
    fig.tight_layout()
    fig.savefig(OUT / "02_artifact_labels_precision_recall_interpretation.png", dpi=220)
    plt.close(fig)


def save_probability_quality(df: pd.DataFrame) -> None:
    plot = df.sort_values("average_precision_pr_auc", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot["display_label"], plot["average_precision_pr_auc"], color="#7c3aed", edgecolor="#111827", linewidth=0.7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average precision / PR-AUC", fontsize=13)
    ax.set_title("Probability quality differs strongly across manual artifact labels", fontsize=17, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(plot["average_precision_pr_auc"]):
        ax.text(value + 0.015, y, f"{value:.3f}", va="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "03_artifact_labels_pr_auc_interpretation.png", dpi=220)
    plt.close(fig)


def save_hausdorff(df: pd.DataFrame) -> None:
    plot = df.sort_values("hausdorff_boundary_seconds", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot["display_label"], plot["hausdorff_boundary_seconds"], color="#0891b2", edgecolor="#111827", linewidth=0.7)
    ax.set_xlabel("Hausdorff boundary distance in seconds (lower is better)", fontsize=13)
    ax.set_title("Boundary mismatch by artifact label", fontsize=17, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(plot["hausdorff_boundary_seconds"]):
        ax.text(value + 0.12, y, f"{value:.2f}s", va="center", fontsize=11, fontweight="bold")
    ax.text(0.05, -0.9, "Hausdorff is a strict worst-case boundary metric, not an average boundary error.",
            fontsize=11, color="#475569")
    fig.tight_layout()
    fig.savefig(OUT / "04_artifact_labels_boundary_mismatch_interpretation.png", dpi=220)
    plt.close(fig)


def save_summary_table(df: pd.DataFrame) -> None:
    def explanation(row):
        label = row["artifact_label"]
        if label == "hor_eyem":
            return "Best class: spatially broad, sustained horizontal eye movement pattern; high precision and high recall."
        if label in {"blink", "eyebrow", "chew"}:
            return "Clear high-amplitude artifact morphology; very high precision, but recall below precision means the model detects confident cores more than full duration."
        if label in {"blink_hor_headm", "blink_eyebrow"}:
            return "Mixed artifact label; detected moderately well, but overlap is harder because the event combines multiple sources."
        if label in {"tongue", "swallow_eyebrow"}:
            return "Shorter or less stereotyped artifact pattern; more false positives/missed boundary portions reduce Dice."
        if label in {"hor_headm", "blink_ver_headm", "ver_headm"}:
            return "Head-movement/vertical movement labels are weakest; likely more variable, lower sample support, and harder to separate from baseline drift."
        return "Artifact label shows intermediate behavior."

    out = df.sort_values("dice", ascending=False).copy()
    out["interpretation"] = out.apply(explanation, axis=1)
    cols = [
        "artifact_label",
        "performance_band",
        "dice",
        "iou_jaccard_artifact",
        "precision",
        "recall_sensitivity",
        "average_precision_pr_auc",
        "hausdorff_boundary_seconds",
        "interpretation",
    ]
    out[cols].to_csv(OUT / "00_artifact_label_interpretation_table.csv", index=False)

    md = ["# Artifact-wise segmentation interpretation\n"]
    md.append("Source: `ARTIFACT_SPECIFIC_METRIC_BARS/00_source_per_artifact_label_metrics.csv`.\n")
    md.append("The model is a binary artifact segmenter. It does not classify artifact type. For this analysis, each manual artifact label is isolated and compared against the same binary predicted artifact mask.\n")
    md.append("| Artifact label | Band | Dice | IoU | Precision | Recall | PR-AUC | Hausdorff (s) | Interpretation |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in out.iterrows():
        md.append(
            f"| {r['artifact_label']} | {r['performance_band']} | {r['dice']:.3f} | {r['iou_jaccard_artifact']:.3f} | "
            f"{r['precision']:.3f} | {r['recall_sensitivity']:.3f} | {r['average_precision_pr_auc']:.3f} | "
            f"{r['hausdorff_boundary_seconds']:.2f} | {r['interpretation']} |"
        )
    (OUT / "00_READ_THIS_ARTIFACT_LABEL_INTERPRETATION.md").write_text("\n".join(md), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    df["performance_band"] = df["dice"].apply(performance_band)
    df["display_label"] = df["artifact_label"].apply(nice_label)
    save_summary_table(df)
    save_ranked_dice(df)
    save_precision_recall(df)
    save_probability_quality(df)
    save_hausdorff(df)
    print(f"Saved interpretation-ready artifact figures in: {OUT}")


if __name__ == "__main__":
    main()
