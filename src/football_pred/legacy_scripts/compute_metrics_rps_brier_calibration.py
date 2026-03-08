"""
Compute RPS, Brier Score, ECE and calibration plots for all 3 model versions.
Uses the saved test predictions CSVs.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

RES_DIR = "/home/ubuntu/football_pred/results"
os.makedirs(RES_DIR, exist_ok=True)

MODELS = {
    "Baseline (v2)": "baseline_v2",
    "Europe (v3)":   "europe_v3",
    "Advanced (v4)": "advanced_v4",
}
COLORS = {
    "Baseline (v2)": "#42A5F5",
    "Europe (v3)":   "#E53935",
    "Advanced (v4)": "#FF6F00",
}
OUTCOMES = ["Home", "Draw", "Away"]
RESULT_MAP = {"H": 0, "D": 1, "A": 2}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
dfs = {}
for label, key in MODELS.items():
    path = f"{RES_DIR}/test_predictions_{key}.csv"
    df = pd.read_csv(path)
    df["y_true"] = df["result"].map(RESULT_MAP)
    dfs[label] = df
    print(f"Loaded {label}: {len(df)} predictions")

# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rps_single(probs, outcome):
    """
    Ranked Probability Score for a single match.
    probs: array of shape (3,) summing to 1 [P_Home, P_Draw, P_Away]
    outcome: integer 0=H, 1=D, 2=A
    """
    n = len(probs)
    actual = np.zeros(n)
    actual[outcome] = 1.0
    cum_pred = np.cumsum(probs)
    cum_actual = np.cumsum(actual)
    return np.sum((cum_pred[:-1] - cum_actual[:-1]) ** 2) / (n - 1)


def compute_rps(df):
    probs = df[["P_Home", "P_Draw", "P_Away"]].values
    outcomes = df["y_true"].values
    scores = [rps_single(probs[i], outcomes[i]) for i in range(len(df))]
    return np.mean(scores), np.std(scores) / np.sqrt(len(scores))


def compute_brier(df):
    """Multi-class Brier Score: mean over matches of sum of (p_k - o_k)^2."""
    probs = df[["P_Home", "P_Draw", "P_Away"]].values
    n = len(df)
    K = 3
    one_hot = np.zeros((n, K))
    for i, y in enumerate(df["y_true"].values):
        one_hot[i, y] = 1.0
    bs_per_match = np.sum((probs - one_hot) ** 2, axis=1)
    return np.mean(bs_per_match), np.std(bs_per_match) / np.sqrt(n)


def compute_ece(df, outcome_idx, n_bins=10):
    """
    Expected Calibration Error for a single outcome class.
    outcome_idx: 0=Home, 1=Draw, 2=A
    """
    col = ["P_Home", "P_Draw", "P_Away"][outcome_idx]
    probs = df[col].values
    actuals = (df["y_true"].values == outcome_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bin_data.append(None)
            continue
        avg_pred = probs[mask].mean()
        avg_actual = actuals[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(avg_pred - avg_actual)
        bin_data.append({
            "bin_center": (lo + hi) / 2,
            "avg_pred": avg_pred,
            "avg_actual": avg_actual,
            "count": mask.sum(),
        })
    return ece, bin_data


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE ALL METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPUTING METRICS")
print("=" * 70)

results = {}
for label, df in dfs.items():
    rps_mean, rps_se = compute_rps(df)
    bs_mean, bs_se = compute_brier(df)
    ece_home, bins_home = compute_ece(df, 0)
    ece_draw, bins_draw = compute_ece(df, 1)
    ece_away, bins_away = compute_ece(df, 2)
    ece_mean = (ece_home + ece_draw + ece_away) / 3

    results[label] = {
        "rps": rps_mean, "rps_se": rps_se,
        "brier": bs_mean, "brier_se": bs_se,
        "ece_home": ece_home, "ece_draw": ece_draw, "ece_away": ece_away,
        "ece_mean": ece_mean,
        "bins_home": bins_home, "bins_draw": bins_draw, "bins_away": bins_away,
    }
    print(f"\n{label}:")
    print(f"  RPS:          {rps_mean:.5f} ± {rps_se:.5f}")
    print(f"  Brier Score:  {bs_mean:.5f} ± {bs_se:.5f}")
    print(f"  ECE (Home):   {ece_home:.5f}")
    print(f"  ECE (Draw):   {ece_draw:.5f}")
    print(f"  ECE (Away):   {ece_away:.5f}")
    print(f"  ECE (Mean):   {ece_mean:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

comp = pd.DataFrame({
    "Metric": [
        "RPS (↓)", "Brier Score (↓)",
        "ECE Home (↓)", "ECE Draw (↓)", "ECE Away (↓)", "ECE Mean (↓)",
    ],
    "Baseline (v2)": [
        results["Baseline (v2)"]["rps"],
        results["Baseline (v2)"]["brier"],
        results["Baseline (v2)"]["ece_home"],
        results["Baseline (v2)"]["ece_draw"],
        results["Baseline (v2)"]["ece_away"],
        results["Baseline (v2)"]["ece_mean"],
    ],
    "Europe (v3)": [
        results["Europe (v3)"]["rps"],
        results["Europe (v3)"]["brier"],
        results["Europe (v3)"]["ece_home"],
        results["Europe (v3)"]["ece_draw"],
        results["Europe (v3)"]["ece_away"],
        results["Europe (v3)"]["ece_mean"],
    ],
    "Advanced (v4)": [
        results["Advanced (v4)"]["rps"],
        results["Advanced (v4)"]["brier"],
        results["Advanced (v4)"]["ece_home"],
        results["Advanced (v4)"]["ece_draw"],
        results["Advanced (v4)"]["ece_away"],
        results["Advanced (v4)"]["ece_mean"],
    ],
})
for col in ["Baseline (v2)", "Europe (v3)", "Advanced (v4)"]:
    comp[col] = comp[col].round(5)

print(comp.to_string(index=False))
comp.to_csv(f"{RES_DIR}/metrics_rps_brier_ece.csv", index=False)
print(f"\nSaved to {RES_DIR}/metrics_rps_brier_ece.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: RPS and Brier Score comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Comparaison des Métriques Probabilistes — v2 vs v3 vs v4", fontsize=14, fontweight="bold")

labels = list(MODELS.keys())
colors = [COLORS[l] for l in labels]

# RPS
rps_vals = [results[l]["rps"] for l in labels]
rps_ses = [results[l]["rps_se"] for l in labels]
bars = axes[0].bar(labels, rps_vals, color=colors, edgecolor="white", linewidth=1.5,
                   yerr=rps_ses, capsize=5, error_kw={"elinewidth": 1.5})
axes[0].set_ylabel("RPS (lower is better)", fontsize=11)
axes[0].set_title("Ranked Probability Score (RPS)", fontsize=12)
y_min = min(rps_vals) * 0.995
y_max = max(rps_vals) * 1.005
axes[0].set_ylim(y_min, y_max)
for bar, v, se in zip(bars, rps_vals, rps_ses):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + se + 0.0001,
                 f"{v:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
# Highlight best
best_rps_idx = np.argmin(rps_vals)
bars[best_rps_idx].set_edgecolor("#2E7D32")
bars[best_rps_idx].set_linewidth(3)

# Brier
brier_vals = [results[l]["brier"] for l in labels]
brier_ses = [results[l]["brier_se"] for l in labels]
bars2 = axes[1].bar(labels, brier_vals, color=colors, edgecolor="white", linewidth=1.5,
                    yerr=brier_ses, capsize=5, error_kw={"elinewidth": 1.5})
axes[1].set_ylabel("Brier Score (lower is better)", fontsize=11)
axes[1].set_title("Brier Score (Multi-class)", fontsize=12)
y_min2 = min(brier_vals) * 0.995
y_max2 = max(brier_vals) * 1.005
axes[1].set_ylim(y_min2, y_max2)
for bar, v, se in zip(bars2, brier_vals, brier_ses):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + se + 0.0001,
                 f"{v:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
best_brier_idx = np.argmin(brier_vals)
bars2[best_brier_idx].set_edgecolor("#2E7D32")
bars2[best_brier_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig(f"{RES_DIR}/rps_brier_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RES_DIR}/rps_brier_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: ECE bar chart per outcome
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Expected Calibration Error (ECE) par Résultat", fontsize=14, fontweight="bold")

outcomes_labels = ["Home Win", "Draw", "Away Win", "Mean"]
ece_keys = ["ece_home", "ece_draw", "ece_away", "ece_mean"]
x = np.arange(len(outcomes_labels))
w = 0.25

for i, (label, color) in enumerate(zip(labels, colors)):
    vals = [results[label][k] for k in ece_keys]
    bars = ax.bar(x + i * w, vals, w, label=label, color=color, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

ax.set_ylabel("ECE (lower is better)", fontsize=11)
ax.set_xticks(x + w)
ax.set_xticklabels(outcomes_labels, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, max(results[l][k] for l in labels for k in ece_keys) * 1.35)
plt.tight_layout()
plt.savefig(f"{RES_DIR}/ece_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RES_DIR}/ece_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Calibration plots — one row per outcome, one column per model
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle("Courbes de Calibration — Probabilités Prédites vs Fréquences Observées",
             fontsize=15, fontweight="bold", y=1.01)

outcome_names = ["Victoire Domicile (Home)", "Match Nul (Draw)", "Victoire Extérieur (Away)"]
bin_keys = ["bins_home", "bins_draw", "bins_away"]

for col_idx, (label, color) in enumerate(zip(labels, colors)):
    for row_idx, (out_name, bin_key) in enumerate(zip(outcome_names, bin_keys)):
        ax = axes[row_idx][col_idx]
        bins = results[label][bin_key]
        valid_bins = [b for b in bins if b is not None]

        if valid_bins:
            pred_vals = [b["avg_pred"] for b in valid_bins]
            actual_vals = [b["avg_actual"] for b in valid_bins]
            counts = [b["count"] for b in valid_bins]
            max_count = max(counts)

            # Scatter with size proportional to count
            sizes = [max(30, 300 * c / max_count) for c in counts]
            ax.scatter(pred_vals, actual_vals, s=sizes, color=color, alpha=0.85,
                       edgecolors="white", linewidth=0.8, zorder=3)
            ax.plot(pred_vals, actual_vals, color=color, alpha=0.6, linewidth=1.5, zorder=2)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Perfect calibration")

        # Shaded confidence band
        ax.fill_between([0, 1], [0, 0.1], [0.1, 0.2], alpha=0.03, color="gray")

        ece_key = ["ece_home", "ece_draw", "ece_away"][row_idx]
        ece_val = results[label][ece_key]

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Probabilité prédite", fontsize=9)
        ax.set_ylabel("Fréquence observée", fontsize=9)
        ax.set_title(f"{label}\n{out_name}\nECE = {ece_val:.4f}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal")

        # Annotate bins with counts
        if valid_bins:
            for b in valid_bins:
                ax.annotate(str(b["count"]),
                            (b["avg_pred"], b["avg_actual"]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, color="gray")

plt.tight_layout()
plt.savefig(f"{RES_DIR}/calibration_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RES_DIR}/calibration_plots.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: Reliability diagram — overlay all 3 models for each outcome
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Diagrammes de Fiabilité — Comparaison des 3 Modèles par Résultat",
             fontsize=14, fontweight="bold")

for row_idx, (out_name, bin_key, ece_key) in enumerate(zip(
        ["Home Win", "Draw", "Away Win"],
        ["bins_home", "bins_draw", "bins_away"],
        ["ece_home", "ece_draw", "ece_away"])):
    ax = axes[row_idx]
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Perfect calibration", zorder=1)

    for label, color in zip(labels, colors):
        bins = results[label][bin_key]
        valid_bins = [b for b in bins if b is not None]
        if valid_bins:
            pred_vals = [b["avg_pred"] for b in valid_bins]
            actual_vals = [b["avg_actual"] for b in valid_bins]
            ece_val = results[label][ece_key]
            ax.plot(pred_vals, actual_vals, "o-", color=color, linewidth=2,
                    markersize=7, label=f"{label} (ECE={ece_val:.4f})", zorder=2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Probabilité prédite", fontsize=10)
    ax.set_ylabel("Fréquence observée", fontsize=10)
    ax.set_title(out_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{RES_DIR}/reliability_diagrams.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RES_DIR}/reliability_diagrams.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5: Probability distribution histograms per outcome per model
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Distribution des Probabilités Prédites par Résultat et Modèle",
             fontsize=14, fontweight="bold")

prob_cols = ["P_Home", "P_Draw", "P_Away"]
for col_idx, (label, color) in enumerate(zip(labels, colors)):
    df = dfs[label]
    for row_idx, (out_name, prob_col) in enumerate(zip(["Home Win", "Draw", "Away Win"], prob_cols)):
        ax = axes[row_idx][col_idx]
        # Split by actual outcome
        correct_mask = df["y_true"] == row_idx
        ax.hist(df.loc[correct_mask, prob_col], bins=20, alpha=0.6, color="#2E7D32",
                label=f"Actual={out_name}", density=True)
        ax.hist(df.loc[~correct_mask, prob_col], bins=20, alpha=0.4, color="#C62828",
                label=f"Actual≠{out_name}", density=True)
        ax.axvline(df[prob_col].mean(), color=color, linestyle="--", linewidth=2,
                   label=f"Mean={df[prob_col].mean():.3f}")
        ax.set_xlabel("Predicted Probability", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"{label}\n{out_name}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RES_DIR}/probability_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RES_DIR}/probability_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(comp.to_string(index=False))
print("\nBest model per metric (lower is better):")
for metric, key in [("RPS", "rps"), ("Brier", "brier"), ("ECE Mean", "ece_mean")]:
    vals = {l: results[l][key] for l in labels}
    best = min(vals, key=vals.get)
    print(f"  {metric}: {best} ({vals[best]:.5f})")
print("\nAll charts saved to:", RES_DIR)
