import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
proba = [0.3784, 0.2768, 0.3448]
labels = ["Nice Win\n(Home)", "Draw", "Rennes Win\n(Away)"]
colors = ["#E53935", "#757575", "#1E88E5"]
odds = [1/p for p in proba]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 2]})

# --- Left: Probability bar chart ---
ax = axes[0]
bars = ax.bar(labels, [p*100 for p in proba], color=colors, width=0.6, edgecolor="white", linewidth=2)
for bar, p, o in zip(bars, proba, odds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{p*100:.1f}%", ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f"Odds: {o:.2f}", ha="center", va="center", fontsize=13, color="white", fontweight="bold")

ax.set_ylim(0, 50)
ax.set_ylabel("Probability (%)", fontsize=14)
ax.set_title("Nice vs Rennes — 8 March 2026\nAllianz Riviera", fontsize=16, fontweight="bold", pad=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Right: Team stats comparison ---
ax2 = axes[1]
ax2.axis("off")

stats = [
    ("ELO Rating", "1542.9", "1584.4"),
    ("Form (5 games)", "0.6 pts/g", "1.8 pts/g"),
    ("Goals scored avg", "1.0", "1.6"),
    ("Goals conceded avg", "1.6", "1.6"),
    ("Goal diff form", "-0.6", "0.0"),
    ("Rest days", "6", "7"),
    ("H2H win rate", "60%", "40%"),
]

# Header
ax2.text(0.5, 0.97, "Team Comparison", ha="center", va="top", fontsize=15, fontweight="bold", transform=ax2.transAxes)
ax2.text(0.2, 0.91, "Nice", ha="center", va="top", fontsize=14, fontweight="bold", color="#E53935", transform=ax2.transAxes)
ax2.text(0.8, 0.91, "Rennes", ha="center", va="top", fontsize=14, fontweight="bold", color="#1E88E5", transform=ax2.transAxes)

y_start = 0.83
y_step = 0.095
for i, (label, nice_val, rennes_val) in enumerate(stats):
    y = y_start - i * y_step
    ax2.text(0.5, y, label, ha="center", va="center", fontsize=11, color="#555555", transform=ax2.transAxes)
    ax2.text(0.2, y, nice_val, ha="center", va="center", fontsize=13, fontweight="bold", color="#E53935", transform=ax2.transAxes)
    ax2.text(0.8, y, rennes_val, ha="center", va="center", fontsize=13, fontweight="bold", color="#1E88E5", transform=ax2.transAxes)
    if i < len(stats) - 1:
        ax2.plot([0.05, 0.95], [y - y_step/2 + 0.01, y - y_step/2 + 0.01], color="#E0E0E0", linewidth=0.5, transform=ax2.transAxes, clip_on=False)

# Recent form
ax2.text(0.5, 0.14, "Recent Form (last 5)", ha="center", va="center", fontsize=11, color="#555555", transform=ax2.transAxes)
# Nice: D D L D L
nice_form = ["D", "D", "L", "D", "L"]
nice_colors_form = {"W": "#4CAF50", "D": "#FFC107", "L": "#E53935"}
for j, r in enumerate(nice_form):
    ax2.text(0.08 + j*0.06, 0.06, r, ha="center", va="center", fontsize=12, fontweight="bold",
             color="white", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=nice_colors_form[r], edgecolor="none"))

# Rennes: L L W W W
rennes_form = ["L", "L", "W", "W", "W"]
for j, r in enumerate(rennes_form):
    ax2.text(0.62 + j*0.06, 0.06, r, ha="center", va="center", fontsize=12, fontweight="bold",
             color="white", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=nice_colors_form[r], edgecolor="none"))

plt.tight_layout()
plt.savefig("/home/ubuntu/nice_vs_rennes_prediction.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Prediction card saved.")
