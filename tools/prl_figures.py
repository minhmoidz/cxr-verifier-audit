"""Draw the PRL leakage-utility frontier figure from outputs/generated/frontier.csv.

Output: outputs/figures/fig_frontier.pdf (two panels: TPR at 1% FPR and AUC versus frozen-
classifier utility). Solid lines: reference-verifier attacker (the only adequate one); dashed,
hollow markers: ImageNet attacker with the full recipe. Error bars: SD over three seeds.
Run tools/prl_tables.py first.
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "outputs", "generated", "frontier.csv")
OUT = os.path.join(ROOT, "outputs", "figures", "fig_frontier.pdf")

# Categorical slots 1-3 of the validated palette (dataviz skill; validator run 2026-11-20:
# all checks pass in light mode; slot 3 is below 3:1 contrast, so every line is also
# direct-labelled and each family has its own marker shape).
FAM = {"ff": ("Learned field", "#2a78d6", "o"), "blur": ("Gaussian blur", "#eb6834", "s"),
       "elastic": ("Random elastic", "#1baf7a", "^")}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"


def load():
    pts = defaultdict(list)
    for r in csv.DictReader(open(SRC)):
        if not r["n"] or int(r["n"]) == 0:
            continue
        pts[(r["family"], r["attacker"])].append(
            dict(u=float(r["utility_frozen"]), auc=float(r["auc"]), auc_sd=float(r["auc_sd"]),
                 tpr=float(r["tpr1"]), tpr_sd=float(r["tpr1_sd"]), san=r["sanitizer"]))
    for v in pts.values():
        v.sort(key=lambda p: p["u"])
    return pts


def panel(ax, pts, metric, ylabel, label_dy):
    for att, ls, filled in (("reference", "-", True), ("imagenet_full_recipe", "--", False)):
        for fam, (name, col, mk) in FAM.items():
            p = pts.get((fam, att), [])
            if not p:
                continue
            x = [q["u"] for q in p]; y = [q[metric] for q in p]; e = [q[metric + "_sd"] for q in p]
            ax.errorbar(x, y, yerr=e, color=col, ls=ls, lw=1.2, marker=mk, ms=4.5,
                        mfc=col if filled else "white", mec=col, mew=1.1, elinewidth=0.7,
                        capsize=1.5, zorder=3 if filled else 2)
            if att == "reference":   # direct label beside each solid line
                q = p[1] if fam == "ff" else p[-1]    # learned field: its middle point, clear of the star
                ax.annotate(name, (q["u"], q[metric]), xytext=label_dy[fam],
                            textcoords="offset points", fontsize=6.5, color=INK, va="center")
        none = pts.get(("none", att), [])
        for q in none:
            ax.plot(q["u"], q[metric], marker="*", ms=7, color=MUTED if filled else "white",
                    mec=MUTED, mew=0.9, ls="none", zorder=4)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlabel("Utility: frozen-classifier macro AUC (%)", color=INK)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42})
    pts = load()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    panel(axes[0], pts, "tpr", "Leakage: TPR at 1% FPR (%)", {"ff": (5, -8), "blur": (4, -5), "elastic": (4, 4)})
    panel(axes[1], pts, "auc", "Leakage: AUC (%)", {"ff": (5, -8), "blur": (4, 3), "elastic": (4, 4)})
    for ax, t in zip(axes, ("(a)", "(b)")):
        ax.set_xlim(73.6, 82.4)
        ax.text(0.01, 0.98, t, transform=ax.transAxes, fontsize=7.5, va="top", color=INK)
    # Legend for line style (attacker) and the unsanitized reference point.
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, ls="-", lw=1.2, marker="o", ms=4, mfc=INK),
         Line2D([], [], color=INK, ls="--", lw=1.2, marker="o", ms=4, mfc="white"),
         Line2D([], [], color=MUTED, ls="none", marker="*", ms=7)]
    fig.legend(h, ["Reference verifier", "ImageNet verifier, full recipe",
                   "Unsanitized images"], loc="upper center", ncol=3, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
