"""Draw the PRL leakage-utility figure from outputs/generated/frontier.csv.

Output: outputs/figures/fig_frontier.pdf (two panels: TPR at 1% FPR and AUC versus frozen-classifier
utility). Readable in grayscale: sanitizer families differ by marker shape, verifiers by line style
and marker fill (reference verifier: solid line, filled markers; ImageNet verifier with the full
recipe: dashed line, open markers; independent strong verifier E5, if outputs/generated/frontier_e5.csv
exists: dotted line, grey-filled markers). One serif font (STIX, matching Times in the manuscript),
black text, no grid. Error bars: SD over seeds. Run tools/prl_tables.py (and tools/v2_tables.py for
E5) first.
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "outputs", "generated", "frontier.csv")
SRC_E5 = os.path.join(ROOT, "outputs", "generated", "frontier_e5.csv")
OUT = os.path.join(ROOT, "outputs", "figures", "fig_frontier.pdf")

FAM = {"ff": ("Learned field", "o"), "blur": ("Gaussian blur", "s"), "elastic": ("Random elastic", "^")}
VER = {"reference": ("Reference verifier", "-", "black"),
       "imagenet_full_recipe": ("ImageNet verifier, full recipe", "--", "white"),
       "e5": ("Independent verifier", ":", "0.6")}
INK = "black"


def load():
    pts = defaultdict(list)
    for src in (SRC, SRC_E5):
        if not os.path.isfile(src):
            continue
        for r in csv.DictReader(open(src)):
            if not r["n"] or int(r["n"]) == 0:
                continue
            pts[(r["family"], r["attacker"])].append(
                dict(u=float(r["utility_frozen"]), auc=float(r["auc"]), auc_sd=float(r["auc_sd"]),
                     tpr=float(r["tpr1"]), tpr_sd=float(r["tpr1_sd"]), san=r["sanitizer"]))
    for v in pts.values():
        v.sort(key=lambda p: p["u"])
    return pts


def panel(ax, pts, metric, ylabel):
    for k, (att, (_, ls, fill)) in enumerate(VER.items()):
        for fam, (_, mk) in FAM.items():
            p = pts.get((fam, att), [])
            if not p:
                continue
            x = [q["u"] for q in p]; y = [q[metric] for q in p]; e = [q[metric + "_sd"] for q in p]
            ax.errorbar(x, y, yerr=e, color=INK, ls=ls, lw=0.9, marker=mk, ms=4.2, mfc=fill, mec=INK,
                        mew=0.8, elinewidth=0.6, capsize=1.5, zorder=4 - k)
        for q in pts.get(("none", att), []):
            ax.plot(q["u"], q[metric], marker="*", ms=7.5, color=INK, mfc=fill, mec=INK, mew=0.7,
                    ls="none", zorder=5)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlabel("Utility: frozen-classifier macro AUC (%)", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelcolor=INK, width=0.6, length=2.5)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.6, "pdf.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK})
    pts = load()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    panel(axes[0], pts, "tpr", "Leakage: TPR at 1% FPR (%)")
    panel(axes[1], pts, "auc", "Leakage: AUC (%)")
    for ax, t in zip(axes, ("(a)", "(b)")):
        ax.set_xlim(73.6, 82.4)
        ax.text(0.01, 0.98, t, transform=ax.transAxes, fontsize=7.5, va="top", color=INK)
    vers = [v for v in VER if any(a == v for _, a in pts)]
    h = [Line2D([], [], color=INK, ls="none", marker=mk, ms=4.2, mfc="black", mec=INK) for _, mk in FAM.values()]
    h += [Line2D([], [], color=INK, ls="none", marker="*", ms=7.5, mfc="black", mec=INK)]
    lab = [n for n, _ in FAM.values()] + ["Unsanitized images"]
    h += [Line2D([], [], color=INK, ls=VER[v][1], lw=0.9, marker="o", ms=4.2, mfc=VER[v][2], mec=INK) for v in vers]
    lab += [VER[v][0] for v in vers]
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if len(lab) > 4 else 0.92), w_pad=2.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
