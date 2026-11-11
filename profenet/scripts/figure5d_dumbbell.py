#!/usr/bin/env python3
"""Figure 5d: per-residue CDR3 binding-energy contributions, original vs optimized.

Faceted dumbbell plot: one mini-panel per RBD background (WT, BA.2.86, JN.1),
x axis = CDR3 positions in WT numbering, dumbbell from the original residue's
per-residue contribution to Delta G_bind (open gray circle) to the optimized
residue's contribution (filled blue circle). Mutated positions are highlighted
in red. Negative values are more favorable.

Input CSV schema (one row per system x state x position):

    system,state,position,residue,dg[,se]
    WT,original,95,E,-3.21,0.42
    WT,optimized,95,N,-5.87,0.51
    ...

  system   : WT | BA.2.86 | JN.1
  state    : original | optimized
  position : integer residue number (WT CDR3 numbering)
  residue  : one-letter amino acid at that position in that state
  dg       : per-residue contribution to Delta G_bind (see --unit)
  se       : optional standard error across MD replicas; drawn as error bars

The per-residue MM-PBSA decomposition table is produced by the MD pipeline;
this script only handles presentation. Units are normalized to kcal/mol
(--unit kj divides by 4.184) to match the manuscript's MM-PBSA sections.

Usage:
    python scripts/figure5d_dumbbell.py --input figure5d_per_residue.csv \
        --out figures/figure5d
    python scripts/figure5d_dumbbell.py --demo --out /tmp/figure5d_demo

Exports both PDF and PNG. The demo uses synthetic values and stamps a
PLACEHOLDER watermark; never ship the demo output.
"""

import argparse
import csv
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SYSTEMS = ["WT", "BA.2.86", "JN.1"]
STATE_ORDER = ["original", "optimized"]

# House palette: blue primary, red accent, neutral gray for the reference.
COLOR_OPT = "#2166ac"
COLOR_ORIG = "#8c8c8c"
COLOR_MUT = "#b2182b"
COLOR_BAND = "#b2182b"
ZERO_LINE = "#444444"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#222222",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def read_table(path, unit):
    factor = 1.0 / 4.184 if unit == "kj" else 1.0
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"system", "state", "position", "residue", "dg"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"input is missing required columns: {sorted(missing)}")
        for row in reader:
            system = row["system"].strip()
            state = row["state"].strip().lower()
            if system not in SYSTEMS:
                raise ValueError(f"unknown system: {system!r} (expected one of {SYSTEMS})")
            if state not in STATE_ORDER:
                raise ValueError(f"unknown state: {state!r}")
            se_raw = (row.get("se") or "").strip()
            rows.append(
                {
                    "system": system,
                    "state": state,
                    "position": int(row["position"]),
                    "residue": row["residue"].strip().upper(),
                    "dg": float(row["dg"]) * factor,
                    "se": (float(se_raw) * factor) if se_raw else None,
                }
            )
    if not rows:
        raise ValueError("input table is empty")
    return rows


def organize(rows):
    positions = sorted({r["position"] for r in rows})
    wt_res = {}
    for r in rows:
        if r["state"] == "original":
            wt_res.setdefault(r["position"], r["residue"])
    for pos in positions:
        wt_res.setdefault(pos, "?")
    mutated = set()
    for pos in positions:
        letters = {r["residue"] for r in rows if r["position"] == pos}
        if len(letters) > 1:
            mutated.add(pos)
    table = {}
    for r in rows:
        table[(r["system"], r["state"], r["position"])] = r
    for system in SYSTEMS:
        for state in STATE_ORDER:
            for pos in positions:
                if (system, state, pos) not in table:
                    raise ValueError(f"missing row for {system} / {state} / position {pos}")
    return positions, wt_res, mutated, table


def demo_rows():
    """Synthetic values purely for layout preview. Not real data.

    Uses the manuscript's actual CDR3 sequence (KLENGGFFYY, positions 93-102)
    and mutation set (E95N, N96Y, F100G) so the layout matches the real panel.
    """
    rng = np.random.default_rng(7)
    wt_seq = {93: "K", 94: "L", 95: "E", 96: "N", 97: "G", 98: "G", 99: "F", 100: "F", 101: "Y", 102: "Y"}
    mut = {95: "N", 96: "Y", 100: "G"}
    rows = []
    for s_i, system in enumerate(SYSTEMS):
        for pos, aa in wt_seq.items():
            base = -6.0 + 4.0 * rng.random() + 0.4 * s_i
            gain = (1.5 + 2.0 * rng.random()) if pos in mut else (0.3 * rng.random() - 0.15)
            rows.append(dict(system=system, state="original", position=pos,
                             residue=aa, dg=base, se=0.3 + 0.2 * rng.random()))
            rows.append(dict(system=system, state="optimized", position=pos,
                             residue=mut.get(pos, aa), dg=base - gain,
                             se=0.3 + 0.2 * rng.random()))
    return rows


def draw(rows, out_prefix, demo=False, dpi=600):
    positions, wt_res, mutated, table = organize(rows)
    x = np.arange(len(positions))

    all_vals = [r["dg"] for r in rows]
    ymax = max(0.5, max(all_vals))
    ymin = min(-0.5, min(all_vals))
    pad = 0.12 * (ymax - ymin)
    ymax += pad
    ymin -= pad

    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    for ax, system in zip(axes, SYSTEMS):
        # red wash behind mutated positions
        for pos in sorted(mutated):
            ax.axvspan(x[positions.index(pos)] - 0.5, x[positions.index(pos)] + 0.5,
                       color=COLOR_BAND, alpha=0.07, lw=0, zorder=0)
        ax.axhline(0.0, color=ZERO_LINE, lw=0.7, zorder=1)

        orig = [table[(system, "original", p)] for p in positions]
        opt = [table[(system, "optimized", p)] for p in positions]
        yo = [r["dg"] for r in orig]
        yt = [r["dg"] for r in opt]

        for xi, (r0, r1) in enumerate(zip(orig, opt)):
            mut = r0["position"] in mutated
            ax.plot([xi, xi], [r0["dg"], r1["dg"]],
                    color=COLOR_MUT if mut else "#bdbdbd",
                    lw=1.4 if mut else 1.0, zorder=2)
            if r0["se"] is not None:
                ax.errorbar([xi], [r0["dg"]], yerr=[r0["se"]], fmt="none",
                            ecolor=COLOR_ORIG, elinewidth=0.6, capsize=1.5, zorder=3)
            if r1["se"] is not None:
                ax.errorbar([xi], [r1["dg"]], yerr=[r1["se"]], fmt="none",
                            ecolor=COLOR_OPT, elinewidth=0.6, capsize=1.5, zorder=3)

        ax.scatter(x, yo, s=26, facecolor="white", edgecolor=COLOR_ORIG,
                   linewidth=1.1, zorder=4, label="Original")
        ax.scatter(x, yt, s=30, facecolor=COLOR_OPT, edgecolor="white",
                   linewidth=0.6, zorder=5, label="Optimized")
        for xi, r1 in zip(x, opt):
            if r1["position"] in mutated:
                ax.scatter([xi], [r1["dg"]], s=64, facecolor="none",
                           edgecolor=COLOR_MUT, linewidth=1.1, zorder=6)

        labels = [f"{wt_res[p]}{p}" for p in positions]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        for tick, pos in zip(ax.get_xticklabels(), positions):
            if pos in mutated:
                tick.set_color(COLOR_MUT)
                tick.set_fontweight("bold")
        ax.set_title(system, fontsize=9, fontweight="bold", ha="center", pad=6)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.7, len(positions) - 0.3)
        ax.tick_params(length=2.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Per-residue contribution to $\\Delta G_{bind}$ (kcal/mol)")
    axes[0].annotate("more favorable ↓", xy=(0.02, 0.04),
                     xycoords="axes fraction", fontsize=6.5, color="#555555")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=7.5, handletextpad=0.4)

    if demo:
        fig.text(0.5, 0.5, "PLACEHOLDER DATA", rotation=28, fontsize=34,
                 color="#b2182b", alpha=0.18, ha="center", va="center", zorder=10)

    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", help="per-residue CSV (see module docstring)")
    parser.add_argument("--out", required=True, help="output prefix (.pdf/.png appended)")
    parser.add_argument("--unit", choices=["kcal", "kj"], default="kcal",
                        help="unit of the dg column in the input (default kcal/mol)")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--demo", action="store_true",
                        help="render with synthetic placeholder values and a watermark")
    args = parser.parse_args()

    if args.demo:
        rows = demo_rows()
    else:
        if not args.input:
            parser.error("--input is required unless --demo is used")
        rows = read_table(args.input, args.unit)

    draw(rows, args.out, demo=args.demo, dpi=args.dpi)
    print(f"wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
