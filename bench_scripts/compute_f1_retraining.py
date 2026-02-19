#!/usr/bin/env python3

import argparse
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# FIX ORDEN OF SUPERFAMILIES (to use NORMALIZED labels)
SUPERFAMILY_ORDER = [
    "LTR",
    "BELPAO",
    "COPIA",
    "ERV",
    "GYPSY",
    "LARD",
    "LINE",
    "CR1",
    "I",
    "L1",
    "R1",
    "RTE",
    "SINE",
    "ALU",
    "TRNA",
    "DIRS",
    "PLE",
    "TIR",
    "CACTA",
    "HAT",
    "MERLIN",
    "MULE",
    "P",
    "PIFHARBINGER",
    "PIGGYBAC",
    "TC1MARINER",
    "ACADEM1",
    "KOLOBOK",
    "CRYPTON",
    "HELITRON",
    "MAVERICK",
]

# Proper "pretty" names
PRETTY_MAP = {
    "TC1MARINER": "Tc1-Mariner",
    "HAT": "hAT",
    "MITES": "MITE",
    "HELITRON": "Helitron",
    "BELPAO": "Bel-Pao",
    "TRNA": "t-RNA",
    "PIFHARBINGER": "PIF-Harbinger",
    "GYPSY": "Gypsy",
    "COPIA": "Copia",
    "JOCKEY": "Jockey",
    "ALU": "Alu",
    "ACADEM1": "Academ-1"
}

def normalize_label(label: str) -> str:
    if label is None:
        return ""
    s = str(label).strip()
    if "/" in s:
        s = s.split("/")[-1]
    s = s.upper()
    s = s.replace(" ", "")
    s = re.sub(r"[^A-Z0-9]+", "", s)

    if s == "LINE1":
        s = "L1"
    if s in {"TC1", "TC1MARINER"} or s.startswith("TC1MARINER"):
        s = "TC1MARINER"
    if s in {"PIFHARBINGER", "PIFHARBI", "HARBI", "PIFHARB"}:
        s = "PIFHARBINGER"
    if s.startswith("DIRS"):
        s = "DIRS"
    if s.startswith("ERV"):
        s = "ERV"
    return s

def pretty_label(norm: str) -> str:
    return PRETTY_MAP.get(norm, norm)

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate an F1 heatmap + support bar plot and Critical Difference (CD) diagram from a wide-format Excel file."
    )
    p.add_argument("input", help="Input Excel file (.xlsx/.xls).")
    p.add_argument("-o", "--output", default="f1_superfamily_by_tool.csv",
                   help="Output CSV file (long format) with Tool, Superfamily, F1, and Support.")
    p.add_argument("--outdir", default="figures", help="Output directory for the figures.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level for CD (Nemenyi test).")
    p.add_argument("--formats", nargs="+", default=["pdf"], choices=["pdf", "png", "svg"],
                   help="Figure output formats.")
    p.add_argument("--sheet", default=0, help="Excel sheet (name or index).")
    return p.parse_args()


def read_f1_wide_excel_with_support(path, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet, header=0)

    colmap = {c: str(c) for c in df.columns}
    low = {str(c).lower(): c for c in df.columns}

    if "superfamily" not in low:
        raise ValueError("No se encontró la columna 'Superfamily' en el Excel.")
    sf_col = low["superfamily"]

    support_col = low.get("support", None)
    if support_col is None:
        raise ValueError("No se encontró la columna 'Support' en el Excel.")

    tool_cols = [c for c in df.columns if c not in (sf_col, support_col) and str(c).upper() != "ID"]
    if not tool_cols:
        raise ValueError("No se detectaron columnas de herramientas con F1.")

    rows = []
    for _, r in df.iterrows():
        sf_raw = r[sf_col]
        sf_norm = normalize_label(sf_raw)
        sf_pretty = pretty_label(sf_norm)

        sup_val = r[support_col]
        try:
            sup_val = int(sup_val)
        except Exception:
            try:
                sup_val = int(float(sup_val))
            except Exception:
                sup_val = np.nan

        for tool in tool_cols:
            val = r[tool]
            try:
                f1 = float(val)
            except Exception:
                f1 = np.nan
            rows.append({
                "Tool": str(tool),
                "Superfamily_norm": sf_norm,
                "Superfamily": sf_pretty,
                "F1": f1,
                "Support": sup_val,
            })

    df_long = pd.DataFrame(rows)

    tools_sorted = sorted(df_long["Tool"].unique(), key=lambda x: x.upper())
    df_long["Tool"] = pd.Categorical(df_long["Tool"], categories=tools_sorted, ordered=True)
    df_long = df_long.sort_values(["Tool", "Superfamily"])
    df_long["Tool"] = df_long["Tool"].astype(str)
    return df_long, tools_sorted

def rank_of_superfamily(sf_norm: str) -> float:
    if sf_norm in SUPERFAMILY_ORDER:
        return SUPERFAMILY_ORDER.index(sf_norm)
    return len(SUPERFAMILY_ORDER) + sorted([sf_norm]).index(sf_norm)

def plot_heatmap_with_support(df_f1, outdir, formats=("pdf",), cmap="viridis"):
    mat = df_f1.pivot(index="Superfamily_norm", columns="Tool", values="F1")

    support = df_f1.groupby("Superfamily_norm")["Support"].max().reindex(mat.index)

    col_order = sorted(mat.columns.tolist(), key=lambda x: str(x).upper())
    mat = mat[col_order]

    known = [sf for sf in SUPERFAMILY_ORDER if sf in mat.index]
    unknown = sorted([sf for sf in mat.index if sf not in SUPERFAMILY_ORDER], key=lambda x: x.upper())
    row_order = known + unknown
    mat = mat.loc[row_order]
    support = support.loc[row_order]

    ylabels_pretty = [pretty_label(sf) for sf in row_order]

    fig = plt.figure(figsize=(10.2, max(4.8, 0.3 * len(row_order))))
    gs = fig.add_gridspec(nrows=1, ncols=10, wspace=0.15, left=0.08, right=0.95, top=0.9, bottom=0.08)
    ax_hm = fig.add_subplot(gs[0, :8])
    ax_sup = fig.add_subplot(gs[0, 8:])

    sns.heatmap(mat, ax=ax_hm, cmap=cmap, vmin=0.0, vmax=1.0, cbar=True,
                cbar_kws={"shrink": 0.6, "label": "F1"},
                linewidths=0.0, linecolor="none", square=False)

    for i, sf in enumerate(mat.index):
        row_vals = mat.loc[sf].values
        if np.all(np.isnan(row_vals)):
            continue
        max_idx = np.nanargmax(row_vals)
        ax_hm.scatter(max_idx + 0.5, i + 0.5, s=18, marker="o",
                      edgecolor="black", facecolor="white", linewidth=0.6)

    ax_hm.set_xlabel("Tool", fontsize=20)
    ax_hm.set_ylabel("Superfamily", fontsize=20)
    ax_hm.set_yticklabels(ylabels_pretty, rotation=0, fontsize=14)
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax_hm.set_title(f"", pad=12, fontsize=32)

    y_pos = np.arange(len(row_order))
    ax_sup.barh(y_pos, support.values, align="center")
    ax_sup.set_ylim(ax_hm.get_ylim())
    ax_sup.set_yticks([])  # Share labels with the heatmap
    ax_sup.set_xlabel("Support (n)")
    ax_sup.set_title("", fontsize=10)

    for fmt in formats:
        outpath = os.path.join(outdir, f"panel_A_heatmap_soporte.{fmt}")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def mean_ranks_from_matrix(mat):
    tools = list(mat.columns)
    ranks_sum = {t: 0.0 for t in tools}
    nrows = 0

    for _, row in mat.iterrows():
        vals = row.values.astype(float)
        if np.all(np.isnan(vals)):
            continue
        vals_clean = vals.copy()
        nan_mask = np.isnan(vals_clean)
        if np.any(~nan_mask):
            min_val = np.nanmin(vals_clean)
        else:
            continue
        vals_clean[nan_mask] = min_val - 1.0

        order = np.argsort(-vals_clean)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(vals_clean) + 1)

        for v in np.unique(vals_clean):
            idx = np.where(vals_clean == v)[0]
            if len(idx) > 1 and v != (min_val - 1.0):
                avg = np.mean(ranks[idx])
                ranks[idx] = avg

        for j, t in enumerate(tools):
            if not np.isnan(row[t]):
                ranks_sum[t] += ranks[j]
        nrows += 1

    mean_ranks = {t: (ranks_sum[t] / nrows if nrows > 0 else np.nan) for t in tools}
    return mean_ranks, nrows

def compute_cd(k, N, alpha=0.05):
    if N <= 0:
        raise ValueError("There are no valid rows (superfamilies) to compute the critical difference (CD).")

    q = float(studentized_range.isf(alpha, k, np.inf) / math.sqrt(2.0))
    return q * math.sqrt(k * (k + 1) / (6.0 * N))

def group_nonsignificant(mean_ranks_sorted, CD):
    segments = []
    n = len(mean_ranks_sorted)
    i = 0
    while i < n - 1:
        jmax = i
        for j in range(i + 1, n):
            rmin = mean_ranks_sorted[i][1]
            rmax = mean_ranks_sorted[j][1]
            if (rmax - rmin) <= CD:
                jmax = j
            else:
                break
        if jmax > i:
            segments.append((mean_ranks_sorted[i][1], mean_ranks_sorted[jmax][1]))
        i += 1

    cleaned = []
    for s in segments:
        if not any((s[0] >= t[0] and s[1] <= t[1]) and s != t for t in segments):
            cleaned.append(s)

    levels = []
    for s in cleaned:
        level = 0
        while any((max(s[0], t[0]) <= min(s[1], t[1])) and (lvl == level) for (t, lvl) in levels):
            level += 1
        levels.append((s, level))
    return [(s[0], s[1], lvl) for (s, lvl) in levels]

def plot_critical_difference(df_f1, outdir, alpha=0.05, formats=("pdf",)):
    mat = df_f1.pivot(index="Superfamily_norm", columns="Tool", values="F1")
    mean_ranks, N = mean_ranks_from_matrix(mat)
    tools = list(mean_ranks.keys())
    k = len(tools)
    if k < 2:
        raise ValueError("At least 2 tools are required for the critical difference (CD).")

    CD = compute_cd(k, N, alpha=alpha)
    items = sorted(mean_ranks.items(), key=lambda x: x[1])
    tool_ranks = [r for _, r in items]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_title("", pad=10, fontsize=11)

    ax.set_xlim(1 - 0.1, k + 0.1)
    ax.set_ylim(-1.2, 2.5)
    ax.hlines(y=0, xmin=1, xmax=k, color="black", linewidth=1.2)

    y_cd = 1.6
    left = (k - CD) / 2.0
    right = left + CD
    ax.hlines(y=y_cd, xmin=left, xmax=right, color="black", linewidth=2)
    ax.vlines([left, right], ymin=y_cd - 0.06, ymax=y_cd + 0.06, color="black", linewidth=2)
    ax.text((left + right) / 2.0, y_cd + 0.2, f"CD (α={alpha:.2g}) = {CD:.2f}",
            ha="center", va="bottom", fontsize=9)

    ax.scatter(tool_ranks, [0] * k, s=28, color="black", zorder=3)

    for i, (name, rank) in enumerate(items):
        if i % 2 == 0:
            ax.text(rank, -0.25, name, ha="center", va="top", fontsize=9)
        else:
            ax.text(rank, 0.25, name, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(np.arange(1, k + 1))
    ax.set_xticklabels([str(i) for i in range(1, k + 1)])
    ax.set_yticks([])
    ax.set_xlabel("Rank (1 = Best)")

    segs = group_nonsignificant(items, CD)     for (x0, x1, lvl) in segs:
        y = 0.9 - lvl * 0.18
        ax.hlines(y=y, xmin=x0, xmax=x1, color="black", linewidth=3, alpha=0.7)

    for fmt in formats:
        outpath = os.path.join(outdir, f"panel_C_critical_difference.{fmt}")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _assign_label_levels(xvals, min_dx=0.35, levels=(-0.25, 0.25, -0.50, 0.50, -0.75, 0.75)):
    last_at_level = [-1e9 for _ in levels]
    y_offsets = []
    segs = []
    for x in xvals:

        chosen = None
        for li, lastx in enumerate(last_at_level):
            if abs(x - lastx) >= min_dx:
                chosen = li
                break
        if chosen is None:
            chosen = len(levels) - 1
        last_at_level[chosen] = x
        y = levels[chosen]
        y_offsets.append(y)
        segs.append(((x, 0.0), (x, y + (0.06 if y > 0 else -0.06))))
    return y_offsets, segs


def plot_critical_difference(df_f1, outdir, alpha=0.05, formats=("pdf",)):
    mat = df_f1.pivot(index="Superfamily_norm", columns="Tool", values="F1")
    mean_ranks, N = mean_ranks_from_matrix(mat)
    tools = list(mean_ranks.keys())
    k = len(tools)
    if k < 2:
        raise ValueError("It is required at least 2 tools for the CD.")

    CD = compute_cd(k, N, alpha=alpha)
    items = sorted(mean_ranks.items(), key=lambda x: x[1])
    tool_names = [n for n, _ in items]
    tool_ranks = [r for _, r in items]

    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.set_xlim(1 - 0.1, k + 0.1)
    ax.set_ylim(-1.2, 2.5)
    ax.hlines(y=0, xmin=1, xmax=k, color="black", linewidth=1.2)

    y_cd = 1.6
    left = (k - CD) / 2.0
    right = left + CD
    ax.hlines(y=y_cd, xmin=left, xmax=right, color="black", linewidth=2)
    ax.vlines([left, right], ymin=y_cd - 0.06, ymax=y_cd + 0.06, color="black", linewidth=2)
    ax.text((left + right) / 2.0, y_cd + 0.2, f"CD (α={alpha:.2g}) = {CD:.2f}",
            ha="center", va="bottom", fontsize=9)

    ax.scatter(tool_ranks, [0] * k, s=30, color="black", zorder=3)

    y_offsets, leaders = _assign_label_levels(tool_ranks, min_dx=0.40)
    for (name, x, yoff) in zip(tool_names, tool_ranks, y_offsets):
        ax.text(x, yoff, name, ha="center",
                va="bottom" if yoff > 0 else "top", fontsize=9)
    
    for (x0, y0), (x1, y1) in leaders:
        ax.plot([x0, x1], [y0, y1], lw=0.8, color="black", alpha=0.8, zorder=2)

    ax.set_xticks(np.arange(1, k + 1))
    ax.set_xticklabels([str(i) for i in range(1, k + 1)])
    ax.set_yticks([])
    ax.set_xlabel("Rank (1 = Best)")

    segs = group_nonsignificant(items, CD)
    for (x0, x1, lvl) in segs:
        y = 0.95 - lvl * 0.18
        ax.hlines(y=y, xmin=x0, xmax=x1, color="black", linewidth=3, alpha=0.7)

    for fmt in formats:
        outpath = os.path.join(outdir, f"panel_C_critical_difference.{fmt}")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

from datetime import datetime

RELEASES = {
    # Tool : "Month Year"
    "BERTE": "January 2024",
    "ClassifyTE": "September 2021",
    "CREATE": "January 2024",
    "DeepTE": "August 2020",
    "Inpactor2_Class": "January 2023",     
    "NeuralTE": "December 2024",
    "TEClass2": "October 2023",
    "TERL": "May 2021",
    "Terrier": "March 2025",
}

def _norm_tool_name(t: str) -> str:
    s = str(t).replace(" ", "").replace("-", "").replace(".", "")
    s = s.replace("Classify", "Class")
    return s


def plot_timeline_rank_vs_release(df_f1, outdir, formats=("pdf",)):

    from sklearn.linear_model import LinearRegression
    import numpy as np

    mat = df_f1.pivot(index="Superfamily_norm", columns="Tool", values="F1")
    mean_ranks, N = mean_ranks_from_matrix(mat)

    rows = []
    for tool, rank in mean_ranks.items():
        match = None
        for k in RELEASES.keys():
            if _norm_tool_name(tool).lower() == _norm_tool_name(k).lower():
                match = k
                break
        if match is None:
            continue
        date_str = RELEASES[match]
        dt = datetime.strptime(date_str, "%B %Y")
        rows.append((match, dt, rank))

    if not rows:
        return

    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    dates = [r[1] for r in rows]
    ranks = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.scatter(dates, ranks, s=30, color="black", zorder=3)
    ax.invert_yaxis()
    ax.set_ylabel("Mean rank (1 = Best)")
    ax.set_xlabel("Release date")

    fig.autofmt_xdate(rotation=30, ha="right")

    import matplotlib.dates as mdates
    xnum = mdates.date2num(dates)
    y_offsets, leaders = _assign_label_levels(
        xnum, min_dx=25, levels=(-0.25, 0.25, -0.45, 0.45, -0.65, 0.65)
    )
    for (name, x, yoff) in zip(names, dates, y_offsets):
        ax.text(
            x,
            ranks[names.index(name)] + yoff,
            name,
            ha="center",
            va="bottom" if yoff > 0 else "top",
            fontsize=9,
        )

    for (x0, y0), (x1, y1) in leaders:
        ax.plot(
            [mdates.num2date(x0), mdates.num2date(x1)],
            [
                ranks[xnum.tolist().index(x0)],
                ranks[xnum.tolist().index(x0)]
                + (0.06 if (y1 - y0) > 0 else -0.06),
            ],
            lw=0.8,
            color="black",
            alpha=0.8,
            zorder=2,
        )

    ax.hlines(
        y=np.mean(ranks),
        xmin=min(dates),
        xmax=max(dates),
        color="gray",
        ls="--",
        lw=1,
        alpha=0.6,
    )

    x = xnum.reshape(-1, 1)
    y = np.array(ranks)

    if len(x) >= 2:
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x)
        r2 = reg.score(x, y)
        slope_per_year = reg.coef_[0] * 365  # Rank change per year

        order = np.argsort(xnum)
        ax.plot(
            np.array(dates)[order],
            y_pred[order],
            color="red",
            lw=1.8,
            alpha=0.8,
            label=f"Slope={slope_per_year:.3f}/year, R²={r2:.2f}",
        )
        ax.legend(frameon=False, fontsize=9)

        print(f"Slope ≈ {slope_per_year:.3f} rank units / year (R²={r2:.3f})")

    for fmt in formats:
        outpath = os.path.join(outdir, f"panel_D_timeline_rank_vs_release.{fmt}")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_f1, tools_sorted = read_f1_wide_excel_with_support(args.input, sheet=args.sheet)

    df_out = df_f1[["Tool", "Superfamily_norm", "Superfamily", "F1", "Support"]].copy()
    df_out["__rank"] = df_out["Superfamily_norm"].apply(rank_of_superfamily)
    
    df_out = df_out.sort_values(
        by=["Tool", "__rank", "Superfamily"],
        key=lambda col: col if col.name != "Tool" else col.str.upper()
    )
    df_out = df_out.drop(columns=["__rank", "Superfamily_norm"])
    df_out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"   CSV saved: {args.output}")
    print(f"   Detected tools: {', '.join(tools_sorted)}")
    print(f"   Superfamilies (unique): {df_out['Superfamily'].nunique()} | Total files: {len(df_out)}")

    plot_heatmap_with_support(df_f1, outdir=args.outdir, formats=tuple(args.formats))
    print(f"  Figure Panel A+B (heatmap+soporte) at: {args.outdir}")

    try:
        plot_critical_difference(df_f1, outdir=args.outdir, alpha=args.alpha, formats=tuple(args.formats))
        print(f"  Figure Panel C (CD) at: {args.outdir}")
    except RuntimeError as e:
        print("  It wasn't generated Panel C (CD):", str(e))

    plot_timeline_rank_vs_release(df_f1, outdir=args.outdir, formats=tuple(args.formats))
    print(f"  Figure Panel D (timeline) at: {args.outdir}")

if __name__ == "__main__":
    main()
