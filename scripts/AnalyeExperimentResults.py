import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL FONT SETTINGS (larger fonts everywhere)
# ============================================================
plt.rcParams.update({
    "font.size": 20,            # base font size
    "axes.titlesize": 18,       # plot title
    "axes.labelsize": 18,       # x/y labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
    "figure.titlesize": 20
})

# ============================================================
# >>>>> HARD‑CODE YOUR PATHS HERE <<<<<
# ============================================================
CSV_PATH = r"C:\Users\lschrove\Desktop\PhD\PoC\POC_Weakspotidentification\Data\weakspot_experiments_master.csv"
OUT_DIR = r"C:\Users\lschrove\Desktop\PhD\PoC\POC_Weakspotidentification\Graphs"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Utility plotting
# ============================================================
def save_plot(fig, name):
    path = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

def _blue_purple_palette(n):
    if n <= 3:
        return sns.color_palette("Blues", 6)[2:2+n]
    nb = max(2, n // 2)
    npur = max(1, n - nb)
    blues = sns.color_palette("Blues", nb + 3)[2:]
    purps = sns.color_palette("Purples", npur + 3)[2:]
    out = []
    for i in range(max(len(blues), len(purps))):
        if i < len(blues): out.append(blues[i])
        if i < len(purps): out.append(purps[i])
    return out[:n]

# ============================================================
# Experiment grid & baseline
# ============================================================
EXPERIMENT_GRID = {
    "gaussian_std":        [0.0, 0.05, 0.10, 0.15],
    "heavy_tail_scale":    [0.0, 0.20],
    "heavy_tail_df":       [3, 8],
    "outlier_prob":        [0.0, 0.05],
    "outlier_magnitude":   [0.2, 0.5],
    "nonsmooth_prob":      [0.0, 0.05],
    "nonsmooth_magnitude": [0.2, 0.4],
    "nonuniform_skew":     [0.0, 0.5],
    "n_per_axis":          [10, 15],
}
DEGREES_REQUESTED = list(range(1, 10))
WEAKSPOTS = [(0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80)]

DEFAULT_BASELINE = {
    **{k: v[0] for k, v in EXPERIMENT_GRID.items()},
    "degree_requested": min(DEGREES_REQUESTED),
    "u0": WEAKSPOTS[0][0],
    "v0": WEAKSPOTS[0][1],
}

# ============================================================
# Baseline resolving
# ============================================================
def _resolve_baseline(df, characteristic=None):
    base = DEFAULT_BASELINE.copy()
    out = {}
    for col, desired in base.items():
        if col == characteristic:
            continue
        if col not in df.columns:
            continue
        if (df[col] == desired).any():
            out[col] = desired
        else:
            fallback = df[col].mode(dropna=True).iloc[0]
            print(f"[INFO] fallback baseline for {col} → {fallback}")
            out[col] = fallback
    return out

def _filter_to_baseline(df, base):
    mask = pd.Series(True, index=df.index)
    for col, val in base.items():
        if col in df.columns:
            mask &= (df[col] == val)
    out = df[mask].copy()
    if out.empty:
        raise ValueError("Baseline filtering removed all rows!")
    return out

# ============================================================
# Summaries
# ============================================================
def _compute_overall_by_method(df):
    return (
        df.groupby("method")
          .agg(mean_dist=("dist","mean"),
               mean_time=("algo_time_s","mean"),
               runs=("method","count"))
          .reset_index()
          .sort_values(["mean_dist"], ascending=True)
    )

def _compute_summary_by_level_and_method(df, characteristic):
    df2 = df.copy()
    df2["level"] = df2[characteristic]
    return (
        df2.groupby(["level","method"])
           .agg(mean_dist=("dist","mean"),
                mean_time=("algo_time_s","mean"),
                runs=("method","count"))
           .reset_index()
    )

def _print_rows_used_checker(df, label):
    print(f"\n=== Rows used per method [{label}] ===")
    counts = df.groupby("method").size()
    for m, c in counts.items():
        print(f"• {m}: {c} rows")

# ============================================================
# Global ordering helper
# ============================================================
GLOBAL_METHOD_ORDER = []

def _apply_global_method_order(methods):
    known = [m for m in GLOBAL_METHOD_ORDER if m in methods]
    unknown = [m for m in methods if m not in GLOBAL_METHOD_ORDER]
    return known + unknown

# ============================================================
# Twin bar plot (unchanged; takes explicit y-limits)
# ============================================================
def _plot_double_bars_twin(overall_df, filename_base, ylims=None, method_order=None):
    """
    Twin bar chart: distance (left y-axis) and runtime (right y-axis).
    Uses externally provided y-limits (ylims) to ensure consistent scaling
    and prevent bars from exceeding the axes.
    """
    dfp = overall_df.copy()

    # Apply method order if provided
    if method_order is not None:
        present = [m for m in method_order if m in dfp["method"].tolist()]
        dfp["method"] = pd.Categorical(dfp["method"], categories=present, ordered=True)
        dfp = dfp.sort_values("method")

    if dfp.empty:
        print(f"[WARN] Empty dataframe in twin plot: {filename_base}")
        return ((0, 1), (0, 1))

    # Colors
    COLOR_DIST = "#1f4e79"
    COLOR_TIME = "#6a5acd"

    # X-axis setup
    x_labels = dfp["method"].tolist()
    x = np.arange(len(x_labels))
    dist_vals = dfp["mean_dist"].values
    time_vals = dfp["mean_time"].values

    # Large figure for readability
    fig, ax_dist = plt.subplots(figsize=(14, 7))
    ax_time = ax_dist.twinx()

    # Plot distance bars (left axis)
    ax_dist.bar(x - 0.25, dist_vals, width=0.45, color=COLOR_DIST)

    # Plot runtime bars (right axis)
    ax_time.bar(x + 0.25, time_vals, width=0.45, color=COLOR_TIME)

    # Axis formatting
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=20)

    ax_dist.set_ylabel("Mean Distance", fontsize=22)
    ax_time.set_ylabel("Runtime (s)", fontsize=22)

    ax_dist.tick_params(axis="both", labelsize=18)
    ax_time.tick_params(axis="both", labelsize=18)

    # Apply global fixed limits if provided
    if ylims is not None:
        dist_lim, run_lim = ylims
        ax_dist.set_ylim(*dist_lim)
        ax_time.set_ylim(*run_lim)

    # Add grid
    ax_dist.grid(axis="y", linestyle="--", alpha=0.3)

    # Return final limits for debugging
    yd = ax_dist.get_ylim()
    yt = ax_time.get_ylim()

    # Save
    save_plot(fig, filename_base)
    return yd, yt

# ============================================================
# Line plot
# ============================================================
def _plot_levelX_methods_line(summary_df, characteristic, fname):
    methods = _apply_global_method_order(summary_df["method"].astype(str).unique().tolist())
    levels = summary_df["level"].dropna().unique().tolist()

    try:
        levels = sorted(levels, key=float)
    except Exception:
        levels = sorted(levels, key=str)

    pivot = summary_df.pivot(index="level", columns="method", values="mean_dist")
    colors = _blue_purple_palette(len(methods))
    x = np.arange(len(levels))

    fig, ax = plt.subplots(figsize=(12,6))
    for i, m in enumerate(methods):
        if m in pivot:
            y = pivot[m].reindex(levels).values
        else:
            y = np.repeat(np.nan,len(levels))
        ax.plot(x, y, marker="o", linewidth=2, color=colors[i], label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel(characteristic)
    ax.set_ylabel("Mean Distance")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_plot(fig, fname)

# ============================================================
# Boxplot sensitivity
# ============================================================
def _plot_boxplot_sensitivity(df_in, characteristic, fname):
    df2 = df_in.dropna(subset=[characteristic, "dist", "method"]).copy()
    methods = _apply_global_method_order(df2["method"].astype(str).unique().tolist())
    df2["method"] = pd.Categorical(df2["method"], categories=methods, ordered=True)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(data=df2, x="method", y="dist", color="#A7C7E7", ax=ax, fliersize=2)

    mean_df = df2.groupby("method")["dist"].mean().reindex(methods)
    ax.scatter(np.arange(len(methods)), mean_df.values, color="red",
               marker="D", s=60, label="Mean")

    ax.set_xlabel("Method")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_xticklabels(methods, rotation=45, ha="right")

    save_plot(fig, fname)

# ============================================================
# Sensitivity index bars
# ============================================================
def _plot_sensitivity_index_bars(summary_df, characteristic, fname):
    methods = _apply_global_method_order(summary_df["method"].astype(str).unique().tolist())
    levels = summary_df["level"].dropna().unique()

    try:
        levels_num = np.array(sorted(levels, key=float), dtype=float)
        numeric = True
    except Exception:
        numeric = False

    pivot = summary_df.pivot(index="level", columns="method", values="mean_dist")
    if numeric:
        pivot = pivot.reindex(index=sorted(pivot.index, key=float))

    sens_vals = []
    for m in methods:
        if m not in pivot:
            sens_vals.append(np.nan)
            continue
        s = pivot[m].dropna()
        if len(s) < 2:
            sens_vals.append(np.nan)
            continue
        if numeric:
            x = np.array([float(v) for v in s.index])
            y = s.values
            slope = np.polyfit(x, y, 1)[0]
            sens_vals.append(slope)
        else:
            sens_vals.append(s.max() - s.min())

    series = pd.Series(sens_vals, index=methods)

    fig, ax = plt.subplots(figsize=(10, 0.4*len(series)+4))
    ax.barh(series.index, series.values, color="#1f4e79")
    ax.set_xlabel("Sensitivity")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    save_plot(fig, fname)

# ============================================================
# Heatmap
# ============================================================
def plot_sensitivity_heatmap(sens_matrix, title, fname):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        sens_matrix,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Sensitivity"},
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, fname)

# ============================================================
# NEW: MODEL COMPLEXITY PLOT
# ============================================================
def plot_complexity_vs_method(df, fname="complexity_vs_method"):
    char = "degree_requested"
    df2 = df.dropna(subset=[char, "dist", "method"]).copy()

    summary = (
        df2.groupby([char, "method"])
           .agg(mean_dist=("dist", "mean"))
           .reset_index()
    )

    methods = _apply_global_method_order(summary["method"].unique().tolist())
    levels = sorted(summary[char].unique())

    pivot = summary.pivot(index=char, columns="method", values="mean_dist")

    colors = _blue_purple_palette(len(methods))
    x = np.arange(len(levels))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, m in enumerate(methods):
        y = pivot[m].reindex(levels).values if m in pivot else np.repeat(np.nan, len(levels))
        ax.plot(x, y, marker="o", linewidth=2, color=colors[i], label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Polynomial Degree (Model Complexity)")
    ax.set_ylabel("Mean Distance")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_plot(fig, fname)

    # ============================================================
# LOAD DATA
# ============================================================
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["x_pred", "dist", "algo_time_s"]).copy()
df["method"] = df["method"].astype(str)

df = df[df["method"] != "Observed min (data)"].copy()
print("Removed 'Observed min (data)'")
_print_rows_used_checker(df, "ALL")

# ============================================================
# OVERALL (FULL + BASELINE)
# ============================================================
overall_all = _compute_overall_by_method(df)

# Prepare baseline
baseline_filter_all = _resolve_baseline(df)
print("\n=== BASELINE CONFIGURATION USED ===")
for k, v in baseline_filter_all.items():
    print(f"{k}: {v}")

df_baseline = _filter_to_baseline(df, baseline_filter_all)
_print_rows_used_checker(df_baseline, "BASELINE")

overall_baseline = _compute_overall_by_method(df_baseline)

# Set global method ordering
GLOBAL_METHOD_ORDER = overall_all["method"].tolist()
print("\n=== GLOBAL METHOD ORDER ===")
for m in GLOBAL_METHOD_ORDER:
    print(" -", m)

# ============================================================
# 🔥 GLOBAL Y-LIMITS FOR OVERVIEW PLOTS (NO CUT-OFF EVER)
# ============================================================
# Distance axis (left)
dist_min = min(overall_all["mean_dist"].min(), overall_baseline["mean_dist"].min())
dist_max = max(overall_all["mean_dist"].max(), overall_baseline["mean_dist"].max())
dist_range = dist_max - dist_min

dist_lower = max(0, dist_min - 0.10 * dist_range)
dist_upper = dist_max + 0.20 * dist_range

# Runtime axis (right)
rt_min = min(overall_all["mean_time"].min(), overall_baseline["mean_time"].min())
rt_max = max(overall_all["mean_time"].max(), overall_baseline["mean_time"].max())
rt_range = rt_max - rt_min

# guarantee headroom on axis
rt_lower = max(0, rt_min - 0.10 * rt_range)
rt_upper = rt_max + 0.25 * rt_range

GLOBAL_OVERVIEW_LIMITS = ((dist_lower, dist_upper), (rt_lower, rt_upper))

print("\n=== GLOBAL OVERVIEW LIMITS (APPLIED TO ALL OVERVIEW PLOTS) ===")
print("Distance:", GLOBAL_OVERVIEW_LIMITS[0])
print("Runtime:", GLOBAL_OVERVIEW_LIMITS[1])

# ============================================================
# 🔥 OVERVIEW PLOTS USING FIXED GLOBAL LIMITS
# ============================================================
_plot_double_bars_twin(
    overall_all,
    filename_base="overview_all_distance_runtime_twin",
    ylims=GLOBAL_OVERVIEW_LIMITS,
    method_order=GLOBAL_METHOD_ORDER
)

_plot_double_bars_twin(
    overall_baseline,
    filename_base="overview_baseline_distance_runtime_twin",
    ylims=GLOBAL_OVERVIEW_LIMITS,
    method_order=GLOBAL_METHOD_ORDER
)

_plot_double_bars_twin(
    overall_all,
    filename_base="overview_all_distance_runtime_twin_baseline_order",
    ylims=GLOBAL_OVERVIEW_LIMITS,
    method_order=GLOBAL_METHOD_ORDER
)

# ============================================================
# CHARACTERISTIC-WISE
# ============================================================
for char in EXPERIMENT_GRID.keys():
    print(f"\n=== Sensitivity FULL: {char} ===")
    df_full = df.dropna(subset=[char]).copy()
    _print_rows_used_checker(df_full, f"FULL {char}")

    summary_full = _compute_summary_by_level_and_method(df_full, char)

    _plot_levelX_methods_line(summary_full, char, f"{char}_FULL_line_levelsX")
    _plot_boxplot_sensitivity(df_full, char, f"{char}_FULL_boxplot_methods")
    _plot_sensitivity_index_bars(summary_full, char, f"{char}_FULL_sensitivity_index")

    print(f"\n=== Sensitivity BASELINE: {char} ===")
    baseline_filter_char = _resolve_baseline(df, characteristic=char)
    df_char_base = _filter_to_baseline(df, baseline_filter_char)
    _print_rows_used_checker(df_char_base, f"BASE {char}")

    summary_base = _compute_summary_by_level_and_method(df_char_base, char)

    _plot_levelX_methods_line(summary_base, char, f"{char}_BASE_line_levelsX")
    _plot_boxplot_sensitivity(df_char_base, char, f"{char}_BASE_boxplot_methods")
    _plot_sensitivity_index_bars(summary_base, char, f"{char}_BASE_sensitivity_index")

# ============================================================
# HEATMAPS
# ============================================================
methods = GLOBAL_METHOD_ORDER
chars = list(EXPERIMENT_GRID.keys())

# FULL heatmap
sens_full = pd.DataFrame(index=methods, columns=chars, dtype=float)

for char in chars:
    df_full = df.dropna(subset=[char]).copy()
    summary_full = _compute_summary_by_level_and_method(df_full, char)
    pivot = summary_full.pivot(index="level", columns="method", values="mean_dist")

    for m in methods:
        if m not in pivot:
            sens_full.loc[m, char] = np.nan
            continue
        s = pivot[m].dropna()
        if len(s) < 2:
            sens_full.loc[m, char] = np.nan
            continue
        try:
            x = np.array([float(v) for v in s.index])
            y = s.values
            slope = np.polyfit(x, y, 1)[0]
            sens_full.loc[m, char] = slope
        except:
            sens_full.loc[m, char] = s.max() - s.min()

plot_sensitivity_heatmap(
    sens_full,
    " ",
    "heatmap_FULL_sensitivity"
)

# BASELINE heatmap
sens_base = pd.DataFrame(index=methods, columns=chars, dtype=float)

for char in chars:
    baseline_char = _resolve_baseline(df, characteristic=char)
    df_char_base = _filter_to_baseline(df, baseline_char)
    summary_base = _compute_summary_by_level_and_method(df_char_base, char)
    pivot = summary_base.pivot(index="level", columns="method", values="mean_dist")

    for m in methods:
        if m not in pivot:
            sens_base.loc[m, char] = np.nan
            continue
        s = pivot[m].dropna()
        if len(s) < 2:
            sens_base.loc[m, char] = np.nan
            continue
        try:
            x = np.array([float(v) for v in s.index])
            y = s.values
            slope = np.polyfit(x, y, 1)[0]
            sens_base.loc[m, char] = slope
        except:
            sens_base.loc[m, char] = s.max() - s.min()

plot_sensitivity_heatmap(
    sens_base,
    " ",
    "heatmap_BASE_sensitivity"
)

# ============================================================
# NEW: COMPLEXITY-PLOT CALL
# ============================================================
print("\n=== Plotting METHODS vs MODEL COMPLEXITY (degree_requested) ===")
plot_complexity_vs_method(df, fname="model_complexity_all")
print("Saved: model_complexity_all.png")

# ============================================================
# DONE
# ============================================================
print("\n=== COMPLETE ===")
print("All output saved to:", OUT_DIR)