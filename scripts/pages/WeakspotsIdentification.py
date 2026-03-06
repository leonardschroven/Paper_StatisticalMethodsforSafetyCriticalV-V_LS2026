import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# >>>> HARD‑CODE YOUR PATHS HERE <<<<
# ============================================================
CSV_PATH = r"C:\Users\lschrove\Desktop\PhD\PoC\POC_Weakspotidentification\Data\weakspot_experiments_master.csv"
OUT_DIR = r"C:\Users\lschrove\Desktop\PhD\PoC\POC_Weakspotidentification\Graphs"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Utility Plotting Function (shared)
# ============================================================
def save_plot(fig, name):
    path = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

# ============================================================
# === Helpers (moved to top and extended) ====================
# ============================================================

def _export_table_png(df_table: pd.DataFrame, filename: str, title: str):
    """
    Export a compact table PNG. Works for both overall and characteristic tables.
    If 'level' exists, it will be shown; otherwise it's ignored.
    """
    rename_map = {
        "method": "Algorithm",
        "level": "Level",
        "mean_dist": "Mean Distance",
        "mean_time": "Mean Runtime (s)",
        "runs": "Runs"
    }
    tbl = df_table.rename(columns={k: v for k, v in rename_map.items() if k in df_table.columns}).copy()

    # Round numeric columns for legibility
    if "Mean Distance" in tbl.columns:
        tbl["Mean Distance"] = pd.to_numeric(tbl["Mean Distance"], errors="coerce").round(6)
    if "Mean Runtime (s)" in tbl.columns:
        tbl["Mean Runtime (s)"] = pd.to_numeric(tbl["Mean Runtime (s)"], errors="coerce").round(3)
    if "Runs" in tbl.columns:
        tbl["Runs"] = pd.to_numeric(tbl["Runs"], errors="coerce").astype("Int64")

    # Dynamic sizing
    fig_w = min(16, 2 + 0.24 * len(tbl.columns))
    fig_h = max(1.2, 1.0 + 0.42 * max(1, len(tbl)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl_obj = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        loc="center",
        cellLoc="center",
    )
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(9)
    tbl_obj.scale(1, 1.2)

    # Header style
    for (r, c), cell in tbl_obj.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    plt.title(title, pad=10)
    save_plot(fig, filename)


def _plot_side_by_side(overall_df: pd.DataFrame, png_name: str, title: str):
    """
    Overall side-by-side bars with two y-axes (distance & time).
    """
    COLOR_DIST = "#1f4e79"   # dark blue
    COLOR_TIME = "#6aaed6"   # light blue

    x_labels = overall_df["method"].astype(str).tolist()
    x = np.arange(len(x_labels))
    dist_vals = overall_df["mean_dist"].astype(float).values
    time_vals = overall_df["mean_time"].astype(float).values

    fig, ax_dist = plt.subplots(figsize=(12, 6))
    ax_time = ax_dist.twinx()
    bar_w = 0.38

    bars_dist = ax_dist.bar(x - bar_w/2, dist_vals, width=bar_w,
                            color=COLOR_DIST, alpha=0.95, edgecolor="none",
                            label="Mean Distance")
    bars_time = ax_time.bar(x + bar_w/2, time_vals, width=bar_w,
                            color=COLOR_TIME, alpha=0.95, edgecolor="none",
                            label="Mean Runtime (s)")

    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_dist.set_xlabel("Algorithm (method)")

    ax_dist.set_ylabel("Mean Distance to True Minimum", color=COLOR_DIST)
    ax_time.set_ylabel("Mean Runtime (s)", color=COLOR_TIME)
    ax_dist.tick_params(axis='y', labelcolor=COLOR_DIST)
    ax_time.tick_params(axis='y', labelcolor=COLOR_TIME)
    ax_dist.grid(axis="y", linestyle="--", alpha=0.3)

    ax_dist.legend([bars_dist, bars_time], ["Mean Distance", "Mean Runtime (s)"],
                   loc="upper left", frameon=True)

    plt.title(title)
    save_plot(fig, png_name)


# =====================
# EXPERIMENT GRID (used to define "simplest" baseline values)
# =====================
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

# Define defaults for "simplest" case
DEFAULT_BASELINE = {
    **{k: v[0] for k, v in EXPERIMENT_GRID.items()},
    "degree_requested": min(DEGREES_REQUESTED) if len(DEGREES_REQUESTED) else 1,
    "u0": WEAKSPOTS[0][0],
    "v0": WEAKSPOTS[0][1],
}

def _resolve_baseline(df: pd.DataFrame, characteristic: str, baseline_overrides=None):
    """
    Build the baseline filter dict for all non-target characteristics.
    If a chosen baseline value is absent in data, fallback to the most frequent value.
    """
    baseline_overrides = baseline_overrides or {}
    baseline = DEFAULT_BASELINE.copy()
    baseline.update(baseline_overrides)

    to_fix = [c for c in baseline.keys() if c in df.columns and c != characteristic]
    fixed = {}
    for col in to_fix:
        desired = baseline[col]
        if (df[col] == desired).any():
            fixed[col] = desired
        else:
            mode_vals = df[col].mode(dropna=True)
            fallback = mode_vals.iloc[0] if len(mode_vals) else df[col].dropna().iloc[0]
            fixed[col] = fallback
            print(f"[INFO] Baseline for '{col}' not found; using fallback mode: {fallback}")
    return fixed

def _filter_to_baseline(df: pd.DataFrame, characteristic: str, baseline_filter: dict) -> pd.DataFrame:
    """
    Filter df so that all columns in baseline_filter are fixed, but the target characteristic varies.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in baseline_filter.items():
        if col == characteristic:  # skip target
            continue
        if col in df.columns:
            mask &= (df[col] == val)
    df_simple = df.loc[mask].copy()
    if df_simple.empty:
        raise ValueError("[ERROR] No rows after applying baseline filter. "
                         "Adjust baseline_overrides or verify your data.")
    return df_simple

def _compute_summary_by_level_and_method(df_simple: pd.DataFrame, characteristic: str) -> pd.DataFrame:
    """
    Aggregate mean distance and mean runtime per (level, method).
    Output columns: ['level', 'method', 'mean_dist', 'mean_time', 'runs']
    """
    if characteristic not in df_simple.columns:
        raise KeyError(f"[ERROR] Column '{characteristic}' not in data.")
    df_simple = df_simple.copy()
    df_simple["level"] = df_simple[characteristic]

    grp = (
        df_simple
        .groupby(["level", "method"], dropna=False)
        .agg(
            mean_dist=("dist", "mean"),
            mean_time=("algo_time_s", "mean"),
            runs=("method", "count"),
        )
        .reset_index()
        .sort_values(["level", "mean_dist", "mean_time"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return grp

def _fairness_check_counts(df_simple: pd.DataFrame, characteristic: str):
    """
    Build counts per (level, method), determine fairness per level.
    Returns counts df and a title suffix string.
    """
    counts = (
        df_simple
        .groupby(["method", characteristic], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .rename(columns={characteristic: "level"})
        .sort_values(["level", "rows"], ascending=[True, False])
    )
    lvl_fair = (
        counts
        .groupby("level")["rows"]
        .apply(lambda s: len(s.unique()) == 1)
        .rename("fair")
        .reset_index()
    )
    all_fair = bool(lvl_fair["fair"].all())
    min_rows = int(counts["rows"].min()) if len(counts) else 0
    max_rows = int(counts["rows"].max()) if len(counts) else 0

    if all_fair:
        suffix = " — Fair (Same N per Method at Each Level)"
        print("[OK] Fairness: within each level, all methods have same N.")
    else:
        suffix = f" — NOT Fair (N per method varies: min={min_rows}, max={max_rows})"
        print("[WARN] Fairness: unequal #rows for at least one level.")
        print("[INFO] See counts CSV for per-level details.")
    return counts, suffix

def _plot_grouped_bars_two_panels(summary_df: pd.DataFrame, characteristic: str, title: str, filename_base: str):
    """
    Two panels: top = mean distance, bottom = mean runtime; grouped by method across levels.
    """
    levels = summary_df["level"].dropna().unique().tolist()
    # Sort levels numerically if possible, else as strings
    try:
        levels = sorted(levels, key=float)
    except Exception:
        levels = sorted(levels, key=lambda x: str(x))

    methods = summary_df["method"].dropna().unique().tolist()

    pivot_dist = summary_df.pivot(index="level", columns="method", values="mean_dist").reindex(levels)
    pivot_time = summary_df.pivot(index="level", columns="method", values="mean_time").reindex(levels)

    num_levels = len(levels)
    num_methods = len(methods)

    fig_w = min(20, 6 + 0.4 * num_levels + 0.15 * num_methods)
    fig_h = 6 + (0.6 if num_methods > 6 else 0.0)
    fig, (ax_dist, ax_time) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

    colors = plt.cm.get_cmap("tab20", max(1, num_methods))
    x = np.arange(num_levels)
    total_group_width = min(0.85, 0.2 + 0.03 * num_methods)
    bar_w = total_group_width / max(1, num_methods)

    # Distance panel
    for i, m in enumerate(methods):
        vals = pivot_dist[m].values if m in pivot_dist.columns else np.repeat(np.nan, num_levels)
        offs = x - total_group_width / 2 + (i + 0.5) * bar_w
        ax_dist.bar(offs, vals, width=bar_w, color=colors(i), alpha=0.95, edgecolor="none", label=m)
    ax_dist.set_ylabel("Mean Distance to True Minimum")
    ax_dist.grid(axis="y", linestyle="--", alpha=0.3)
    ax_dist.set_title(title)

    # Runtime panel
    for i, m in enumerate(methods):
        vals = pivot_time[m].values if m in pivot_time.columns else np.repeat(np.nan, num_levels)
        offs = x - total_group_width / 2 + (i + 0.5) * bar_w
        ax_time.bar(offs, vals, width=bar_w, color=colors(i), alpha=0.95, edgecolor="none")
    ax_time.set_ylabel("Mean Runtime (s)")
    ax_time.grid(axis="y", linestyle="--", alpha=0.3)

    ax_time.set_xticks(x)
    ax_time.set_xticklabels([str(l) for l in levels])
    ax_time.set_xlabel(characteristic)

    handles, labels = ax_dist.get_legend_handles_labels()
    if len(labels) <= 15:
        ax_dist.legend(handles, labels, loc="upper left", ncol=min(4, len(labels)))
    else:
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), bbox_to_anchor=(0.5, -0.02))

    save_plot(fig, filename_base)

# ---------- Optional: Baseline-only comparator (kept for flexibility) ----------
def compare_methods_over_characteristic(
    df: pd.DataFrame,
    characteristic: str,
    baseline_overrides: dict | None = None,
    export_prefix: str | None = None
):
    """
    Fix all non-target characteristics to their simplest baseline values,
    vary only `characteristic`, and export:
      - summary CSV, counts CSV, table PNG
    """
    if export_prefix is None:
        export_prefix = f"by_{characteristic}_"

    # 1) Resolve/apply baseline
    baseline_filter = _resolve_baseline(df, characteristic, baseline_overrides)
    df_simple = _filter_to_baseline(df, characteristic, baseline_filter)

    # 2) Build summary across levels/methods
    summary = _compute_summary_by_level_and_method(df_simple, characteristic)

    # 3) Fairness check
    counts, fairness_suffix = _fairness_check_counts(df_simple, characteristic)

    # 4) Save CSVs
    summary_csv = os.path.join(OUT_DIR, f"{export_prefix}method_summary.csv")
    counts_csv = os.path.join(OUT_DIR, f"{export_prefix}counts.csv")
    summary.to_csv(summary_csv, index=False)
    counts.to_csv(counts_csv, index=False)
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {counts_csv}")

    # 5) Table
    table_cols = [c for c in ["method", "level", "mean_dist", "mean_time", "runs"] if c in summary.columns]
    _export_table_png(
        df_table=summary[table_cols],
        filename=f"{export_prefix}summary_table",
        title=f"Summary: Methods vs {characteristic}" + fairness_suffix
    )

    # Baseline diagnostics
    print("\n=== Baseline filter applied (non-target characteristics) ===")
    for k, v in baseline_filter.items():
        if k != characteristic:
            print(f"• {k} = {v}")

    return {
        "summary": summary,
        "counts": counts,
        "fairness_suffix": fairness_suffix,
        "baseline_used": baseline_filter
    }

# ---------- NEW: Full vs Baseline (SEPARATE figures per characteristic) ----------
def compare_methods_full_and_baseline_separate(
    df: pd.DataFrame,
    characteristic: str,
    baseline_overrides: dict | None = None,
    export_prefix: str | None = None
):
    """
    For a given characteristic:
      - FULL: average over the entire experiment at each level (other factors vary).
      - BASELINE: average at each level with all other factors fixed to a simple base combo.
    Exports:
      - FULL summary + counts + table + plot
      - BASELINE summary + counts + table + plot
      - MERGED CSV with `mode` column (FULL/BASELINE)
    """
    if export_prefix is None:
        export_prefix = f"{characteristic}_"

    # --- FULL (no filtering except requiring the characteristic) ---
    df_full = df.dropna(subset=[characteristic]).copy()
    summary_full = _compute_summary_by_level_and_method(df_full, characteristic)
    counts_full, fairness_full = _fairness_check_counts(df_full, characteristic)

    # --- BASELINE (fix all non-targets to simplest combo) ---
    baseline_filter = _resolve_baseline(df, characteristic, baseline_overrides)
    df_base = _filter_to_baseline(df, characteristic, baseline_filter)
    summary_base = _compute_summary_by_level_and_method(df_base, characteristic)
    counts_base, fairness_base = _fairness_check_counts(df_base, characteristic)

    # --- Save CSVs ---
    summary_full_csv = os.path.join(OUT_DIR, f"{export_prefix}FULL_method_summary.csv")
    counts_full_csv  = os.path.join(OUT_DIR, f"{export_prefix}FULL_counts.csv")
    summary_base_csv = os.path.join(OUT_DIR, f"{export_prefix}BASE_method_summary.csv")
    counts_base_csv  = os.path.join(OUT_DIR, f"{export_prefix}BASE_counts.csv")

    summary_full.to_csv(summary_full_csv, index=False)
    counts_full.to_csv(counts_full_csv, index=False)
    summary_base.to_csv(summary_base_csv, index=False)
    counts_base.to_csv(counts_base_csv, index=False)

    print(f"[Saved] {summary_full_csv}")
    print(f"[Saved] {counts_full_csv}")
    print(f"[Saved] {summary_base_csv}")
    print(f"[Saved] {counts_base_csv}")

    # --- Also save a single merged CSV with a 'mode' column (FULL vs BASELINE) ---
    merged = pd.concat(
        [summary_full.assign(mode="FULL"), summary_base.assign(mode="BASELINE")],
        ignore_index=True
    )
    merged_csv = os.path.join(OUT_DIR, f"{export_prefix}MERGED_full_vs_baseline.csv")
    merged.to_csv(merged_csv, index=False)
    print(f"[Saved] {merged_csv}")

    # --- Tables ---
    table_cols = [c for c in ["method", "level", "mean_dist", "mean_time", "runs"] if c in summary_full.columns]
    _export_table_png(
        df_table=summary_full[table_cols],
        filename=f"{export_prefix}FULL_summary_table",
        title=f"FULL Summary: Methods vs {characteristic}" + fairness_full
    )
    _export_table_png(
        df_table=summary_base[table_cols],
        filename=f"{export_prefix}BASE_summary_table",
        title=f"BASELINE Summary: Methods vs {characteristic}" + fairness_base
    )

    # --- Plots (SEPARATE) ---
    _plot_grouped_bars_two_panels(
        summary_df=summary_full,
        characteristic=characteristic,
        title=f"FULL — Methods vs {characteristic}" + fairness_full,
        filename_base=f"{export_prefix}FULL_groupedbars"
    )
    _plot_grouped_bars_two_panels(
        summary_df=summary_base,
        characteristic=characteristic,
        title=f"BASELINE — Methods vs {characteristic}" + fairness_base,
        filename_base=f"{export_prefix}BASE_groupedbars"
    )

    # --- Diagnostics ---
    print("\n=== Baseline combo used (non-target characteristics fixed) ===")
    for k, v in baseline_filter.items():
        if k != characteristic:
            print(f"• {k} = {v}")

    return {
        "summary_full": summary_full,
        "counts_full": counts_full,
        "summary_base": summary_base,
        "counts_base": counts_base,
        "merged": merged,
        "baseline_used": baseline_filter,
        "fairness_full": fairness_full,
        "fairness_base": fairness_base
    }

# ============================================================
# Load data
# ============================================================
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# --- Detect and report rows with NaN in x_pred or dist ---
nan_mask = df["x_pred"].isna() | df["dist"].isna()
nan_rows = df[nan_mask].copy()

if len(nan_rows) > 0:
    nan_path = os.path.join(OUT_DIR, "rows_with_nan_in_predictions.csv")
    nan_rows.to_csv(nan_path, index=False)
    print(f"[WARN] {len(nan_rows)} rows dropped due to NaN in x_pred or dist.")
    print(f"[WARN] Saved offending rows to: {nan_path}")
else:
    print("[OK] No NaNs found in x_pred or dist.")

# Filter NaNs out
df = df[~nan_mask].copy()

# Clean method names
df["method"] = df["method"].astype(str)

# ============================================================
# 1) OVERALL ALGORITHM COMPARISON — Side-by-side bars with two y-axes
#     + export CSV and table PNG (original section)
# ============================================================

# ---------- (1) Compute raw summary ----------
overall = (
    df.groupby("method", dropna=False)
      .agg(mean_dist=("dist", "mean"),
           mean_time=("algo_time_s", "mean"),
           runs=("method", "count"))
      .reset_index()
      .sort_values(["mean_dist", "mean_time"], ascending=[True, True])
      .reset_index(drop=True)
)

# ---------- How many rows were used for each method's averages (console only) ----------
print("\n=== Rows used per method for averages (no downsampling) ===")
for _, row in overall.sort_values("method").iterrows():
    print(f"• {row['method']}: {int(row['runs'])} rows")

min_rows = int(overall["runs"].min()) if len(overall) else 0
max_rows = int(overall["runs"].max()) if len(overall) else 0
print(f"\n[Fairness] Same-N across methods? {'YES' if min_rows == max_rows else 'NO'} "
      f"(min={min_rows}, max={max_rows})")

TOP_N = 20
if len(overall) > TOP_N:
    print(f"\nTop {TOP_N} methods by rows used:")
    topn = overall.sort_values("runs", ascending=False).head(TOP_N)
    for _, row in topn.iterrows():
        print(f"  - {row['method']}: {int(row['runs'])} rows")

overall_csv = os.path.join(OUT_DIR, "overall_algorithm_summary.csv")
overall.to_csv(overall_csv, index=False)
print("[Saved] overall_algorithm_summary.csv")

# ---------- (2) Fairness check ----------
method_counts = (
    df.groupby("method", dropna=False)
      .size()
      .rename("rows")
      .reset_index()
      .sort_values("rows", ascending=False)
)
method_counts.to_csv(os.path.join(OUT_DIR, "overall_method_row_counts.csv"), index=False)
print("[Saved] overall_method_row_counts.csv")

unique_counts = method_counts["rows"].unique()
min_rows = int(method_counts["rows"].min()) if len(method_counts) else 0
max_rows = int(method_counts["rows"].max()) if len(method_counts) else 0

if len(unique_counts) == 1:
    fairness_title_suffix = " — Fair (Same N per Method)"
    print(f"[OK] Fairness check: all methods use the same number of rows ({unique_counts[0]}).")
else:
    fairness_title_suffix = f" — NOT Fair (N per method varies: min={min_rows}, max={max_rows})"
    print(f"[WARN] Fairness check: unequal #rows per method "
          f"(min={min_rows}, max={max_rows}).")
    print(f"[INFO] See 'overall_method_row_counts.csv' for details. Plots reflect raw data only.")

# ---------- (3) Plot side-by-side bars (raw data) ----------
_plot_side_by_side(
    overall,
    png_name="overall_distance_runtime_side_by_side",
    title="Overall Algorithm Comparison — Mean Distance & Mean Runtime" + fairness_title_suffix
)

# ---------- (4) Export table as PNG ----------
_export_table_png(
    overall,
    filename="overall_algorithm_summary_table",
    title="Overall Algorithm Summary" + fairness_title_suffix
)

# ============================================================
# 2) METHODS vs EACH CHARACTERISTIC — FULL and BASELINE (separate figures)
#    For each characteristic:
#       - FULL: avg over all other parameters
#       - BASELINE: fix all other parameters to simple base combo
# ============================================================

print("\n=== Methods vs each characteristic — FULL and BASELINE (separate figures) ===")
for ch in EXPERIMENT_GRID.keys():
    print(f"\n--- {ch} ---")
    try:
        compare_methods_full_and_baseline_separate(df, ch)
    except Exception as e:
        print(f"[ERROR] {ch} comparison failed: {e}")

print("=== Analysis Complete ===")
print("All PNG plots and CSV summaries stored in:", OUT_DIR)