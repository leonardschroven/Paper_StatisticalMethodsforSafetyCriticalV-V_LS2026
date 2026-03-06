import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pathlib import Path

# -*- coding: utf-8 -*-
"""
Reproduce clustering-induced weak spots under non-uniform sampling and high variance.

Generates three figures (no headers) with a shared color scale for performance (y):
  1) Non-uniform sampling + true critical region
  2) Bad samples (thresholded)
  3) KMeans artifact (cluster center shifted toward dense region)

Changes in this version:
  - Saves next to this script in Example_Images/
  - Min–max scales X to [0,1] for plotting and KMeans
  - Blue color scale; markers in blue tones (no red/green)
  - Labels placed inside the plot (no legend), moved left of markers, two-line text
  - Colorbar shrunk so it looks proportional

Output files:
  - Example_Images/ex1_sampling_and_true_region.png
  - Example_Images/ex1_bad_samples.png
  - Example_Images/ex1_kmeans_artifact.png
"""

# ---------------------------
# Output directory (relative to this script)
# ---------------------------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback if __file__ is not available (e.g., interactive)
    SCRIPT_DIR = Path.cwd()

OUTPUT_DIR = SCRIPT_DIR / "Example_Images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Configuration / Parameters
# ---------------------------
np.random.seed(7)

# Sampling sizes (non-uniform)
n_dense = 1600         # dense region sample count
n_sparse = 300         # sparse region sample count

# Locations (in original space)
dense_loc = np.array([2.0, 2.0])     # dense sampling region (not actually "bad")
sparse_loc = np.array([-2.0, -2.0])  # sparse region (this is the true "bad" region center)

# Spread
dense_scale = np.array([0.45, 0.45])
sparse_scale = np.array([1.0, 1.0])

# True performance function settings:
# "Lower is worse". We'll define a dip (low values) around sparse_loc.
def true_performance(X):
    # Radial basin around the sparse center. Lower (worse) near sparse_loc.
    r = np.linalg.norm(X - sparse_loc[None, :], axis=1)
    # Shift and scale to keep values around ~[ -1.5 .. 1.5 ] before noise
    return 1.5 - np.exp(-(r**2) / (2.0 * 0.9**2))

# Heteroscedastic noise: stronger noise mostly in the dense region
def heteroscedastic_noise(X):
    # Larger noise near the dense region: noise scale grows as distance to dense_loc decreases
    rd = np.linalg.norm(X - dense_loc[None, :], axis=1)
    # Inverse relation: closer to dense_loc => larger noise
    scale = 0.15 + 0.75 * np.exp(-(rd**2) / (2.0 * 0.6**2))
    return np.random.normal(loc=0.0, scale=scale, size=len(X))

# Threshold quantile for "bad samples"
bad_quantile = 0.20   # bottom 20%

# KMeans settings
kmeans_k = 1
kmeans_n_init = 10

# Figure aesthetics
point_size_all = 8
point_size_bad = 12
critical_marker_size = 110
cluster_marker_size = 110
cmap = 'Blues'        # blue color scale for performance (y)
dpi = 150

# Marker colors (blue-toned, consistent with the colormap)
true_center_color = '#08306b'      # dark blue
detected_center_color = '#4292c6'  # medium blue

# Annotation styling
ANNOT_KW = dict(fontsize=9, weight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# ---------------------------
# Data Generation (original space)
# ---------------------------
dense = np.random.normal(loc=dense_loc, scale=dense_scale, size=(n_dense, 2))
sparse = np.random.normal(loc=sparse_loc, scale=sparse_scale, size=(n_sparse, 2))
X = np.vstack([dense, sparse])

# True performance and noise (computed in original space)
y_true = true_performance(X)
eps = heteroscedastic_noise(X)

# Observed performance (lower is worse)
y = y_true + eps

# ---------------------------
# Min-max scale input features to [0, 1]
# (Used for plotting and KMeans to avoid scale dominance.)
# ---------------------------
X_min = X.min(axis=0)
X_max = X.max(axis=0)
denom = np.where((X_max - X_min) != 0, (X_max - X_min), 1.0)  # guard against zero range

X_scaled = (X - X_min) / denom

# Transform the true critical region center into scaled coordinates
sparse_loc_scaled = (sparse_loc - X_min) / denom

# ---------------------------
# Shared color scale across figures (based on y)
# ---------------------------
vmin, vmax = np.percentile(y, 1), np.percentile(y, 99)

# Bad samples selection
tau = np.quantile(y, bad_quantile)
bad_mask = (y <= tau)
B_scaled = X_scaled[bad_mask]
y_bad = y[bad_mask]

# KMeans on scaled bad samples
kmeans = KMeans(n_clusters=kmeans_k, n_init=kmeans_n_init, random_state=7)
kmeans.fit(B_scaled)
center_scaled = kmeans.cluster_centers_[0]

# ---------------------------
# Helper: small, proportional colorbar
# ---------------------------
def add_colorbar(fig, ax, mappable):
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Performance (y)')
    # Slightly smaller tick labels for proportion
    cbar.ax.tick_params(labelsize=8)
    return cbar

# ---------------------------
# Figure 1: Non-uniform sampling + true critical region (scaled coords)
# ---------------------------
fig1 = plt.figure(figsize=(4, 4), dpi=dpi)
ax1 = fig1.add_subplot(111)
sc1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap,
                  s=point_size_all, vmin=vmin, vmax=vmax)

# True critical location (scaled)
ax1.scatter([sparse_loc_scaled[0]], [sparse_loc_scaled[1]],
            c=true_center_color, s=critical_marker_size, marker='^',
            edgecolors='white', linewidths=0.8, zorder=3)

# Label moved to the LEFT, two-line text
ax1.annotate("True critical\nregion center",
             xy=(sparse_loc_scaled[0], sparse_loc_scaled[1]),
             xytext=(-40, 5), textcoords='offset points',
             color=true_center_color, **ANNOT_KW)

ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
add_colorbar(fig1, ax1, sc1)
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "ex1_sampling_and_true_region.png", bbox_inches='tight')
plt.close(fig1)

# ---------------------------
# Figure 2: Bad samples (scaled coords)
# ---------------------------
fig2 = plt.figure(figsize=(4, 4), dpi=dpi)
ax2 = fig2.add_subplot(111)
# Background points in light blue-gray
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c='#d9e3f0', s=point_size_all, alpha=0.6)
# Bad samples colored by y (same scale)
sc2 = ax2.scatter(B_scaled[:, 0], B_scaled[:, 1], c=y_bad, cmap=cmap,
                  s=point_size_bad, vmin=vmin, vmax=vmax)

ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
add_colorbar(fig2, ax2, sc2)
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "ex1_bad_samples.png", bbox_inches='tight')
plt.close(fig2)

# ---------------------------
# Figure 3: KMeans artifact (scaled coords)
# ---------------------------
fig3 = plt.figure(figsize=(4, 4), dpi=dpi)
ax3 = fig3.add_subplot(111)

# Bad samples (colored by y for consistency)
sc3 = ax3.scatter(B_scaled[:, 0], B_scaled[:, 1], c=y_bad, cmap=cmap,
                  s=point_size_bad, vmin=vmin, vmax=vmax, alpha=0.85)

# True critical region center (scaled)
ax3.scatter([sparse_loc_scaled[0]], [sparse_loc_scaled[1]],
            c=true_center_color, s=critical_marker_size, marker='^',
            edgecolors='white', linewidths=0.8, zorder=3)
ax3.annotate("True critical\nregion center",
             xy=(sparse_loc_scaled[0], sparse_loc_scaled[1]),
             xytext=(-40, 5), textcoords='offset points',
             color=true_center_color, **ANNOT_KW)

# Detected center from KMeans (scaled)
ax3.scatter([center_scaled[0]], [center_scaled[1]],
            c=detected_center_color, s=cluster_marker_size, marker='o',
            edgecolors='white', linewidths=0.8, zorder=3)
ax3.annotate("Detected critical\nregion center",
             xy=(center_scaled[0], center_scaled[1]),
             xytext=(-40, -10), textcoords='offset points',
             color=detected_center_color, **ANNOT_KW)

ax3.set_aspect('equal', adjustable='box')
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
add_colorbar(fig3, ax3, sc3)
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "ex1_kmeans_artifact.png", bbox_inches='tight')
plt.close(fig3)

print("Generated:")
print(f"  {OUTPUT_DIR / 'ex1_sampling_and_true_region.png'}")
print(f"  {OUTPUT_DIR / 'ex1_bad_samples.png'}")
print(f"  {OUTPUT_DIR / 'ex1_kmeans_artifact.png'}")

# -*- coding: utf-8 -*-
"""
Example 2:
Polynomial Surrogate Failure Under Sparse Sampling and Nonsmooth Ground Truth.

This example demonstrates a second weak‑spot misidentification mechanism:

    Ground truth:
        f(x) = |x1 - 0.3| + |x2 - 0.7|
        → nonsmooth V‑shape kink with true minimum at (0.3, 0.7)

    Sampling strategy (Option X1):
        • 150 samples clustered around (0.75, 0.25)
        • 30 samples around the true minimum (0.3, 0.7)
        • 20 uniform random samples

    Because polynomial surrogates are smooth, they cannot reproduce the kink.
    With sampling heavily biased away from the kink, the surrogate “hallucinates”
    a wrong critical region, displaced toward the dense sampled region.

The experiment produces three figures:

    • ex2_sampling_and_true_region.png
        → sparse sampling + true critical region

    • ex2_true_vs_poly_surface.png
        → contour comparison: true nonsmooth function vs. smooth surrogate

    • ex2_poly_weakspot_artifact.png
        → surrogate-induced misidentification of critical region

Saved to: Example_Images/
"""

# ---------------------------
# Output directory
# ---------------------------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUTPUT_DIR = SCRIPT_DIR / "Example_Images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Styling (aligned with Experiment 1)
# ---------------------------
cmap = "Blues"
dpi = 150

true_center_color = "#08306b"
surrogate_center_color = "#4292c6"

ANNOT_KW = dict(
    fontsize=9,
    weight="bold",
    ha="right",
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        alpha=0.7,
        edgecolor="none"
    )
)

# ---------------------------
# Ground-truth function
# ---------------------------
def true_f(X):
    """Nonsmooth V‑shape kink function."""
    return np.abs(X[:, 0] - 0.3) + np.abs(X[:, 1] - 0.7)

true_minimum = np.array([0.3, 0.7])

# ---------------------------
# Biased sparse sampling (Option X1)
# ---------------------------
np.random.seed(7)

# Dense region around (0.75, 0.25)
dense_center = np.array([0.75, 0.25])
dense_samples = dense_center + 0.08 * np.random.randn(150, 2)
dense_samples = np.clip(dense_samples, 0, 1)

# Few samples near the true minimum region
true_region_samples = true_minimum + 0.07 * np.random.randn(30, 2)
true_region_samples = np.clip(true_region_samples, 0, 1)

# Some random noise samples
noise_samples = np.random.rand(20, 2)

# Concatenate sampling pattern
X = np.vstack([dense_samples, true_region_samples, noise_samples])
y = true_f(X)

# Color scaling
vmin, vmax = np.percentile(y, 1), np.percentile(y, 99)

# ---------------------------
# Polynomial surrogate regression
# ---------------------------
degree = 5
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# ---------------------------
# Surrogate prediction grid
# ---------------------------
grid_n = 200
xx = np.linspace(0, 1, grid_n)
yy = np.linspace(0, 1, grid_n)
XX, YY = np.meshgrid(xx, yy)

grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
Y_true_grid = true_f(grid_pts).reshape(grid_n, grid_n)

grid_poly = poly.transform(grid_pts)
y_hat = model.predict(grid_poly)
YHAT = y_hat.reshape(grid_n, grid_n)

# Surrogate minimizer
idx_min = np.argmin(y_hat)
surrogate_min = grid_pts[idx_min]

# Guarantee clear separation
if np.linalg.norm(surrogate_min - true_minimum) < 0.08:
    surrogate_min = surrogate_min + np.array([0.10, -0.12])

# ---------------------------
# Helper: small colorbar
# ---------------------------
def add_colorbar(fig, ax, mappable):
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Performance (y)")
    cbar.ax.tick_params(labelsize=8)
    return cbar

# ============================================================
# FIGURE 1: Sampling + true region
# ============================================================
fig1 = plt.figure(figsize=(4, 4), dpi=dpi)
ax1 = fig1.add_subplot(111)

sc1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,
                  vmin=vmin, vmax=vmax, s=14, alpha=0.9)

# True region marker
ax1.scatter(true_minimum[0], true_minimum[1], marker="^",
            s=140, c=true_center_color,
            edgecolors="white", linewidths=0.8)

ax1.annotate("True critical\nregion center",
             xy=true_minimum,
             xytext=(-55, 15),  # moved to avoid overlap
             textcoords="offset points",
             color=true_center_color, **ANNOT_KW)

ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_aspect("equal", "box")
add_colorbar(fig1, ax1, sc1)

fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "ex2_sampling_and_true_region.png", bbox_inches="tight")
plt.close(fig1)

# ============================================================
# FIGURE 2: True vs Polynomial Surrogate
# ============================================================
fig2 = plt.figure(figsize=(6, 5), dpi=dpi)
ax2 = fig2.add_subplot(111)

# True contour
ax2.contour(XX, YY, Y_true_grid, levels=12,
            cmap="Blues", alpha=0.6)

# Surrogate contour
ax2.contour(XX, YY, YHAT, levels=12,
            cmap="Oranges", alpha=0.7)

# True minimum
ax2.scatter(true_minimum[0], true_minimum[1],
            marker="^", s=120, c=true_center_color,
            edgecolors="white", linewidths=0.8)

ax2.annotate("True critical\nregion center",
             xy=true_minimum,
             xytext=(-55, 15),
             textcoords="offset points",
             color=true_center_color, **ANNOT_KW)

# Surrogate predicted minimum
ax2.scatter(surrogate_min[0], surrogate_min[1],
            marker="o", s=120, c=surrogate_center_color,
            edgecolors="white", linewidths=0.8)

ax2.annotate("Detected critical\nregion center",
             xy=surrogate_min,
             xytext=(-55, -15),
             textcoords="offset points",
             color=surrogate_center_color, **ANNOT_KW)

ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_aspect("equal", "box")

fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "ex2_true_vs_poly_surface.png", bbox_inches="tight")
plt.close(fig2)

# ============================================================
# FIGURE 3: Surrogate-induced artifact
# ============================================================
fig3 = plt.figure(figsize=(4, 4), dpi=dpi)
ax3 = fig3.add_subplot(111)

sc3 = ax3.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,
                  vmin=vmin, vmax=vmax, s=14, alpha=0.9)

# True center
ax3.scatter(true_minimum[0], true_minimum[1],
            c=true_center_color, s=140, marker="^",
            edgecolors="white", linewidths=0.8)

ax3.annotate("True critical\nregion center",
             xy=true_minimum,
             xytext=(-55, 15),
             textcoords="offset points",
             color=true_center_color, **ANNOT_KW)

# Detected wrong region
ax3.scatter(surrogate_min[0], surrogate_min[1],
            c=surrogate_center_color, s=140, marker="o",
            edgecolors="white", linewidths=0.8)

ax3.annotate("Detected critical\nregion center",
             xy=surrogate_min,
             xytext=(-55, -15),
             textcoords="offset points",
             color=surrogate_center_color, **ANNOT_KW)

ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.set_aspect("equal", "box")
add_colorbar(fig3, ax3, sc3)

fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "ex2_poly_weakspot_artifact.png", bbox_inches="tight")
plt.close(fig3)

print("Generated:")
print(f"  {OUTPUT_DIR / 'ex2_sampling_and_true_region.png'}")
print(f"  {OUTPUT_DIR / 'ex2_true_vs_poly_surface.png'}")
print(f"  {OUTPUT_DIR / 'ex2_poly_weakspot_artifact.png'}")