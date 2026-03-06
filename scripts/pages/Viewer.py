# pages/Experiment Results Viewer.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Experiment Results Viewer", layout="wide")
st.title("Experiment Results Viewer")
st.caption("Explore results across Data Characteristics, polynomial degrees, and weakspots.")

# HARD-CODED path (match main app)
MASTER_CSV = r"C:\Users\lschrove\Desktop\PhD\PoC\POC_Weakspotidentification\Data\weakspot_experiments_master.csv"

path = st.text_input("Path to CSV (leave default to use master):", MASTER_CSV)

if not os.path.exists(path):
    st.warning("No experiment file found yet. Please run the experiment from the main page first.")
    st.stop()

df = pd.read_csv(path)
st.success(f"Loaded {len(df):,} rows from: {path}")

with st.sidebar:
    st.header("Filters")
    if "method" in df.columns:
        algos = sorted(df['method'].dropna().unique())
        sel_algos = st.multiselect("Algorithms", options=algos, default=algos)
        df = df[df['method'].isin(sel_algos)]

    if "degree_requested" in df.columns:
        degrees = sorted(df['degree_requested'].dropna().unique())
        sel_degrees = st.multiselect("Degree (requested)", options=degrees, default=degrees)
        df = df[df['degree_requested'].isin(sel_degrees)]

    # Data Characteristics filters
    for col in ["gaussian_std","heavy_tail_scale","heavy_tail_df","outlier_prob","outlier_magnitude",
                "nonsmooth_prob","nonsmooth_magnitude","nonuniform_skew","n_per_axis"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique())
            st.markdown(f"**{col}**")
            sel = st.multiselect(f"{col}", options=vals, default=vals, key=f"sel_{col}")
            df = df[df[col].isin(sel)]

    # Weakspot UI filters (unit space)
    if "u0" in df.columns and "v0" in df.columns:
        uvals = sorted(df["u0"].dropna().unique())
        vvals = sorted(df["v0"].dropna().unique())
        sel_u = st.multiselect("u0 positions", options=uvals, default=uvals)
        sel_v = st.multiselect("v0 positions", options=vvals, default=vvals)
        df = df[df["u0"].isin(sel_u) & df["v0"].isin(sel_v)]

st.subheader("Filtered data (head)")
st.dataframe(df.sort_values(["config_id","repeat_id","degree_requested","method"]).head(100))

st.markdown("---")
st.subheader("Mean Euclidean distance vs degree (by algorithm)")
if len(df):
    agg = df.groupby(["method", "degree_requested"], as_index=False)["dist"].mean().rename(columns={"dist":"mean_dist"})
    fig = px.line(agg, x="degree_requested", y="mean_dist", color="method", markers=True)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Value gap vs degree (by algorithm)")
if len(df):
    agg_gap = df.groupby(["method", "degree_requested"], as_index=False)["value_gap"].mean()
    fig2 = px.line(agg_gap, x="degree_requested", y="value_gap", color="method", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Per-condition summary (pivot)")
if len(df):
    summary = df.groupby(
        ["method","degree_requested","gaussian_std","heavy_tail_scale","outlier_prob","nonsmooth_prob","nonuniform_skew","n_per_axis"]
    ).agg(
        mean_dist=("dist","mean"),
        std_dist=("dist","std"),
        mean_value_gap=("value_gap","mean"),
        mean_elapsed_s=("elapsed_s","mean"),
        runs=("dist","count")
    ).reset_index()
    st.dataframe(summary.head(200))

st.markdown("---")
st.subheader("Distribution of distance by algorithm")
if len(df):
    fig3 = px.box(df, x="method", y="dist", points="suspectedoutliers", color="method", title="Distance distribution")
    st.plotly_chart(fig3, use_container_width=True)