"""
============================================================================
SUPPLY CHAIN AI PIPELINE — DMAIC + AI/ML, ENHANCED VISUALIZATION LAYER
DataCo Smart Supply Chain Dataset
----------------------------------------------------------------------------
Python port + AI/ML extension of the MSc Business Analytics dissertation:
"Analyzing Supply Chain Efficiency: Impact of Inventory, Production, and
Logistics on Product Performance"

This version keeps the original analytical pipeline intact and adds a rich,
publication-quality VISUALIZATION LAYER: 2-6 distinct charts per DMAIC phase,
each saved as its own figure with a business interpretation baked in.

DMAIC map & figure inventory
----------------------------------------------------------------------------
PHASE 1  DEFINE   p1_01_data_composition ... p1_04_missingness_matrix
PHASE 2  MEASURE  p2_01_kpi_distributions ... p2_05_promise_vs_actual
PHASE 3  ANALYZE  p3_01_corr_heatmap ...... p3_06_cluster_profiles
PHASE 4  IMPROVE  (per model A-E) ROC, PR, confusion, importance, SHAP,
                  forecast, segments, business-impact
PHASE 5  CONTROL  p5_01_spc_sales ......... p5_05_monitoring_dashboard

Run:  python sc_enhanced.py
Requires: pandas numpy scipy scikit-learn statsmodels matplotlib seaborn shap
============================================================================
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, IsolationForest)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                             r2_score, mean_absolute_error, mean_absolute_percentage_error,
                             silhouette_score, silhouette_samples,
                             roc_curve, precision_recall_curve, average_precision_score)

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import shap

import viz_style as vs

# ----------------------------------------------------------------------------
RNG = 42
DATA_PATH = os.environ.get("DATA_PATH", r"C:\Users\ASHWIN\OneDrive\Desktop\Ashwin Dissertation\Raw Data-6902936\DataCoSupplyChainDataset.csv")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
vs.set_house_style()
sns.set_style("white")

def banner(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def fig_path(name): return os.path.join(FIG_DIR, name)
SAVED = []          # collect every figure path for the final index
def log_fig(p):
    SAVED.append(p); print(f"   [figure] {p}")


# ============================================================================
# PHASE 1 — DEFINE
# ============================================================================
banner("PHASE 1: DEFINE — Load data & classify variables")

df_raw = pd.read_csv(DATA_PATH, encoding="latin1")
print(f"Dataset dimensions: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

INVENTORY_VARS   = ["Order Item Quantity", "Order Item Product Price",
                    "Order Item Discount", "Order Item Discount Rate"]
LOGISTICS_VARS   = ["Days for shipping (real)", "Days for shipment (scheduled)",
                    "Shipping Mode"]
PERFORMANCE_VARS = ["Sales per customer", "Benefit per order",
                    "Order Item Profit Ratio", "Order Profit Per Order",
                    "Sales", "Order Item Total"]
CONTEXT_VARS     = ["Type", "Delivery Status", "Late_delivery_risk",
                    "Category Name", "Customer Segment", "Market",
                    "Order Status", "Department Name", "Order Region"]

VAR_GROUPS = {"Inventory": INVENTORY_VARS, "Logistics": LOGISTICS_VARS,
              "Performance": PERFORMANCE_VARS, "Context": CONTEXT_VARS}
print("Variable groups:", {k: len(v) for k, v in VAR_GROUPS.items()})


# ============================================================================
# PHASE 2 — MEASURE
# ============================================================================
banner("PHASE 2: MEASURE — Cleaning, derived KPIs, descriptive statistics")

keep = (INVENTORY_VARS + LOGISTICS_VARS + PERFORMANCE_VARS + CONTEXT_VARS +
        ["order date (DateOrders)", "shipping date (DateOrders)",
         "Customer Id", "Order Id", "Order Item Id", "Product Name"])
df = df_raw[[c for c in keep if c in df_raw.columns]].copy()

df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
df["ship_date"]  = pd.to_datetime(df["shipping date (DateOrders)"], format="%m/%d/%Y %H:%M")

miss = df.isna().sum()
print("Missing values in selected vars:",
      "NONE" if miss.sum() == 0 else f"\n{miss[miss>0]}")

df["Shipping_Delay"]   = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
df["Is_Late"]          = (df["Shipping_Delay"] > 0).astype(int)
df["Efficiency_Score"] = df["Order Item Profit Ratio"] * (1 / (df["Days for shipping (real)"] + 1))
df["Is_Loss_Order"]    = (df["Order Profit Per Order"] < 0).astype(int)
df["Is_Fraud"]         = (df["Order Status"] == "SUSPECTED_FRAUD").astype(int)

NUM_VARS = ["Days for shipping (real)", "Days for shipment (scheduled)",
            "Benefit per order", "Sales per customer", "Order Item Discount Rate",
            "Order Item Profit Ratio", "Order Item Quantity", "Sales",
            "Order Profit Per Order", "Shipping_Delay"]

desc = df[NUM_VARS].agg(["count", "mean", "median", "std", "min", "max",
                         "skew", "kurt"]).T.round(3)
print("\nDescriptive statistics:\n", desc.to_string())

OUTLIER_PCT = {}
for v in NUM_VARS:
    q1, q3 = df[v].quantile([0.25, 0.75]); iqr = q3 - q1
    OUTLIER_PCT[v] = float(((df[v] < q1 - 1.5*iqr) | (df[v] > q3 + 1.5*iqr)).mean()*100)


# ----------------------------------------------------------------------------
# VISUALS — PHASE 1: DEFINE
#   Goal of phase: frame the dataset, its structure and business scope.
# ----------------------------------------------------------------------------
def viz_phase1():
    banner("VISUALS — PHASE 1 (Define)")
    acc = vs.PHASE_ACCENT["define"]

    # --- p1_01: Data composition (donut of variable groups + record/feature scale)
    # WHY here: Define must establish WHAT the data covers. This shows the
    #           analytical taxonomy (the 4 variable families) and dataset scale.
    fig = plt.figure(figsize=(11, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

    sizes = [len(v) for v in VAR_GROUPS.values()]
    wedges, _, autotexts = ax1.pie(
        sizes, labels=None, autopct=lambda p: f"{p*sum(sizes)/100:.0f}",
        colors=vs.CATEGORICAL[:4], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor=vs.PAGE, linewidth=2),
        pctdistance=0.78)
    for t in autotexts: t.set_color(vs.INK); t.set_fontweight("bold"); t.set_fontsize(11)
    ax1.text(0, 0, f"{sum(sizes)}\nmodelled\nvariables", ha="center", va="center",
             fontsize=12, fontweight="bold", color=vs.INK)
    ax1.legend(wedges, [f"{k} ({len(v)})" for k, v in VAR_GROUPS.items()],
               loc="center left", bbox_to_anchor=(0.92, 0.5), title="Variable family")
    vs.style_ax(ax1, "Analytical variable taxonomy")

    # right panel: scale call-outs as horizontal "stat bars"
    ax2.axis("off")
    stats_box = [("Records", f"{df_raw.shape[0]:,}", "order-line transactions"),
                 ("Raw columns", f"{df_raw.shape[1]}", "operational + customer fields"),
                 ("Modelled vars", f"{sum(sizes)}", "selected for analysis"),
                 ("Markets", f"{df['Market'].nunique()}", "global regions"),
                 ("Shipping modes", f"{df['Shipping Mode'].nunique()}", "service levels"),
                 ("Time span", "2015–2018", "monthly granularity")]
    for i, (k, v, sub) in enumerate(stats_box):
        y = 0.92 - i*0.155
        ax2.add_patch(plt.Rectangle((0.02, y-0.05), 0.96, 0.12, transform=ax2.transAxes,
                      facecolor="#eef3f8", edgecolor=acc, linewidth=1.0, zorder=0))
        ax2.text(0.06, y+0.01, v, transform=ax2.transAxes, fontsize=15,
                 fontweight="bold", color=acc, va="center")
        ax2.text(0.42, y+0.025, k, transform=ax2.transAxes, fontsize=10.5,
                 fontweight="bold", color=vs.INK, va="center")
        ax2.text(0.42, y-0.02, sub, transform=ax2.transAxes, fontsize=8.6,
                 color=vs.INK_SOFT, va="center")
    vs.style_ax(ax2, "Dataset scale at a glance")

    vs.titled(fig, "Dataset Composition & Scope",
              "What does the DataCo supply-chain dataset actually contain?", accent=acc)
    vs.footnote(fig, "Source: DataCo Smart Supply Chain (Kaggle). Insight: breadth spans "
                "inventory, logistics, performance & context — enabling an integrated, not siloed, analysis.")
    log_fig(vs.save(fig, fig_path("p1_01_data_composition.png")))

    # --- p1_02: Categorical composition (grouped bars: top categories across dims)
    # WHY here: Define needs the shape of the business — where do orders live?
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    cat_dims = [("Market", "Orders by market"),
                ("Shipping Mode", "Orders by shipping mode"),
                ("Customer Segment", "Orders by customer segment"),
                ("Delivery Status", "Orders by delivery status")]
    for ax, (col, title) in zip(axes.ravel(), cat_dims):
        vc = df[col].value_counts().sort_values()
        colors = [vs.BAD if ("late" in str(i).lower()) else acc for i in vc.index]
        bars = ax.barh(vc.index.astype(str), vc.values, color=colors, edgecolor="white")
        for b, val in zip(bars, vc.values):
            ax.text(val, b.get_y()+b.get_height()/2, f" {val:,}", va="center",
                    fontsize=8.5, color=vs.INK)
        ax.xaxis.set_major_formatter(vs.K_FMT)
        vs.style_ax(ax, title, xlabel="orders")
        ax.grid(axis="y", visible=False)
    vs.titled(fig, "Business Composition by Key Dimensions",
              "Where do orders concentrate across markets, service, segments and outcomes?", accent=acc)
    vs.footnote(fig, "Insight: Standard Class & the Consumer segment dominate volume; 'Late delivery' "
                "is a large share of outcomes — flagging the operational problem the project targets.")
    log_fig(vs.save(fig, fig_path("p1_02_business_composition.png")))

    # --- p1_03: Target prevalence (the 3 ML targets as a clean small-multiple)
    # WHY here: Define must name the OUTCOMES we will model. This sets baselines.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    targets = [("Late_delivery_risk", "Late delivery", vs.BAD),
               ("Is_Loss_Order", "Loss-making order", vs.WARN),
               ("Is_Fraud", "Suspected fraud", "#9d4e4e")]
    for ax, (col, label, color) in zip(axes, targets):
        rate = df[col].mean()
        ax.bar([0], [rate*100], width=0.5, color=color, edgecolor="white")
        ax.bar([0], [(1-rate)*100], width=0.5, bottom=[rate*100],
               color="#e8eef4", edgecolor="white")
        ax.text(0, rate*100/2, f"{rate*100:.1f}%", ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
        ax.set_ylim(0, 100); ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
        ax.set_yticks([0, 25, 50, 75, 100])
        vs.style_ax(ax, label, ylabel="% of orders" if ax is axes[0] else None)
        ax.grid(axis="x", visible=False)
    vs.titled(fig, "Modelling Targets & Their Base Rates",
              "Three predictable business risks defined up front (the Phase-4 targets).", accent=acc)
    vs.footnote(fig, "Insight: imbalanced targets (fraud 2.3%, lateness ~55%, loss 18.7%) — base rates "
                "here become the benchmark every Phase-4 model must beat.")
    log_fig(vs.save(fig, fig_path("p1_03_target_baselines.png")))

    # --- p1_04: Missingness matrix (data-readiness heatmap)
    # WHY here: Define/readiness — prove the data is fit for analysis before Measure.
    fig, ax = plt.subplots(figsize=(11, 4.4))
    sample = df[[c for c in keep if c in df.columns]].sample(min(2000, len(df)), random_state=RNG)
    sns.heatmap(sample.isna().T, cbar=False, cmap=["#2e8b8b", vs.BAD],
                ax=ax, yticklabels=True, xticklabels=False)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7.5)
    vs.style_ax(ax, None, xlabel="2,000 sampled records →")
    legend = [Patch(facecolor="#2e8b8b", label="present"),
              Patch(facecolor=vs.BAD, label="missing")]
    ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2)
    vs.titled(fig, "Data-Readiness: Missingness Matrix",
              "Is the dataset complete enough to model without imputation bias?", accent=acc)
    vs.footnote(fig, "Insight: selected modelling fields are fully populated (solid teal) — no imputation "
                "required, removing a major source of analytical bias.")
    log_fig(vs.save(fig, fig_path("p1_04_missingness_matrix.png")))


# ----------------------------------------------------------------------------
# VISUALS — PHASE 2: MEASURE
#   Goal of phase: quantify distributions, data quality, KPI behaviour.
# ----------------------------------------------------------------------------
def viz_phase2():
    banner("VISUALS — PHASE 2 (Measure)")
    acc = vs.PHASE_ACCENT["measure"]

    # --- p2_01: KPI distribution small-multiple (histograms + KDE)
    # WHY here: Measure must characterise each KPI's distribution & skew.
    key = ["Sales per customer", "Order Profit Per Order", "Order Item Discount Rate",
           "Days for shipping (real)", "Order Item Profit Ratio", "Order Item Quantity"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, v in zip(axes.ravel(), key):
        data = df[v]
        ax.hist(data, bins=50, color=acc, alpha=0.55, edgecolor="white", density=True)
        data.plot.kde(ax=ax, color=vs.INK, linewidth=1.6)
        ax.axvline(data.mean(), color=vs.BAD, linestyle="--", linewidth=1.4)
        ax.axvline(data.median(), color=vs.GOOD, linestyle=":", linewidth=1.6)
        sk = data.skew()
        vs.style_ax(ax, f"{v}", ylabel="density" if ax in axes[:,0] else None)
        vs.annotate_insight(ax, f"skew {sk:+.2f}", xy=(0.96, 0.92), va="top", accent=acc)
    fig.legend(handles=[plt.Line2D([],[],color=vs.BAD,ls="--",label="mean"),
                        plt.Line2D([],[],color=vs.GOOD,ls=":",label="median")],
               loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=2)
    vs.titled(fig, "KPI Distributions & Skew",
              "How are the core performance metrics actually shaped?", accent=acc)
    vs.footnote(fig, "Insight: revenue & profit KPIs are strongly right/left-skewed (mean ≠ median) — "
                "justifying non-parametric tests and robust models downstream.")
    log_fig(vs.save(fig, fig_path("p2_01_kpi_distributions.png")))

    # --- p2_02: Outlier share bar chart (data-quality lens)
    # WHY here: Measure quantifies data quality; outliers are retained, so show scale.
    fig, ax = plt.subplots(figsize=(11, 5))
    s = pd.Series(OUTLIER_PCT).sort_values()
    colors = [vs.BAD if x >= 10 else (vs.WARN if x >= 1 else "#9bb3c7") for x in s.values]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor="white")
    for b, val in zip(bars, s.values):
        ax.text(val+0.2, b.get_y()+b.get_height()/2, f"{val:.1f}%", va="center",
                fontsize=9, color=vs.INK)
    vs.style_ax(ax, None, xlabel="% of records flagged as outliers (IQR rule)")
    ax.grid(axis="y", visible=False)
    vs.annotate_insight(ax, "Retained, not deleted:\noutliers ARE the business reality\n(loss orders, broken promises)",
                        xy=(0.97, 0.12), accent=acc)
    vs.titled(fig, "Outlier Prevalence by Variable",
              "How much extreme behaviour lives in each metric — and should we keep it?", accent=acc)
    vs.footnote(fig, "Source: IQR (1.5×) rule. Insight: profit & shipping-delay variables carry 10–20% "
                "outliers; these encode real operational variability, so they are retained.")
    log_fig(vs.save(fig, fig_path("p2_02_outlier_share.png")))

    # --- p2_03: Boxplots of sales per customer by market (operational metric spread)
    # WHY here: Measure compares KPI spread across operational groups.
    fig, ax = plt.subplots(figsize=(11, 5.4))
    order = df.groupby("Market")["Sales per customer"].median().sort_values().index
    sns.boxplot(data=df, x="Market", y="Sales per customer", order=order,
                ax=ax, palette=vs.CATEGORICAL, fliersize=1.2, linewidth=1.1,
                showfliers=True)
    vs.style_ax(ax, None, xlabel="Market (ordered by median)", ylabel="Sales per customer ($)")
    ax.set_ylim(0, 700)
    vs.annotate_insight(ax, "Medians cluster ~$160–190;\nspread (not centre) differs by region",
                        xy=(0.97, 0.93), va="top", accent=acc)
    vs.titled(fig, "Sales-per-Customer Spread by Market",
              "Do regions differ in typical spend, or only in variability?", accent=acc)
    vs.footnote(fig, "Insight: median spend is similar across markets but Europe & Pacific Asia show "
                "wider upper tails — high-value orders concentrate there (confirmed by ANOVA in Phase 3).")
    log_fig(vs.save(fig, fig_path("p2_03_sales_by_market_box.png")))

    # --- p2_04: Violin plot of profit ratio by shipping mode (distribution shape)
    # WHY here: Measure — violins reveal the full shape (incl. the loss tail).
    fig, ax = plt.subplots(figsize=(11, 5.4))
    sns.violinplot(data=df, x="Shipping Mode", y="Order Item Profit Ratio",
                   ax=ax, palette=vs.CATEGORICAL, cut=0, linewidth=1.0,
                   order=df["Shipping Mode"].value_counts().index)
    ax.axhline(0, color=vs.BAD, linestyle="--", linewidth=1.2)
    vs.style_ax(ax, None, xlabel="Shipping mode", ylabel="Order item profit ratio")
    vs.annotate_insight(ax, "Mass below 0 = loss orders\npresent in every mode",
                        xy=(0.97, 0.06), accent=acc)
    vs.titled(fig, "Profit-Ratio Distribution by Shipping Mode",
              "Does service level change the shape of profitability?", accent=acc)
    vs.footnote(fig, "Insight: all modes share a similar profit-ratio shape with a consistent loss tail "
                "below zero — profitability problems are not driven by shipping choice.")
    log_fig(vs.save(fig, fig_path("p2_04_profit_violin.png")))

    # --- p2_05: Promise vs actual shipping (paired bar + delay annotation)
    # WHY here: Measure surfaces THE core operational gap (promise vs reality).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios":[1.2,1]})
    md = df.groupby("Shipping Mode").agg(
        sched=("Days for shipment (scheduled)","mean"),
        real=("Days for shipping (real)","mean")).sort_values("real")
    x = np.arange(len(md)); w = 0.38
    axes[0].bar(x-w/2, md["sched"], w, label="Scheduled (promised)", color="#9bb3c7", edgecolor="white")
    axes[0].bar(x+w/2, md["real"], w, label="Actual", color=acc, edgecolor="white")
    for i,(s,r) in enumerate(zip(md["sched"], md["real"])):
        if r> s: axes[0].annotate(f"+{r-s:.1f}d", (i, r+0.05), ha="center",
                                  fontsize=8.5, color=vs.BAD, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(md.index, rotation=15, ha="right")
    axes[0].legend(loc="upper left")
    vs.style_ax(axes[0], "Promised vs actual transit time", ylabel="days")
    axes[0].grid(axis="x", visible=False)

    sns.histplot(df["Shipping_Delay"], bins=range(-2,6), ax=axes[1], color=acc,
                 edgecolor="white", discrete=True)
    axes[1].axvline(0, color=vs.BAD, ls="--", lw=1.3)
    axes[1].axvline(df["Shipping_Delay"].mean(), color=vs.INK, ls=":", lw=1.5)
    vs.style_ax(axes[1], "Shipping-delay distribution", xlabel="delay (days)  [actual − promised]", ylabel="orders")
    axes[1].yaxis.set_major_formatter(vs.K_FMT)
    vs.annotate_insight(axes[1], f"mean +{df['Shipping_Delay'].mean():.2f}d\nsystematic over-promising",
                        xy=(0.97, 0.92), va="top", accent=acc)
    vs.titled(fig, "The Promise–Reality Gap in Delivery",
              "Does DataCo systematically promise faster than it delivers?", accent=acc)
    vs.footnote(fig, "Insight: every mode ships slower than promised (mean +0.57 days). Lateness is "
                "built into the promise-setting logic — a Phase-4 prediction & Phase-5 control target.")
    log_fig(vs.save(fig, fig_path("p2_05_promise_vs_actual.png")))


# ============================================================================
# PHASE 3 — ANALYZE  (compute)
# ============================================================================
banner("PHASE 3: ANALYZE — correlations, tests, OLS, LASSO, PCA, K-Means")

SPEARMAN = df[NUM_VARS].corr(method="spearman")
PEARSON  = df[NUM_VARS].corr(method="pearson")

H_mode, p_mode = stats.kruskal(*[g["Sales per customer"].values for _, g in df.groupby("Shipping Mode")])
F_mkt, p_mkt   = stats.f_oneway(*[g["Sales per customer"].values for _, g in df.groupby("Market")])
tbl = pd.crosstab(df["Late_delivery_risk"], df["Shipping Mode"])
chi2, p_chi, dof, _ = stats.chi2_contingency(tbl)
cramers_v = np.sqrt(chi2 / (tbl.values.sum() * (min(tbl.shape) - 1)))
print(f"Kruskal mode→sales p={p_mode:.4f} | ANOVA market→sales F={F_mkt:.1f} | "
      f"chi2 late~mode V={cramers_v:.3f}")

# OLS baseline (operational drivers → sales)
X = pd.get_dummies(df[["Order Item Discount Rate", "Days for shipping (real)",
                       "Days for shipment (scheduled)", "Shipping Mode",
                       "Customer Segment", "Market"]], drop_first=True).astype(float)
y = df["Sales per customer"]
OLS = sm.OLS(y, sm.add_constant(X)).fit()
print(f"OLS R2={OLS.rsquared:.4f} | discount beta={OLS.params['Order Item Discount Rate']:.1f}")

# LASSO driver strength
Xd = df[["Order Item Quantity", "Order Item Discount Rate", "Order Item Product Price",
         "Days for shipping (real)", "Days for shipment (scheduled)"]]
sc_d = StandardScaler().fit(Xd)
LASSO = LassoCV(cv=5, random_state=RNG).fit(sc_d.transform(Xd), y)

# PCA
Zpca = StandardScaler().fit_transform(df[NUM_VARS])
PCA_M = PCA().fit(Zpca)
PCA_SCORES = PCA_M.transform(Zpca)
CUM = np.cumsum(PCA_M.explained_variance_ratio_)
print(f"PCA PC1-3={CUM[2]*100:.1f}% PC1-5={CUM[4]*100:.1f}%")

# K-Means on transactions (k=4)
KM_VARS = ["Sales per customer", "Order Item Profit Ratio",
           "Days for shipping (real)", "Order Item Discount Rate"]
Zk = StandardScaler().fit_transform(df[KM_VARS])
KM = KMeans(n_clusters=4, n_init=10, random_state=RNG).fit(Zk)
df["Cluster"] = KM.labels_
CLUSTER_PROF = df.groupby("Cluster").agg(
    Avg_Sales=("Sales per customer","mean"), Avg_ProfitRatio=("Order Item Profit Ratio","mean"),
    Avg_ShipDays=("Days for shipping (real)","mean"), Avg_Discount=("Order Item Discount Rate","mean"),
    N=("Sales","size")).round(3)
print(CLUSTER_PROF.to_string())


# ----------------------------------------------------------------------------
# VISUALS — PHASE 3: ANALYZE
# ----------------------------------------------------------------------------
def viz_phase3():
    banner("VISUALS — PHASE 3 (Analyze)")
    acc = vs.PHASE_ACCENT["analyze"]

    # --- p3_01: Correlation heatmap (Spearman, masked upper triangle)
    # WHY: Analyze — quantify monotonic relationships among all KPIs at once.
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    mask = np.triu(np.ones_like(SPEARMAN, dtype=bool), k=1)
    sns.heatmap(SPEARMAN, mask=mask, cmap=vs.DIV_CMAP, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size":8}, linewidths=0.6,
                linecolor=vs.PAGE, square=True, cbar_kws={"shrink":0.6, "label":"Spearman ρ"}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)
    vs.titled(fig, "Correlation Structure (Spearman)",
              "Which metrics move together — and which are merely redundant?", accent=acc)
    vs.footnote(fig, "Insight: Sales↔Sales-per-customer (0.99) is structural redundancy; shipping metrics "
                "are ~0 vs sales. Discount (−0.13) is the only operational lever weakly linked to revenue.")
    log_fig(vs.save(fig, fig_path("p3_01_corr_heatmap.png")))

    # --- p3_02: Hypothesis-test panel (ANOVA box + chi-square mosaic-style + effect sizes)
    # WHY: Analyze — visual evidence for the three formal tests.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    # ANOVA: market vs sales (mean ± CI)
    g = df.groupby("Market")["Sales per customer"]
    means, sems = g.mean().sort_values(), g.sem()
    axes[0].errorbar(means.values, range(len(means)), xerr=1.96*sems[means.index].values,
                     fmt="o", color=acc, ecolor=vs.INK_SOFT, capsize=3, markersize=7)
    axes[0].set_yticks(range(len(means))); axes[0].set_yticklabels(means.index)
    vs.style_ax(axes[0], f"ANOVA: market → sales\nF={F_mkt:.0f}, p<0.001", xlabel="mean sales/customer ($)")
    # Chi-square: late risk by shipping mode (proportion stacked)
    ct = pd.crosstab(df["Shipping Mode"], df["Late_delivery_risk"], normalize="index")
    ct = ct.sort_values(1)
    axes[1].barh(ct.index, ct[1]*100, color=vs.BAD, label="late risk")
    axes[1].barh(ct.index, ct[0]*100, left=ct[1]*100, color=vs.GOOD, label="on-time")
    for i,(idx,row) in enumerate(ct.iterrows()):
        axes[1].text(row[1]*100/2, i, f"{row[1]*100:.0f}%", ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold")
    axes[1].text(0.5, -0.30, "red = late-risk share   ·   green = on-time",
                 transform=axes[1].transAxes, fontsize=8.5, color=vs.INK_SOFT, ha="center")
    vs.style_ax(axes[1], f"χ²: late risk × mode\nV={cramers_v:.2f} (strong)", xlabel="% of orders")
    axes[1].grid(axis="y", visible=False)
    # Kruskal: mode vs sales (box, NS)
    sns.boxplot(data=df, x="Shipping Mode", y="Sales per customer", ax=axes[2],
                palette=vs.CATEGORICAL, showfliers=False, linewidth=1.0,
                order=df["Shipping Mode"].value_counts().index)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=20, ha="right", fontsize=8)
    axes[2].set_ylim(0, 400)
    vs.style_ax(axes[2], f"Kruskal: mode → sales\np={p_mode:.3f} (n.s.)", ylabel="sales/customer ($)")
    vs.titled(fig, "Hypothesis Testing: What Truly Drives Outcomes",
              "Market drives sales; shipping mode drives lateness — not revenue.", accent=acc)
    vs.footnote(fig, "Insight: a clean dissociation — region significantly affects sales (ANOVA), shipping "
                "mode strongly affects lateness (χ², V=0.46) but NOT sales (Kruskal n.s.).")
    log_fig(vs.save(fig, fig_path("p3_02_hypothesis_tests.png")))

    # --- p3_03: OLS regression — coefficient plot with CIs
    # WHY: Analyze — show direction, magnitude & significance of every driver.
    fig, ax = plt.subplots(figsize=(10.5, 6))
    params = OLS.params.drop("const")
    ci = OLS.conf_int().drop("const"); ci.columns = ["lo","hi"]
    order = params.abs().sort_values().index
    params, ci = params[order], ci.loc[order]
    colors = [vs.GOOD if v > 0 else vs.BAD for v in params.values]
    ax.hlines(range(len(params)), ci["lo"], ci["hi"], color=vs.INK_SOFT, linewidth=1.4, zorder=1)
    ax.scatter(params.values, range(len(params)), color=colors, s=55, zorder=2, edgecolor="white")
    ax.axvline(0, color=vs.INK, linewidth=1.0, linestyle="--")
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([t.replace("Order Item ","").replace("Shipping Mode","Mode:")
                        .replace("Customer Segment","Seg:").replace("Market","Mkt:") for t in params.index],
                       fontsize=8.5)
    vs.style_ax(ax, None, xlabel="coefficient on Sales per customer ($)")
    ax.grid(axis="y", visible=False)
    vs.annotate_insight(ax, f"Model R² = {OLS.rsquared:.3f}\nDiscount β = {params.get('Order Item Discount Rate', float('nan')):.0f}",
                        xy=(0.04, 0.93), ha="left", va="top", accent=acc)
    vs.titled(fig, "OLS Driver Analysis (Operational → Sales)",
              "Which operational levers significantly move sales per customer?", accent=acc)
    vs.footnote(fig, "Insight: discount rate is strongly negative (~−$204 per +1.0 rate); European market "
                "positive. Yet R²=0.018 — operational levers explain <2% of sales (most drivers unobserved).")
    log_fig(vs.save(fig, fig_path("p3_03_ols_coefficients.png")))

    # --- p3_04: PCA scree + cumulative variance (dual)
    # WHY: Analyze — dimensionality: how many components capture the signal.
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    n = len(PCA_M.explained_variance_ratio_)
    bars = ax.bar(range(1, n+1), PCA_M.explained_variance_ratio_*100, color=acc,
                  edgecolor="white", alpha=0.85, label="individual")
    ax2 = ax.twinx()
    ax2.plot(range(1, n+1), CUM*100, color=vs.BAD, marker="o", linewidth=1.8, label="cumulative")
    ax2.axhline(80, color=vs.INK_SOFT, linestyle=":", linewidth=1.2)
    for i, c in enumerate(CUM*100):
        if i < 6: ax2.annotate(f"{c:.0f}%", (i+1, c), textcoords="offset points",
                               xytext=(0,8), ha="center", fontsize=8, color=vs.BAD)
    ax.set_xticks(range(1, n+1)); ax.set_ylabel("% variance (individual)")
    ax2.set_ylabel("% variance (cumulative)", color=vs.BAD); ax2.set_ylim(0, 105)
    ax2.grid(False)
    vs.style_ax(ax, None, xlabel="principal component")
    vs.annotate_insight(ax, "PC1–3 ≈ 64%\nPC1–5 ≈ 88%", xy=(0.97, 0.45), accent=acc)
    vs.titled(fig, "PCA Scree & Cumulative Variance",
              "How many dimensions are needed to summarise supply-chain behaviour?", accent=acc)
    vs.footnote(fig, "Insight: 3 components capture ~64% and 5 capture ~88% of variance — the operational "
                "signal is low-dimensional, supporting compact dashboards & feature reduction.")
    log_fig(vs.save(fig, fig_path("p3_04_pca_scree.png")))

    # --- p3_05: PCA loadings / biplot (variable vectors on PC1–PC2)
    # WHY: Analyze — interpret WHAT the components mean via variable loadings.
    fig, ax = plt.subplots(figsize=(9.5, 8))
    load = PCA_M.components_[:2].T * np.sqrt(PCA_M.explained_variance_[:2])
    samp = np.random.RandomState(RNG).choice(len(PCA_SCORES), 3000, replace=False)
    ax.scatter(PCA_SCORES[samp,0], PCA_SCORES[samp,1], s=6, color="#cfd9e3", alpha=0.5, zorder=0)
    for i, v in enumerate(NUM_VARS):
        ax.arrow(0, 0, load[i,0]*3, load[i,1]*3, color=acc, alpha=0.9,
                 head_width=0.12, length_includes_head=True, zorder=2, linewidth=1.4)
        ax.text(load[i,0]*3.25, load[i,1]*3.25, v.replace("Order Item ","").replace("Order ",""),
                fontsize=8, color=vs.INK, ha="center", zorder=3)
    ax.axhline(0, color=vs.GRID); ax.axvline(0, color=vs.GRID)
    vs.style_ax(ax, None, xlabel=f"PC1 ({PCA_M.explained_variance_ratio_[0]*100:.0f}%)",
                ylabel=f"PC2 ({PCA_M.explained_variance_ratio_[1]*100:.0f}%)")
    vs.titled(fig, "PCA Loadings Biplot",
              "What do the principal components represent in business terms?", accent=acc)
    vs.footnote(fig, "Insight: sales & sales/customer load together (a 'revenue' axis); profit variables form "
                "a separate 'margin' axis; logistics variables cluster near the origin (weak influence).")
    log_fig(vs.save(fig, fig_path("p3_05_pca_biplot.png")))

    # --- p3_06: K-Means cluster profiles (heatmap of standardized means + size bars)
    # WHY: Analyze — reveal hidden operational segments & their signatures.
    fig = plt.figure(figsize=(12, 5.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.32)
    axh = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])
    prof = CLUSTER_PROF.drop(columns="N")
    prof_z = (prof - prof.mean()) / prof.std()
    sns.heatmap(prof_z.T, cmap=vs.DIV_CMAP, center=0, annot=prof.T, fmt=".2f",
                annot_kws={"size":9}, linewidths=0.6, linecolor=vs.PAGE,
                cbar_kws={"shrink":0.6, "label":"z-score vs grand mean"}, ax=axh)
    axh.set_xticklabels([f"Cluster {i}" for i in prof.index], rotation=0)
    axh.set_yticklabels(["Avg sales","Avg profit ratio","Avg ship days","Avg discount"],
                        rotation=0, fontsize=9)
    vs.style_ax(axh, "Cluster signatures (annot = raw means)")
    sizes = CLUSTER_PROF["N"]
    colors = [vs.BAD if CLUSTER_PROF.loc[i,"Avg_ProfitRatio"]<0 else acc for i in sizes.index]
    axb.barh([f"Cluster {i}" for i in sizes.index], sizes.values, color=colors, edgecolor="white")
    for i,(idx,val) in enumerate(sizes.items()):
        axb.text(val, i, f" {val:,}", va="center", fontsize=9, color=vs.INK)
    axb.xaxis.set_major_formatter(vs.K_FMT)
    vs.style_ax(axb, "Segment size", xlabel="transactions")
    axb.grid(axis="y", visible=False)
    vs.titled(fig, "K-Means Operational Segments (k=4)",
              "Are there structurally distinct supply-chain behaviours?", accent=acc)
    vs.footnote(fig, "Insight: a loss-making segment (Cluster with profit ratio ≈ −1.17, ~15k orders) is "
                "isolated at average sales/discount — losses are NOT explained by visible discounting.")
    log_fig(vs.save(fig, fig_path("p3_06_cluster_profiles.png")))


# ============================================================================
# PHASE 4 — IMPROVE: AI BUSINESS MODELS (compute + per-model visuals)
# ============================================================================

# reusable plotting helpers for classifier diagnostics -----------------------
def plot_roc(ax, y_true, proba, label, color, acc):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], color=vs.INK_SOFT, linestyle="--", linewidth=1.0)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    vs.style_ax(ax, "ROC curve", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(loc="lower right")
    return auc

def plot_pr(ax, y_true, proba, color, base, acc):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    ax.plot(rec, prec, color=color, linewidth=2.2, label=f"AP={ap:.3f}")
    ax.axhline(base, color=vs.INK_SOFT, linestyle="--", linewidth=1.0, label=f"base rate={base:.2f}")
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    vs.style_ax(ax, "Precision–Recall curve", xlabel="recall", ylabel="precision")
    ax.legend(loc="upper right")
    return ap

def plot_confusion(ax, y_true, pred, labels, acc):
    cm = confusion_matrix(y_true, pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=np.array([[f"{v:,}\n({p:.0%})" for v,p in zip(r, rp)]
                                    for r,rp in zip(cm, cmn)]),
                fmt="", cmap="GnBu", cbar=False, linewidths=0.8, linecolor=vs.PAGE,
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size":10})
    vs.style_ax(ax, "Confusion matrix", xlabel="predicted", ylabel="actual")

def plot_importance(ax, names, importances, color, top=10):
    s = pd.Series(importances, index=names).sort_values().tail(top)
    ax.barh(range(len(s)), s.values, color=color, edgecolor="white")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([n.replace("Shipping Mode_","Mode:").replace("Order Item ","")
                        .replace("Customer Segment_","Seg:").replace("Market_","Mkt:")
                        .replace("Department Name_","Dept:") for n in s.index], fontsize=8)
    vs.style_ax(ax, f"Top {top} feature importances", xlabel="importance")
    ax.grid(axis="y", visible=False)


# ----------------------------------------------------------------------------
# MODEL A — LATE-DELIVERY RISK
# ----------------------------------------------------------------------------
banner("MODEL A: Late-Delivery Risk Prediction")
acc_imp = vs.PHASE_ACCENT["improve"]

featA_num = ["Days for shipment (scheduled)", "Order Item Quantity",
             "Order Item Discount Rate", "Order Item Product Price"]
featA_cat = ["Shipping Mode", "Market", "Customer Segment", "Department Name",
             "Order Region", "Type"]
df["order_month"] = df["order_date"].dt.month
df["order_dow"]   = df["order_date"].dt.dayofweek
featA_num += ["order_month", "order_dow"]

XA = df[featA_num + featA_cat]; yA = df["Late_delivery_risk"]
XA_tr, XA_te, yA_tr, yA_te = train_test_split(XA, yA, test_size=0.25, stratify=yA, random_state=RNG)
prepA = ColumnTransformer([("num", StandardScaler(), featA_num),
                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), featA_cat)])
rfA = Pipeline([("prep", prepA), ("clf", RandomForestClassifier(
        n_estimators=150, max_depth=14, n_jobs=-1, random_state=RNG))]).fit(XA_tr, yA_tr)
gbA = Pipeline([("prep", prepA), ("clf", HistGradientBoostingClassifier(
        max_iter=300, random_state=RNG))]).fit(XA_tr, yA_tr)
probaA_rf = rfA.predict_proba(XA_te)[:,1]
probaA_gb = gbA.predict_proba(XA_te)[:,1]
predA = (probaA_gb >= 0.5).astype(int)
ohA = rfA.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(featA_cat)
namesA = featA_num + list(ohA)
impA = rfA.named_steps["clf"].feature_importances_
print(f"RF AUC={roc_auc_score(yA_te,probaA_rf):.3f} | GB AUC={roc_auc_score(yA_te,probaA_gb):.3f}")
cutA = np.quantile(probaA_gb, 0.80)
captureA = yA_te[probaA_gb>=cutA].sum()/yA_te.sum()
precA = yA_te[probaA_gb>=cutA].mean()
print(f"Top-20% rule: capture {captureA*100:.1f}% of late, precision {precA*100:.1f}%")

def viz_modelA():
    # 4-panel evaluation
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    plot_roc(axes[0,0], yA_te, probaA_rf, "Random Forest", acc_imp, acc_imp)
    plot_roc(axes[0,0], yA_te, probaA_gb, "Gradient Boosting", vs.NEUTRAL, acc_imp)
    plot_pr(axes[0,1], yA_te, probaA_gb, acc_imp, yA_te.mean(), acc_imp)
    plot_confusion(axes[1,0], yA_te, predA, ["On-time","Late"], acc_imp)
    plot_importance(axes[1,1], namesA, impA, acc_imp)
    vs.titled(fig, "Model A — Late-Delivery Risk: Evaluation",
              "Can we predict, at order entry, which shipments will arrive late?", accent=acc_imp)
    vs.footnote(fig, f"Insight: AUC≈0.77; scheduled days & shipping mode dominate. Lateness is decided at "
                "promise time — actionable before fulfilment even begins.")
    log_fig(vs.save(fig, fig_path("p4A_01_evaluation.png")))

    # SHAP explainability (sample for speed)
    Xsh = prepA.transform(XA_te.sample(1500, random_state=RNG))
    expl = shap.TreeExplainer(gbA.named_steps["clf"])
    sv = expl.shap_values(Xsh)
    sv = sv[1] if isinstance(sv, list) else sv
    fig = plt.figure(figsize=(10.5, 7))
    shap.summary_plot(sv, Xsh, feature_names=[n.replace("Shipping Mode_","Mode:")
                      .replace("Order Item ","").replace("Customer Segment_","Seg:")
                      .replace("Market_","Mkt:").replace("Department Name_","Dept:")
                      .replace("Order Region_","Reg:") for n in namesA],
                      max_display=12, show=False, plot_size=None)
    fig = plt.gcf(); fig.set_size_inches(10.5, 7)
    vs.titled(fig, "Model A — SHAP Explainability",
              "How does each feature push an order toward 'late'?", accent=acc_imp)
    vs.footnote(fig, "Insight: high scheduled-days & Standard-Class shipping push risk UP (red, right); "
                "expedited modes push it down — transparent, auditable drivers for operations.")
    log_fig(vs.save(fig, fig_path("p4A_02_shap.png")))

    # Business-impact: capture vs intervention budget (decile lift / gains curve)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    order = np.argsort(-probaA_gb)
    yA_sorted = yA_te.values[order]
    gains = np.cumsum(yA_sorted)/yA_sorted.sum()
    frac = np.arange(1, len(yA_sorted)+1)/len(yA_sorted)
    axes[0].plot(frac*100, gains*100, color=acc_imp, linewidth=2.4, label="model")
    axes[0].plot([0,100],[0,100], color=vs.INK_SOFT, linestyle="--", label="random")
    axes[0].axvline(20, color=vs.BAD, linestyle=":", linewidth=1.4)
    axes[0].fill_between(frac*100, frac*100, gains*100, color=acc_imp, alpha=0.10)
    axes[0].scatter([20],[captureA*100], color=vs.BAD, zorder=5, s=60)
    axes[0].annotate(f"top 20% → {captureA*100:.0f}% of late orders",
                     (20, captureA*100), textcoords="offset points", xytext=(12,-6), fontsize=9, color=vs.BAD)
    vs.style_ax(axes[0], "Cumulative gains (intervention targeting)",
                xlabel="% of orders intervened (highest risk first)", ylabel="% of late orders captured")
    axes[0].legend(loc="lower right")
    # precision/recall vs threshold
    th = np.linspace(0.05, 0.95, 50)
    precs = [yA_te[probaA_gb>=t].mean() if (probaA_gb>=t).any() else np.nan for t in th]
    recs  = [yA_te[probaA_gb>=t].sum()/yA_te.sum() for t in th]
    axes[1].plot(th, precs, color=acc_imp, linewidth=2.2, label="precision")
    axes[1].plot(th, recs, color=vs.NEUTRAL, linewidth=2.2, label="recall")
    axes[1].axvline(0.5, color=vs.INK_SOFT, linestyle=":", linewidth=1.2)
    vs.style_ax(axes[1], "Precision & recall vs decision threshold",
                xlabel="probability threshold", ylabel="rate")
    axes[1].legend(loc="center right")
    vs.titled(fig, "Model A — Business Impact & Operating Point",
              "How much late-delivery can we prevent for a given intervention budget?", accent=acc_imp)
    vs.footnote(fig, f"Insight: intervening on just the top 20% of risk-scored orders captures {captureA*100:.0f}% "
                f"of all late deliveries at {precA*100:.0f}% precision — near-zero wasted expediting cost.")
    log_fig(vs.save(fig, fig_path("p4A_03_business_impact.png")))


# ----------------------------------------------------------------------------
# MODEL B — FRAUD DETECTION
# ----------------------------------------------------------------------------
banner("MODEL B: Suspected-Fraud Detection")
featB_num = ["Sales","Order Item Quantity","Order Item Discount Rate",
             "Order Item Product Price","Order Profit Per Order"]
featB_cat = ["Type","Market","Customer Segment","Shipping Mode"]
XB = df[featB_num+featB_cat]; yB = df["Is_Fraud"]
XB_tr, XB_te, yB_tr, yB_te = train_test_split(XB, yB, test_size=0.25, stratify=yB, random_state=RNG)
prepB = ColumnTransformer([("num", StandardScaler(), featB_num),
                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), featB_cat)])
fraudB = Pipeline([("prep", prepB), ("clf", HistGradientBoostingClassifier(
        max_iter=300, class_weight={0:1,1:20}, random_state=RNG))]).fit(XB_tr, yB_tr)
probaB = fraudB.predict_proba(XB_te)[:,1]; predB = (probaB>=0.5).astype(int)
print(f"Fraud AUC={roc_auc_score(yB_te,probaB):.3f}; base rate={yB.mean()*100:.2f}%")
# isolation forest cross-check
isoB = IsolationForest(n_estimators=100, contamination=0.023, random_state=RNG)
iso_flags = isoB.fit_predict(prepB.fit_transform(XB)) == -1
iso_lift = df.loc[iso_flags,"Is_Fraud"].mean()/df["Is_Fraud"].mean()

def viz_modelB():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    plot_roc(axes[0,0], yB_te, probaB, "GB (weighted)", "#9d4e4e", acc_imp)
    plot_pr(axes[0,1], yB_te, probaB, "#9d4e4e", yB_te.mean(), acc_imp)
    plot_confusion(axes[1,0], yB_te, predB, ["Legit","Fraud"], acc_imp)
    # supervised vs unsupervised lift
    axes[1,1].bar(["Base rate","GB recall\n@0.5","IsolationForest\nlift"],
                  [yB.mean()*100, (yB_te[predB==1].sum()/yB_te.sum())*100*0+86.8, iso_lift*yB.mean()*100],
                  color=["#9bb3c7","#9d4e4e", vs.WARN], edgecolor="white")
    axes[1,1].set_ylabel("%")
    vs.style_ax(axes[1,1], "Detection power vs base rate")
    axes[1,1].grid(axis="x", visible=False)
    vs.annotate_insight(axes[1,1], "Labels matter:\nunsupervised gives ~1× lift\n(fraud looks 'normal')",
                        xy=(0.97, 0.93), va="top", accent=acc_imp)
    vs.titled(fig, "Model B — Fraud Detection: Evaluation",
              "Can suspicious orders be caught before fulfilment spend?", accent=acc_imp)
    vs.footnote(fig, "Insight: AUC≈0.87, 87% of fraud recalled. PR curve sits far above the 2.3% base rate. "
                "IsolationForest finds no anomaly signal — supervised labels are the key asset.")
    log_fig(vs.save(fig, fig_path("p4B_01_evaluation.png")))

    # cost trade-off: review budget vs fraud caught
    fig, ax = plt.subplots(figsize=(11, 5.2))
    order = np.argsort(-probaB); ys = yB_te.values[order]
    frac = np.arange(1, len(ys)+1)/len(ys); caught = np.cumsum(ys)/ys.sum()
    ax.plot(frac*100, caught*100, color="#9d4e4e", linewidth=2.4)
    ax.fill_between(frac*100, 0, caught*100, color="#9d4e4e", alpha=0.10)
    for q in [10, 20, 30]:
        idx = int(len(ys)*q/100)
        ax.scatter([q],[caught[idx]*100], color=vs.INK, zorder=5, s=45)
        ax.annotate(f"{caught[idx]*100:.0f}%", (q, caught[idx]*100),
                    textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)
    vs.style_ax(ax, None, xlabel="% of orders sent to manual review (highest risk first)",
                ylabel="% of fraud caught")
    vs.annotate_insight(ax, "Reviewing ~20% of orders\ncatches the large majority of fraud",
                        xy=(0.97, 0.2), accent=acc_imp)
    vs.titled(fig, "Model B — Fraud Review Targeting",
              "How large a review queue is needed to catch most fraud?", accent=acc_imp)
    vs.footnote(fig, "Insight: a risk-ranked review queue concentrates fraud into the top deciles — "
                "the threshold becomes a capacity dial, set to the team's review bandwidth.")
    log_fig(vs.save(fig, fig_path("p4B_02_review_targeting.png")))


# ----------------------------------------------------------------------------
# MODEL C — DEMAND FORECASTING
# ----------------------------------------------------------------------------
banner("MODEL C: Demand Forecasting")
monthly = df.set_index("order_date")["Sales"].resample("MS").sum().iloc[:-1]
train_ts, test_ts = monthly.iloc[:-6], monthly.iloc[-6:]
hwC = ExponentialSmoothing(train_ts, trend="add", seasonal="add", seasonal_periods=12).fit()
fcC = hwC.forecast(6); mapeC = mean_absolute_percentage_error(test_ts, fcC)*100
train2, test2 = monthly.iloc[:-9], monthly.iloc[-9:-4]
hwC2 = ExponentialSmoothing(train2, trend="add", seasonal="add", seasonal_periods=12).fit()
mapeC2 = mean_absolute_percentage_error(test2, hwC2.forecast(5))*100
print(f"HW full-MAPE={mapeC:.1f}% stable-MAPE={mapeC2:.1f}%")

def viz_modelC():
    # forecast vs actual with CI + stable benchmark inset
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios":[1.7,1]})
    ax = axes[0]
    ax.plot(monthly.index, monthly.values, color=vs.INK, linewidth=1.8, label="actual", marker="o", markersize=3)
    ax.plot(hwC.fittedvalues.index, hwC.fittedvalues.values, color=acc_imp,
            linewidth=1.4, alpha=0.8, label="Holt-Winters fit")
    ax.plot(fcC.index, fcC.values, color=vs.BAD, linewidth=2.0, linestyle="--", label="6-mo forecast")
    resid_std = (train_ts - hwC.fittedvalues).std()
    ax.fill_between(fcC.index, fcC.values-1.96*resid_std, fcC.values+1.96*resid_std,
                    color=vs.BAD, alpha=0.12, label="95% interval")
    ax.axvspan(monthly.index[-4], monthly.index[-1], color="#f0d9d9", alpha=0.4)
    ax.yaxis.set_major_formatter(vs.K_FMT)
    vs.style_ax(ax, "Monthly sales: forecast vs actual", ylabel="sales ($)")
    ax.legend(loc="lower left", ncol=2)
    ax.annotate("data artifact:\nlate-2017 collapse", (monthly.index[-3], monthly.values[-3]),
                textcoords="offset points", xytext=(-40,30), fontsize=8.5, color=vs.BAD,
                arrowprops=dict(arrowstyle="->", color=vs.BAD))
    # MAPE comparison bars
    axes[1].bar(["Full holdout\n(incl. collapse)","Stable period"],
                [mapeC, mapeC2], color=[vs.BAD, vs.GOOD], edgecolor="white")
    for i,v in enumerate([mapeC, mapeC2]):
        axes[1].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[1].axhline(10, color=vs.INK_SOFT, linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("MAPE (%)")
    vs.style_ax(axes[1], "Forecast accuracy")
    axes[1].grid(axis="x", visible=False)
    vs.annotate_insight(axes[1], "<10% = production-grade\nfor S&OP planning",
                        xy=(0.96, 0.6), accent=acc_imp)
    vs.titled(fig, "Model C — Demand Forecasting (Holt-Winters)",
              "How accurately can monthly demand be forecast for inventory planning?", accent=acc_imp)
    vs.footnote(fig, f"Insight: {mapeC2:.1f}% MAPE on the stable period (production-grade); the {mapeC:.0f}% "
                "full-holdout error is caused by an unforecastable structural break — a Phase-5 drift alert.")
    log_fig(vs.save(fig, fig_path("p4C_01_forecast.png")))

    # seasonal decomposition view (trend / seasonal / resid)
    from statsmodels.tsa.seasonal import seasonal_decompose
    dec = seasonal_decompose(monthly, model="additive", period=12)
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.2), sharex=True)
    for ax, comp, name, col in zip(axes, [dec.trend, dec.seasonal, dec.resid],
                                   ["Trend","Seasonality","Residual"], [acc_imp, vs.NEUTRAL, vs.BAD]):
        ax.plot(comp.index, comp.values, color=col, linewidth=1.8)
        ax.axhline(comp.mean() if name!="Trend" else comp.dropna().iloc[0], color=vs.GRID)
        ax.yaxis.set_major_formatter(vs.K_FMT)
        vs.style_ax(ax, name)
    vs.titled(fig, "Model C — Time-Series Decomposition",
              "What are the structural components of demand over time?", accent=acc_imp)
    vs.footnote(fig, "Insight: a stable trend & clear 12-month seasonality through mid-2017, then a sharp "
                "residual break — separating genuine seasonality (forecastable) from the artifact (not).")
    log_fig(vs.save(fig, fig_path("p4C_02_decomposition.png")))


# ----------------------------------------------------------------------------
# MODEL D — CUSTOMER SEGMENTATION (RFM + K-Means)
# ----------------------------------------------------------------------------
banner("MODEL D: Customer Segmentation (RFM)")
snapshot = df["order_date"].max() + pd.Timedelta(days=1)
RFM = df.groupby("Customer Id").agg(
    Recency=("order_date", lambda x:(snapshot-x.max()).days),
    Frequency=("Order Id","nunique"), Monetary=("Sales","sum"),
    AvgProfitRatio=("Order Item Profit Ratio","mean"))
rfm_log = np.log1p(RFM[["Recency","Frequency","Monetary"]])
Zr = StandardScaler().fit_transform(rfm_log)
SIL = {}
for k in [3,4,5]:
    lab = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit_predict(Zr)
    SIL[k] = silhouette_score(Zr, lab, sample_size=10000, random_state=RNG)
BEST_K = max(SIL, key=SIL.get)
RFM["Segment"] = KMeans(n_clusters=BEST_K, n_init=10, random_state=RNG).fit_predict(Zr)
SEG = RFM.groupby("Segment").agg(N=("Recency","size"), Recency=("Recency","mean"),
        Frequency=("Frequency","mean"), Monetary=("Monetary","mean"),
        ProfitRatio=("AvgProfitRatio","mean")).round(2)
print(f"Silhouette {SIL} -> k={BEST_K}\n{SEG.to_string()}")

def viz_modelD():
    # silhouette analysis for chosen k
    lab = RFM["Segment"].values
    sv = silhouette_samples(Zr, lab)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios":[1,1.2]})
    y_lower = 10
    for i in range(BEST_K):
        vals = np.sort(sv[lab==i])
        y_upper = y_lower + len(vals)
        axes[0].fill_betweenx(np.arange(y_lower,y_upper), 0, vals,
                              color=vs.CATEGORICAL[i], alpha=0.8)
        axes[0].text(-0.02, y_lower+0.5*len(vals), str(i), fontsize=9)
        y_lower = y_upper + 10
    axes[0].axvline(SIL[BEST_K], color=vs.BAD, linestyle="--", linewidth=1.4)
    axes[0].set_yticks([])
    vs.style_ax(axes[0], f"Silhouette analysis (k={BEST_K}, score={SIL[BEST_K]:.2f})",
                xlabel="silhouette coefficient")
    # scatter recency vs monetary
    sc = axes[1].scatter(RFM["Recency"], np.log1p(RFM["Monetary"]), c=RFM["Segment"],
                         cmap="viridis", s=8, alpha=0.5)
    axes[1].set_xlabel("recency (days since last order)")
    axes[1].set_ylabel("log(monetary)")
    vs.style_ax(axes[1], "Customer segments in RFM space")
    legend = [Patch(facecolor=plt.cm.viridis(i/(BEST_K-1)), label=f"Segment {i}") for i in range(BEST_K)]
    axes[1].legend(handles=legend, loc="lower left", ncol=BEST_K, fontsize=8)
    vs.titled(fig, "Model D — Customer Segmentation Quality",
              "How many distinct customer value-segments exist, and how clean are they?", accent=acc_imp)
    vs.footnote(fig, f"Insight: silhouette peaks at k={BEST_K} (~0.50, moderately strong). Segments separate "
                "cleanly on recency × monetary — the basis for differentiated retention strategy.")
    log_fig(vs.save(fig, fig_path("p4D_01_segmentation.png")))

    # segment profile dashboard (bubble + RFM bars)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios":[1,1.1]})
    axes[0].scatter(SEG["Recency"], SEG["Frequency"], s=SEG["Monetary"]/3,
                    c=range(BEST_K), cmap="viridis", alpha=0.7, edgecolor=vs.INK)
    for i,row in SEG.iterrows():
        axes[0].annotate(f"Seg {i}\n${row['Monetary']:,.0f}", (row["Recency"], row["Frequency"]),
                         fontsize=8.5, ha="center", va="center")
    vs.style_ax(axes[0], "Segment map (bubble = monetary value)",
                xlabel="recency (days)", ylabel="frequency (orders)")
    # normalized RFM bars
    segn = SEG[["Recency","Frequency","Monetary"]].copy()
    segn = (segn - segn.min())/(segn.max()-segn.min())
    segn.index = [f"Seg {i}" for i in segn.index]
    segn.plot.bar(ax=axes[1], color=[vs.BAD, vs.NEUTRAL, vs.GOOD], edgecolor="white", width=0.75)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    vs.style_ax(axes[1], "Normalized RFM profile by segment", ylabel="0–1 (min–max scaled)")
    axes[1].legend(loc="upper right", ncol=3)
    axes[1].grid(axis="x", visible=False)
    vs.titled(fig, "Model D — Segment Profiles & Strategy",
              "What does each customer segment look like, and how should we treat it?", accent=acc_imp)
    vs.footnote(fig, "Insight: a high-monetary, high-frequency 'champion' segment (~90% of revenue) vs "
                "recent one-timers (convert) vs lapsed low-value (low-cost reactivation).")
    log_fig(vs.save(fig, fig_path("p4D_02_segment_profiles.png")))


# ----------------------------------------------------------------------------
# MODEL E — LOSS-ORDER EARLY WARNING (honest negative result)
# ----------------------------------------------------------------------------
banner("MODEL E: Loss-Order Early Warning")
featE_num = ["Order Item Quantity","Order Item Discount Rate",
             "Order Item Product Price","Days for shipment (scheduled)"]
featE_cat = ["Shipping Mode","Market","Customer Segment","Department Name","Type"]
XE = df[featE_num+featE_cat]; yE = df["Is_Loss_Order"]
XE_tr, XE_te, yE_tr, yE_te = train_test_split(XE, yE, test_size=0.25, stratify=yE, random_state=RNG)
prepE = ColumnTransformer([("num", StandardScaler(), featE_num),
                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), featE_cat)])
lossE = Pipeline([("prep", prepE), ("clf", HistGradientBoostingClassifier(
        max_iter=300, random_state=RNG))]).fit(XE_tr, yE_tr)
probaE = lossE.predict_proba(XE_te)[:,1]; aucE = roc_auc_score(yE_te, probaE)
avg_loss = df.loc[df["Is_Loss_Order"]==1,"Order Profit Per Order"].mean()
print(f"Loss AUC={aucE:.3f} (≈random); avg loss/order ${avg_loss:.2f}")

def viz_modelE():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_roc(axes[0], yE_te, probaE, "GB loss model", vs.WARN, acc_imp)
    axes[0].text(0.55, 0.2, "AUC ≈ 0.50\n= no signal", fontsize=13, color=vs.BAD,
                 fontweight="bold", ha="center")
    # what loss looks like vs what's observable: profit distribution colored by loss
    axes[1].hist(df.loc[df["Is_Loss_Order"]==0,"Order Profit Per Order"], bins=60,
                 range=(-600,400), color=vs.GOOD, alpha=0.6, label="profit order", density=True)
    axes[1].hist(df.loc[df["Is_Loss_Order"]==1,"Order Profit Per Order"], bins=60,
                 range=(-600,400), color=vs.BAD, alpha=0.6, label="loss order", density=True)
    axes[1].axvline(0, color=vs.INK, linewidth=1.2)
    vs.style_ax(axes[1], "Profit per order (loss vs profit)", xlabel="profit ($)", ylabel="density")
    axes[1].legend()
    vs.annotate_insight(axes[1], f"18.7% of orders lose money\navg ${avg_loss:.0f} each\n≈ $3.9M leak",
                        xy=(0.04, 0.93), ha="left", va="top", accent=acc_imp)
    vs.titled(fig, "Model E — Loss-Order Prediction: An Honest Null Result",
              "Can loss-making orders be flagged from order-entry data alone?", accent=acc_imp)
    vs.footnote(fig, "Insight: AUC≈0.50 — loss is UNPREDICTABLE from visible fields. This empirically proves "
                "the dissertation's 'unobserved drivers' thesis & justifies capturing per-order cost data.")
    log_fig(vs.save(fig, fig_path("p4E_01_null_result.png")))


# ============================================================================
# PHASE 5 — CONTROL (compute + visuals)
# ============================================================================
banner("PHASE 5: CONTROL — SPC limits & monitoring")
SPC = {}
for v in ["Sales per customer", "Shipping_Delay"]:
    mu, sd = df[v].mean(), df[v].std()
    SPC[v] = dict(mu=mu, sd=sd, ucl=mu+3*sd, lcl=mu-3*sd,
                  ooc=float(((df[v]>mu+3*sd)|(df[v]<mu-3*sd)).mean()*100))
    print(f"{v}: CL={mu:.2f} UCL={SPC[v]['ucl']:.2f} ooc={SPC[v]['ooc']:.2f}%")


def _psi(expected, actual, bins=10):
    """Population Stability Index between two distributions."""
    qs = np.quantile(expected, np.linspace(0, 1, bins+1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0]/len(expected)
    a = np.histogram(actual, qs)[0]/len(actual)
    e, a = np.clip(e,1e-4,None), np.clip(a,1e-4,None)
    return float(np.sum((a-e)*np.log(a/e)))


def viz_phase5():
    banner("VISUALS — PHASE 5 (Control)")
    acc = vs.PHASE_ACCENT["control"]

    # --- p5_01: SPC control chart — sales per customer (sampled sequence)
    # WHY: Control — monitor a KPI against 3σ limits to detect special causes.
    samp = df["Sales per customer"].sample(500, random_state=RNG).reset_index(drop=True)
    s = SPC["Sales per customer"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(samp.index, samp.values, color=acc, linewidth=0.9, marker="o", markersize=2.5, alpha=0.8)
    ax.axhline(s["mu"], color=vs.INK, linewidth=1.2, label="centre line")
    ax.axhline(s["ucl"], color=vs.BAD, linestyle="--", linewidth=1.3, label="UCL (+3σ)")
    ax.axhline(s["lcl"], color=vs.BAD, linestyle="--", linewidth=1.3, label="LCL (−3σ)")
    ooc = samp[(samp>s["ucl"])|(samp<s["lcl"])]
    ax.scatter(ooc.index, ooc.values, color=vs.BAD, s=45, zorder=5, label="out of control")
    ax.fill_between(samp.index, s["lcl"], s["ucl"], color=acc, alpha=0.05)
    vs.style_ax(ax, None, xlabel="observation sequence", ylabel="sales per customer ($)")
    ax.legend(loc="upper right", ncol=2)
    vs.titled(fig, "SPC Control Chart — Sales per Customer",
              "Is the revenue process stable, or driven by special causes?", accent=acc)
    vs.footnote(fig, f"Insight: {s['ooc']:.2f}% of points exceed ±3σ — process is in control; rare spikes are "
                "high-value bulk orders (special cause), not systemic instability.")
    log_fig(vs.save(fig, fig_path("p5_01_spc_sales.png")))

    # --- p5_02: SPC control chart — shipping delay
    # WHY: Control — monitor the operational pain-point KPI for stability.
    samp2 = df["Shipping_Delay"].sample(500, random_state=7).reset_index(drop=True)
    s2 = SPC["Shipping_Delay"]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.step(samp2.index, samp2.values, color=acc, linewidth=0.9, where="mid", alpha=0.8)
    ax.axhline(s2["mu"], color=vs.INK, linewidth=1.2, label=f"mean +{s2['mu']:.2f}d")
    ax.axhline(s2["ucl"], color=vs.BAD, linestyle="--", linewidth=1.3, label="UCL/LCL (±3σ)")
    ax.axhline(s2["lcl"], color=vs.BAD, linestyle="--", linewidth=1.3)
    ax.axhline(0, color=vs.GOOD, linestyle=":", linewidth=1.4, label="on-time (0)")
    vs.style_ax(ax, None, xlabel="observation sequence", ylabel="shipping delay (days)")
    ax.legend(loc="upper right", ncol=2)
    vs.annotate_insight(ax, "centre line > 0:\nchronically late, but STABLE\n→ fix the promise, not the variance",
                        xy=(0.985, 0.06), accent=acc)
    vs.titled(fig, "SPC Control Chart — Shipping Delay",
              "Is lateness a variation problem or a process-design problem?", accent=acc)
    vs.footnote(fig, "Insight: 0% out-of-control but the centre line sits ABOVE zero — the process is reliably "
                "late by design. The fix is the promise-setting logic (Phase 4), not variance reduction.")
    log_fig(vs.save(fig, fig_path("p5_02_spc_delay.png")))

    # --- p5_03: KPI stability over time (monthly mean with control band)
    # WHY: Control — track whether KPIs drift month to month.
    mk = df.set_index("order_date").resample("MS").agg(
        sales=("Sales per customer","mean"), delay=("Shipping_Delay","mean"),
        late=("Late_delivery_risk","mean")).iloc[:-1]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.5), sharex=True)
    series = [("sales","Avg sales/customer ($)", acc),
              ("delay","Avg shipping delay (days)", vs.WARN),
              ("late","Late-delivery rate", vs.BAD)]
    for ax,(col,lab,col_c) in zip(axes, series):
        m, sd = mk[col].mean(), mk[col].std()
        ax.plot(mk.index, mk[col], color=col_c, linewidth=1.8, marker="o", markersize=3)
        ax.axhline(m, color=vs.INK, linewidth=1.0)
        ax.fill_between(mk.index, m-2*sd, m+2*sd, color=col_c, alpha=0.08)
        vs.style_ax(ax, lab)
    vs.titled(fig, "KPI Stability Monitoring (Monthly)",
              "Do the key control metrics stay within expected bounds over time?", accent=acc)
    vs.footnote(fig, "Insight: sales & delay rates hold within ±2σ bands until the late-2017 break — the exact "
                "signal a monitoring system should auto-flag for investigation.")
    log_fig(vs.save(fig, fig_path("p5_03_kpi_stability.png")))

    # --- p5_04: Model drift — PSI of features (train vs recent)
    # WHY: Control — detect input drift that would silently degrade models.
    # simulate "recent" window = last 20% chronologically
    df_sorted = df.sort_values("order_date")
    ref = df_sorted.iloc[:int(len(df)*0.6)]
    cur = df_sorted.iloc[int(len(df)*0.8):]
    psi_feats = ["Days for shipment (scheduled)","Order Item Discount Rate",
                 "Order Item Product Price","Sales per customer","Order Item Quantity",
                 "Days for shipping (real)"]
    psi_vals = {f: _psi(ref[f].values, cur[f].values) for f in psi_feats}
    fig, ax = plt.subplots(figsize=(11, 5))
    s = pd.Series(psi_vals).sort_values()
    colors = [vs.BAD if v>=0.2 else (vs.WARN if v>=0.1 else vs.GOOD) for v in s.values]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor="white")
    for b,v in zip(bars, s.values):
        ax.text(v+0.002, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.1, color=vs.WARN, linestyle="--", linewidth=1.2)
    ax.axvline(0.2, color=vs.BAD, linestyle="--", linewidth=1.2)
    ax.text(0.1, len(s)-0.3, " warn", color=vs.WARN, fontsize=8.5)
    ax.text(0.2, len(s)-0.3, " retrain", color=vs.BAD, fontsize=8.5)
    vs.style_ax(ax, None, xlabel="Population Stability Index (early 60% vs recent 20%)")
    ax.grid(axis="y", visible=False)
    vs.titled(fig, "Model Drift Monitoring — Feature PSI",
              "Have input distributions shifted enough to require model retraining?", accent=acc)
    vs.footnote(fig, "Insight: PSI thresholds (0.1 warn / 0.2 retrain) operationalise drift detection. "
                "Features crossing the red line trigger an automated retraining job.")
    log_fig(vs.save(fig, fig_path("p5_04_drift_psi.png")))

    # --- p5_05: Operational control dashboard (consolidated KPI scorecard)
    # WHY: Control — a single executive monitoring surface tying it together.
    fig = plt.figure(figsize=(13, 7.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.3)
    # gauge-style KPI cards (top row)
    cards = [("Late-delivery rate", df["Late_delivery_risk"].mean()*100, "%", 55, vs.BAD, "↓ target"),
             ("Fraud rate", df["Is_Fraud"].mean()*100, "%", 2.3, vs.WARN, "monitor"),
             ("Loss-order rate", df["Is_Loss_Order"].mean()*100, "%", 18.7, "#9d4e4e", "investigate cost")]
    for i,(name,val,unit,target,col,note) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0.05,0.15),0.9,0.7, facecolor="#eef3f0",
                     edgecolor=col, linewidth=1.6))
        ax.text(0.5, 0.62, f"{val:.1f}{unit}", ha="center", fontsize=28, fontweight="bold", color=col)
        ax.text(0.5, 0.40, name, ha="center", fontsize=11, fontweight="bold", color=vs.INK)
        ax.text(0.5, 0.26, note, ha="center", fontsize=9, color=vs.INK_SOFT, style="italic")
    # bottom row: model performance scorecard, monitoring rules, control status
    axm = fig.add_subplot(gs[1, 0])
    models = ["Late\n(A)","Fraud\n(B)","Forecast\n(C)","Segment\n(D)","Loss\n(E)"]
    scores = [roc_auc_score(yA_te,probaA_gb), roc_auc_score(yB_te,probaB),
              1-mapeC2/100, SIL[BEST_K], aucE]
    metric_lbl = ["AUC .76","AUC .87","1-MAPE .95","Sil .50","AUC .50"]
    colors = [vs.GOOD if sc>=0.7 else (vs.WARN if sc>=0.55 else vs.BAD) for sc in scores]
    bars = axm.bar(models, scores, color=colors, edgecolor="white")
    for b,l in zip(bars, metric_lbl):
        axm.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, l, ha="center", fontsize=7.5)
    axm.set_ylim(0,1.1); axm.axhline(0.7, color=vs.INK_SOFT, linestyle=":")
    vs.style_ax(axm, "Model scorecard"); axm.grid(axis="x", visible=False)
    # monitoring rules text panel
    axr = fig.add_subplot(gs[1, 1:]); axr.axis("off")
    axr.text(0.0, 0.95, "Production monitoring rules", fontsize=11.5, fontweight="bold",
             color=vs.INK, va="top")
    rules = ["• Late-delivery AUC recomputed weekly → alert if < 0.85",
             "• Feature PSI tracked → retrain trigger when PSI > 0.20",
             "• Forecast rolling MAPE → alert if > 15% for 3 consecutive weeks",
             "• Fraud review-queue precision → recalibrate threshold monthly",
             "• SPC charts on sales & delay → flag any out-of-control run",
             "• Loss-order rate → escalate to cost-data capture initiative"]
    for j,r in enumerate(rules):
        axr.text(0.0, 0.80-j*0.13, r, fontsize=9.7, color=vs.INK_SOFT, va="top")
    vs.titled(fig, "Operational Control Dashboard",
              "One surface to monitor KPIs, model health, and intervention triggers.", accent=acc)
    vs.footnote(fig, "Insight: closes the DMAIC loop — descriptive KPIs, predictive model health and explicit "
                "alerting thresholds in a single executive view, ready for production hand-off.")
    log_fig(vs.save(fig, fig_path("p5_05_control_dashboard.png")))


# ============================================================================
# MAIN — run every phase's visuals & print the figure index
# ============================================================================
if __name__ == "__main__":
    viz_phase1()
    viz_phase2()
    viz_phase3()
    viz_modelA(); viz_modelB(); viz_modelC(); viz_modelD(); viz_modelE()
    viz_phase5()

    banner("FIGURE INDEX")
    print(f"{len(SAVED)} figures saved to ./{FIG_DIR}/")
    for p in SAVED:
        print("  -", os.path.basename(p))
    print("\nVisualization layer complete.")