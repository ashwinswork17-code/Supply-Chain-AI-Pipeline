# Supply Chain AI Pipeline — Enhanced Visualization Layer

A publication-quality visual storytelling layer for the DMAIC supply-chain
analysis. **30 figures**, 2–6 per DMAIC phase, each carrying a title, a
business-question subtitle, an interpretation footnote, and a consistent house
style (one signature accent colour per phase; semantic red/green for
risk/positive used consistently throughout).

## How to run

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn shap
# point DATA_PATH at your CSV (env var or edit the constant in sc_enhanced.py)
export DATA_PATH=/path/to/DataCoSupplyChainDataset.csv
python sc_enhanced.py
```

Figures are written to `./figures/`. `viz_style.py` holds the shared visual
identity (palette, typography, header/footnote helpers) and must sit next to
`sc_enhanced.py`.

## Design system (`viz_style.py`)

- **Palette** — cool slate base (`INK`/`INK_SOFT`), hairline grids, white
  panels. Semantic colours are fixed across all 30 charts: `GOOD` green =
  positive/on-time/profit, `BAD` red = risk/late/loss/fraud, `WARN` amber =
  caution.
- **Phase accents** — each DMAIC phase has one signature colour so a reader can
  identify the phase from the chart alone: Define steel-blue, Measure teal,
  Analyze violet, Improve sienna, Control pine.
- **Every figure** gets `titled()` (heavy title + italic business question +
  accent rule) and `footnote()` (source + one-line insight), so the deck reads
  as a single coherent deliverable.

---

## Phase 1 — DEFINE  *(frame the data, structure and scope)*

| File | Type | Why it's in DEFINE | Business insight |
|---|---|---|---|
| `p1_01_data_composition.png` | Donut + stat cards | Establishes WHAT the data covers — the 4-family variable taxonomy and dataset scale | Breadth spans inventory, logistics, performance & context → enables an *integrated*, not siloed, analysis |
| `p1_02_business_composition.png` | Grouped horizontal bars (small multiple) | Define needs the shape of the business — where orders concentrate | Standard Class & Consumer segment dominate; "Late delivery" is a large outcome share → names the problem |
| `p1_03_target_baselines.png` | Stacked proportion bars | Names the OUTCOMES to be modelled and sets base rates | Imbalanced targets (fraud 2.3%, lateness ~55%, loss 18.7%) — the benchmarks every Phase-4 model must beat |
| `p1_04_missingness_matrix.png` | Missingness heatmap | Data-readiness proof before Measure | Modelling fields fully populated → no imputation bias |

## Phase 2 — MEASURE  *(quantify distributions, quality, KPIs)*

| File | Type | Why it's in MEASURE | Business insight |
|---|---|---|---|
| `p2_01_kpi_distributions.png` | Histograms + KDE (small multiple) | Characterise each KPI's distribution & skew | Revenue/profit KPIs strongly skewed (mean ≠ median) → justifies non-parametric tests & robust models |
| `p2_02_outlier_share.png` | Horizontal bar (data-quality) | Quantify outlier prevalence (which are retained) | Profit & delay carry 10–20% outliers — real operational variability, so retained not deleted |
| `p2_03_sales_by_market_box.png` | Boxplots by group | Compare KPI spread across operational groups | Median spend similar across markets; Europe/Pacific Asia have wider upper tails (high-value orders) |
| `p2_04_profit_violin.png` | Violin plots | Reveal full distribution shape incl. the loss tail | All shipping modes share a loss tail below zero → profitability problems aren't shipping-driven |
| `p2_05_promise_vs_actual.png` | Paired bars + delay histogram | Surface THE core operational gap | Every mode ships slower than promised (mean +0.57d) — lateness is built into promise-setting |

## Phase 3 — ANALYZE  *(relationships, tests, regression, PCA, clustering)*

| File | Type | Why it's in ANALYZE | Business insight |
|---|---|---|---|
| `p3_01_corr_heatmap.png` | Masked correlation heatmap | Quantify all monotonic relationships at once | Sales↔Sales-per-customer (0.99) is structural redundancy; discount (−0.13) the only operational lever linked to revenue |
| `p3_02_hypothesis_tests.png` | 3-panel (error bars + stacked + box) | Visual evidence for the 3 formal tests | Clean dissociation: market drives sales (ANOVA); mode drives lateness (χ², V=0.46) but NOT sales (Kruskal n.s.) |
| `p3_03_ols_coefficients.png` | Coefficient plot with CIs | Direction/magnitude/significance of drivers | Discount strongly negative (~−$204); R²=0.018 → operational levers explain <2% of sales |
| `p3_04_pca_scree.png` | Scree + cumulative (dual axis) | Dimensionality of the signal | 3 components ≈ 64%, 5 ≈ 88% → low-dimensional, supports compact dashboards |
| `p3_05_pca_biplot.png` | PCA loadings biplot | Interpret what the components mean | Revenue axis vs separate margin axis; logistics clusters near origin (weak influence) |
| `p3_06_cluster_profiles.png` | Cluster heatmap + size bars | Reveal hidden operational segments | A loss-making segment (profit ratio ≈ −1.17, ~15k orders) isolated at average sales/discount |

## Phase 4 — IMPROVE  *(five AI business models — eval, explainability, impact)*

**Model A — Late-Delivery Risk**
| File | Type | Insight |
|---|---|---|
| `p4A_01_evaluation.png` | ROC + PR + confusion + importance | AUC≈0.77; scheduled days & shipping mode dominate |
| `p4A_02_shap.png` | SHAP beeswarm | High scheduled-days & Standard-Class push risk up — auditable drivers |
| `p4A_03_business_impact.png` | Cumulative gains + threshold curves | Top-20% intervention captures 34% of late orders at 94% precision |

**Model B — Fraud Detection**
| File | Type | Insight |
|---|---|---|
| `p4B_01_evaluation.png` | ROC + PR + confusion + lift | AUC≈0.87, 87% fraud recall; unsupervised gives ~1× lift (labels matter) |
| `p4B_02_review_targeting.png` | Fraud-capture gains curve | A risk-ranked review queue concentrates fraud into the top deciles |

**Model C — Demand Forecasting**
| File | Type | Insight |
|---|---|---|
| `p4C_01_forecast.png` | Forecast vs actual + MAPE bars | 5.1% MAPE on stable period (production-grade); 34% full-holdout = structural break |
| `p4C_02_decomposition.png` | Trend / seasonal / residual | Stable trend + 12-mo seasonality, then a sharp residual break (the artifact) |

**Model D — Customer Segmentation (RFM)**
| File | Type | Insight |
|---|---|---|
| `p4D_01_segmentation.png` | Silhouette + RFM scatter | Silhouette peaks at k=3 (~0.50); clean separation on recency × monetary |
| `p4D_02_segment_profiles.png` | Bubble map + normalized RFM bars | Champions (~90% of revenue) vs recent one-timers vs lapsed low-value |

**Model E — Loss-Order Early Warning**
| File | Type | Insight |
|---|---|---|
| `p4E_01_null_result.png` | ROC (≈0.50) + profit distribution | Loss is UNPREDICTABLE from visible fields → proves "unobserved drivers"; capture cost data |

## Phase 5 — CONTROL  *(SPC, KPI stability, drift, operational monitoring)*

| File | Type | Why it's in CONTROL | Business insight |
|---|---|---|---|
| `p5_01_spc_sales.png` | SPC control chart (±3σ) | Monitor revenue KPI for special causes | 0.26% beyond ±3σ — in control; rare spikes are bulk orders |
| `p5_02_spc_delay.png` | SPC step chart | Monitor the operational pain-point | 0% out-of-control but centre line > 0 → reliably late *by design* (fix the promise) |
| `p5_03_kpi_stability.png` | Multi-KPI time series + bands | Track month-to-month drift | KPIs hold within ±2σ until the late-2017 break — the signal to auto-flag |
| `p5_04_drift_psi.png` | Feature PSI bar chart | Detect input drift that degrades models | PSI thresholds (0.1 warn / 0.2 retrain) operationalise drift detection |
| `p5_05_control_dashboard.png` | Executive composite | One surface tying KPIs + model health + triggers | Closes the DMAIC loop; ready for production hand-off |

---

## Files in this delivery

- `sc_enhanced.py` — full pipeline (compute + 30 visualizations)
- `viz_style.py` — shared visual identity module (import dependency)
- `figures/` — 30 phase figures (`p1_*` … `p5_*`) + original `ai_models_dashboard.png`
- `contact_sheet.png` — all 30 figures on one page for quick review
- `VISUALIZATION_README.md` — this file
