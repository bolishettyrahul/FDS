# FDS — Case Studies 2 & 5 (Batch 2)

Fundamentals of Data Science (22CSC59) · **Bolishetty Rahul · Roll No. 160124733167** · B.E. CSE, CBIT Hyderabad

| File | What it is |
|---|---|
| [`FDS_Case_Study_Report_Rahul_160124733167.pdf`](FDS_Case_Study_Report_Rahul_160124733167.pdf) | Case study report (PDF) |
| [`Case_Studies_2_and_5.ipynb`](Case_Studies_2_and_5.ipynb) | Jupyter notebook with all code, outputs, plots and observations |
| [`case_studies_2_and_5.py`](case_studies_2_and_5.py) | Same code as a standalone Python script |
| `figures/` | All plots produced by the notebook |
| `data/` | `EastWestAirlines.xlsx`, `winequality-red.csv`, `winequality-white.csv` |

## Case Study 2 — Airline customer segmentation (clustering)
- **Preprocessing:** checked for missing values, removed duplicates, dropped near-constant columns, removed outliers with the Z-score rule (\|z\| > 3), and scaled the features with StandardScaler.
- **EDA:** descriptive statistics, histograms, box plots, scatter plots, a correlation heatmap, a pair plot and PCA.
- **K-means:** the elbow method, silhouette score and Davies–Bouldin index all point to **k = 4** (silhouette 0.331).
- **Hierarchical clustering:** single, complete, average and ward linkage were compared; **ward** gave the most useful clusters.
- **DBSCAN:** a grid of eps × min_samples values was tested.
- **Segments found:**
  - Low-activity members (59%)
  - Credit-card earners (32%)
  - Frequent flyers (7%)
  - Topflight qualifiers (2%)

## Case Study 5 — Wine quality (regression)
- **Descriptive statistics:** the coefficient of variation shows residual sugar varies most. Pearson correlation between alcohol and quality is r = +0.476.
- **EDA:** histogram, box plot and correlation heatmap.
- **Multiple linear regression** (80/20 split, `random_state=42`):
  - MAE 0.504
  - MSE 0.390
  - RMSE 0.625
  - R² 0.403
- **Key predictors:** alcohol (+) and volatile acidity (−). Multicollinearity was checked with VIF.
- **Conclusion:** the model is useful for screening, but it should not replace sensory evaluation.

```bash
pip install -r requirements.txt
jupyter notebook Case_Studies_2_and_5.ipynb      # or: python case_studies_2_and_5.py
```
