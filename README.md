# Facial IRT Calibration Generalizability

Reproducible analysis code and independently verified results for a secondary analysis testing whether facial infrared thermography (IRT) calibration residuals differ systematically across demographic subgroups (sex, ethnicity).

**Manuscript:** "Calibration Generalizability of Facial Infrared Thermography Across Demographic Subgroups: A Secondary Analysis of a Public Clinical Measurement Dataset"
**Author:** Dr. Faiz Moosa ([ORCID](https://orcid.org/0009-0001-1702-7707) | [LinkedIn](https://www.linkedin.com/in/dr-faiz-t/))
**Target journal:** Quantitative InfraRed Thermography Journal (submission in progress)

## What this is

A demographic-blind calibration model was built for two facial IRT device types (FLIR, ICI) using data from PhysioNet's publicly available "Facial and oral temperature data from a large set of human subject volunteers" dataset. Post-calibration residuals were tested for association with sex, ethnicity, and age band, adjusting for ambient temperature, humidity, distance, and device — including a paired cross-device sensitivity analysis and explicit missingness diagnostics.

**Finding:** No statistically significant demographic heterogeneity in calibration residuals was detected. This is reported as a precision-bounded null result, not a demonstration of subgroup equivalence — see the manuscript's Discussion for the full caveats (wide limits of agreement, a young study population, small subgroup cell sizes, and sex-associated missingness on one device).

## Data source

- Dataset: [PhysioNet — Facial and oral temperature data from a large set of human subject volunteers](https://physionet.org/content/face-oral-temp-data/1.0.0/) (v1.0.0), DOI 10.13026/3bhc-9065
- License: CC0 1.0 Universal (public domain) — open access, no credentialing required
- Not redistributed here; download directly from PhysioNet (instructions below)

## Repository contents

| File | Description |
|---|---|
| `run_analysis.py` | Full offline reproduction script — blind calibration models, subgroup regression (HC3 robust SEs), bootstrap CIs, Bland–Altman analysis, paired sensitivity analysis, missingness tests |
| `verification/report.txt` | Full plain-text output log from an independent run of this script against the real PhysioNet data |
| `verification/table1_demographics.csv` | Demographic composition of the analytic sample |
| `verification/table2_regression_coefficients.csv` | Adjusted regression coefficients (FLIR, ICI, paired sensitivity model) |
| `verification/table3_subgroup_metrics.csv` | Subgroup bias/MAE/RMSE with bootstrap 95% CIs |
| `figures/` | Bland–Altman plots by ethnicity and sex, both devices |

## Reproducing this analysis

1. Install dependencies:
   ```
   pip install pandas numpy statsmodels matplotlib scipy
   ```
2. Download the two source files from PhysioNet (open access, no login needed) and place them in this folder:
   - https://physionet.org/content/face-oral-temp-data/1.0.0/FLIR_groups1and2.csv
   - https://physionet.org/content/face-oral-temp-data/1.0.0/ICI_groups1and2.csv
3. Run:
   ```
   python run_analysis.py
   ```
4. Results are written to a new `report/` folder — compare against `verification/` here to confirm reproducibility.

This script was independently verified twice: once by dry-run against synthetic data matching the real schema, and once against the real PhysioNet CSVs via Google Colab. Both runs matched the manuscript's reported numbers to within expected bootstrap-resampling noise.

## Citation

If you use this code, please cite the manuscript (citation details to be added upon publication) and the original dataset:

> Wang Q, Zhou Y, Ghassemi P, Chenna D, Chen M, Casamento J, Pfefer TJ, McBride D. Facial and oral temperature data from a large set of human subject volunteers (version 1.0.0). PhysioNet; 2023. https://doi.org/10.13026/3bhc-9065

## License

Code in this repository is released under the MIT License (see `LICENSE`). The underlying dataset is CC0 1.0 Universal, per PhysioNet.
