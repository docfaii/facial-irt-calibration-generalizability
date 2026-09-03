"""
Offline reproduction of the facial-IRT calibration-heterogeneity analysis.

Reproduces every step from the manuscript's Methods section:
  1. Blind calibration model (FLIR, ICI)
  2. Residual computation
  3. Subgroup heterogeneity regression (OLS, HC3 robust SEs)
  4. Subgroup bias / MAE / RMSE with bootstrap CIs
  5. Bland-Altman by subgroup (numeric + plots)
  6. Paired (matched-date, both-device) sensitivity analysis, clustered SEs
  7. Missingness association tests (chi-square / Fisher exact)

USAGE:
    Download these two files from PhysioNet first and place them next to
    this script (or edit DATA_DIR below):
      https://physionet.org/content/face-oral-temp-data/1.0.0/FLIR_groups1and2.csv
      https://physionet.org/content/face-oral-temp-data/1.0.0/ICI_groups1and2.csv

    Then run:
      python run_analysis.py

OUTPUT (written to ./report/):
    report/report.txt                  -- full human-readable report, every number, every test
    report/table1_demographics.csv
    report/table2_regression_coefficients.csv
    report/table3_subgroup_metrics.csv
    report/table4_missingness.csv
    report/FLIR_BlandAltman_Ethnicity.png
    report/FLIR_BlandAltman_Sex.png
    report/ICI_BlandAltman_Ethnicity.png
    report/ICI_BlandAltman_Sex.png
    report/residuals_flir.csv          -- full per-subject residuals, for independent spot-checking
    report/residuals_ici.csv
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scistats

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR = Path(".")
FLIR_PATH = DATA_DIR / "FLIR_groups1and2.csv"
ICI_PATH = DATA_DIR / "ICI_groups1and2.csv"
OUT_DIR = Path("report")
MIN_CELL_N = 10  # exclusion threshold for inferential subgroup comparison
N_BOOT = 1000
RANDOM_SEED = 42

EXCLUDED_ETHNICITY = "American Indian or Alaskan Native"
AGE_LE30 = {"18-20", "21-25", "21-30", "26-30"}
AGE_GT30 = {"31-40", "41-50", "51-60", ">60"}


def fail(msg):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def check_inputs():
    if not FLIR_PATH.exists() or not ICI_PATH.exists():
        fail(
            "Could not find FLIR_groups1and2.csv and/or ICI_groups1and2.csv.\n"
            f"Expected them in: {DATA_DIR.resolve()}\n\n"
            "Download from PhysioNet (open access, no credentialing required):\n"
            "  https://physionet.org/content/face-oral-temp-data/1.0.0/FLIR_groups1and2.csv\n"
            "  https://physionet.org/content/face-oral-temp-data/1.0.0/ICI_groups1and2.csv\n"
        )


def load(path):
    df = pd.read_csv(path, skiprows=2)
    return df


def age_band(age_str):
    if age_str in AGE_LE30:
        return "<=30"
    if age_str in AGE_GT30:
        return ">30"
    return np.nan


def add_derived_columns(df):
    df = df.copy()
    df["T_OR_Max"] = df[[f"T_OR_Max{i}" for i in range(1, 5)]].mean(axis=1, skipna=False)
    df["T_FH_Max"] = df[[f"T_FH_Max{i}" for i in range(1, 5)]].mean(axis=1, skipna=False)
    df["age_band"] = df["Age"].apply(age_band)
    return df


def fit_blind_model(df, feature, log):
    cols = ["T_atm", "Humidity", "Distance", feature, "aveOralM"]
    d = df.dropna(subset=cols).copy()
    X = sm.add_constant(d[["T_atm", "Humidity", "Distance", feature]])
    y = d["aveOralM"]
    model = sm.OLS(y, X).fit()
    d["pred"] = model.predict(X)
    d["resid"] = d["pred"] - y
    rmse = float(np.sqrt(np.mean(d["resid"] ** 2)))
    log.append(
        f"  feature={feature}: n={len(d)}, R2={model.rsquared:.6f}, "
        f"adjR2={model.rsquared_adj:.6f}, RMSE={rmse:.6f}, AIC={model.aic:.3f}"
    )
    return d, model, rmse


def choose_feature(df, device_name, log):
    log.append(f"\n[{device_name}] Feature comparison (T_OR_Max vs T_FH_Max):")
    d_or, m_or, rmse_or = fit_blind_model(df, "T_OR_Max", log)
    d_fh, m_fh, rmse_fh = fit_blind_model(df, "T_FH_Max", log)
    if rmse_or <= rmse_fh:
        log.append(f"  -> CHOSEN: T_OR_Max (lower RMSE)")
        return "T_OR_Max", d_or, m_or
    else:
        log.append(f"  -> CHOSEN: T_FH_Max (lower RMSE)")
        return "T_FH_Max", d_fh, m_fh


def prep_subgroup_frame(d):
    d = d.copy()
    d = d[d["Ethnicity"] != EXCLUDED_ETHNICITY]
    d = d.dropna(subset=["age_band"])
    return d


def cell_counts(d, log, label):
    ct = d.groupby(["Ethnicity", "Gender"]).size()
    log.append(f"\n[{label}] Ethnicity x Sex cell counts:")
    for (eth, sex), n in ct.items():
        log.append(f"    {eth} / {sex}: n={n}")
    small = ct[ct < MIN_CELL_N]
    if len(small):
        log.append(f"  Cells below n={MIN_CELL_N}: {dict(small)}")
    else:
        log.append(f"  No cells below n={MIN_CELL_N}.")
    return ct


def heterogeneity_regression(d, log, label, cluster_col=None, extra_terms=None):
    d = d.copy()
    d["Gender_Male"] = (d["Gender"] == "Male").astype(int)
    eth_dummies = pd.get_dummies(d["Ethnicity"], prefix="Eth", drop_first=False)
    ref_eth = "Asian"
    eth_cols = [c for c in eth_dummies.columns if c != f"Eth_{ref_eth}"]
    d = pd.concat([d, eth_dummies[eth_cols]], axis=1)
    d["age_gt30"] = (d["age_band"] == ">30").astype(int)

    predictors = ["T_atm", "Humidity", "Distance", "Gender_Male"] + eth_cols + ["age_gt30"]
    if extra_terms:
        predictors += extra_terms

    X = sm.add_constant(d[predictors].astype(float))
    y = d["resid"].astype(float)

    if cluster_col is not None:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster_col]}
        )
    else:
        model = sm.OLS(y, X).fit(cov_type="HC3")

    log.append(f"\n[{label}] Adjusted OLS: residual ~ sex + ethnicity + age_band + T_atm + Humidity + Distance"
                + (" + device" if extra_terms else ""))
    log.append(model.summary().as_text())

    ci = model.conf_int()
    table = pd.DataFrame({
        "coef": model.params,
        "CI_low": ci[0],
        "CI_high": ci[1],
        "p": model.pvalues,
    })
    return table, model


def bootstrap_metric(values, func, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boots[i] = func(sample)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def subgroup_metrics(d, group_col, log, label):
    rows = []
    for group, sub in d.groupby(group_col):
        resid = sub["resid"].values
        bias = float(np.mean(resid))
        mae = float(np.mean(np.abs(resid)))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        bias_lo, bias_hi = bootstrap_metric(resid, np.mean)
        mae_lo, mae_hi = bootstrap_metric(resid, lambda x: np.mean(np.abs(x)))
        rmse_lo, rmse_hi = bootstrap_metric(resid, lambda x: np.sqrt(np.mean(x ** 2)))
        rows.append({
            "device": label, "group_var": group_col, "group": group, "n": len(sub),
            "bias": bias, "bias_lo": bias_lo, "bias_hi": bias_hi,
            "mae": mae, "mae_lo": mae_lo, "mae_hi": mae_hi,
            "rmse": rmse, "rmse_lo": rmse_lo, "rmse_hi": rmse_hi,
        })
    return pd.DataFrame(rows)


def bland_altman_plot(d, group_col, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for group, sub in d.groupby(group_col):
        mean_val = (sub["pred"] + sub["aveOralM"]) / 2
        diff = sub["pred"] - sub["aveOralM"]
        ax.scatter(mean_val, diff, alpha=0.4, s=18, label=f"{group} (n={len(sub)})")
    ax.axhline(0, color="steelblue", linewidth=1)
    ax.set_xlabel("Mean(predicted, observed)")
    ax.set_ylabel("Predicted - observed")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def missingness_test(df_full, log, label):
    round_cols = [c for c in df_full.columns if any(c.endswith(str(r)) for r in range(1, 5))
                  and not c.startswith("Unnamed")]
    round_cols = [c for c in round_cols if c not in ("age_band",)]
    df_full = df_full.copy()
    df_full["any_missing"] = df_full[round_cols].isnull().any(axis=1)

    log.append(f"\n[{label}] Missingness: {df_full['any_missing'].sum()} / {len(df_full)} "
                f"subjects with >=1 missing round-level facial value "
                f"({100*df_full['any_missing'].mean():.2f}%)")

    # Sex
    ct_sex = pd.crosstab(df_full["Gender"], df_full["any_missing"])
    chi2, p, dof, _ = scistats.chi2_contingency(ct_sex)
    log.append(f"  Sex vs missingness: chi2={chi2:.4f}, df={dof}, p={p:.6f}")
    if ct_sex.shape == (2, 2):
        odds, fp = scistats.fisher_exact(ct_sex.values)
        log.append(f"  Sex vs missingness (Fisher exact): OR={odds:.4f}, p={fp:.6f}")

    # Ethnicity (excluding tiny AIAN cell per protocol)
    eth_df = df_full[df_full["Ethnicity"] != EXCLUDED_ETHNICITY]
    ct_eth = pd.crosstab(eth_df["Ethnicity"], eth_df["any_missing"])
    chi2e, pe, dofe, expected = scistats.chi2_contingency(ct_eth)
    min_expected = expected.min()
    log.append(f"  Ethnicity vs missingness: chi2={chi2e:.4f}, df={dofe}, p={pe:.6f}, "
                f"min expected cell={min_expected:.4f}"
                + ("  [CAUTION: <5, chi-square approximation may be unreliable]" if min_expected < 5 else ""))

    return ct_sex, ct_eth


def run_device(path, device_name, log):
    log.append(f"\n{'='*70}\n{device_name}\n{'='*70}")
    raw = load(path)
    log.append(f"Raw shape: {raw.shape}")
    df = add_derived_columns(raw)

    feature, d_model, model = choose_feature(df, device_name, log)
    d_sub = prep_subgroup_frame(d_model)
    cell_counts(d_sub, log, device_name)
    reg_table, reg_model = heterogeneity_regression(d_sub, log, device_name)
    metrics_eth = subgroup_metrics(d_sub, "Ethnicity", log, device_name)
    metrics_sex = subgroup_metrics(d_sub, "Gender", log, device_name)
    ba_eth_path = OUT_DIR / f"{device_name}_BlandAltman_Ethnicity.png"
    ba_sex_path = OUT_DIR / f"{device_name}_BlandAltman_Sex.png"
    bland_altman_plot(d_sub, "Ethnicity", f"{device_name} Bland-Altman by Ethnicity", ba_eth_path)
    bland_altman_plot(d_sub, "Gender", f"{device_name} Bland-Altman by Sex", ba_sex_path)

    ct_sex, ct_eth = missingness_test(df, log, device_name)

    return {
        "feature": feature,
        "d_model": d_model,
        "d_sub": d_sub,
        "reg_table": reg_table,
        "metrics_eth": metrics_eth,
        "metrics_sex": metrics_sex,
        "raw_df": raw,
        "full_df": df,
    }


def paired_sensitivity(flir_res, ici_res, log):
    log.append(f"\n{'='*70}\nPAIRED SENSITIVITY ANALYSIS (matched SubjectID + Date)\n{'='*70}")
    fl = flir_res["full_df"][["SubjectID", "Date"]].rename(columns={"Date": "FLIR_Date"})
    ic = ici_res["full_df"][["SubjectID", "Date"]].rename(columns={"Date": "ICI_Date"})
    merged = fl.merge(ic, on="SubjectID", how="inner")
    log.append(f"Overlap SubjectIDs: {len(merged)}")
    log.append(f"Same date: {(merged['FLIR_Date'] == merged['ICI_Date']).sum()}")
    log.append(f"Different date: {(merged['FLIR_Date'] != merged['ICI_Date']).sum()}")

    fl_d = flir_res["d_sub"].copy()
    fl_d["device"] = "FLIR"
    ic_d = ici_res["d_sub"].copy()
    ic_d["device"] = "ICI"

    paired_ids = set(merged["SubjectID"])
    fl_p = fl_d[fl_d["SubjectID"].isin(paired_ids)]
    ic_p = ic_d[ic_d["SubjectID"].isin(paired_ids)]
    combined = pd.concat([fl_p, ic_p], axis=0, ignore_index=True)

    log.append(f"FLIR usable in paired cohort: {len(fl_p)}")
    log.append(f"ICI usable in paired cohort: {len(ic_p)}")
    log.append(f"Combined long-format rows: {len(combined)}; unique subjects: {combined['SubjectID'].nunique()}")

    combined["device_ICI"] = (combined["device"] == "ICI").astype(int)
    cell_counts(combined, log, "Paired")
    reg_table, model = heterogeneity_regression(
        combined, log, "Paired sensitivity", cluster_col="SubjectID",
        extra_terms=["device_ICI"]
    )
    return reg_table, combined


def main():
    check_inputs()
    OUT_DIR.mkdir(exist_ok=True)
    log = []
    log.append("OFFLINE REPRODUCTION: Facial IRT calibration-heterogeneity analysis")
    log.append(f"Random seed: {RANDOM_SEED}  |  Bootstrap resamples: {N_BOOT}")

    flir_res = run_device(FLIR_PATH, "FLIR", log)
    ici_res = run_device(ICI_PATH, "ICI", log)
    paired_table, paired_df = paired_sensitivity(flir_res, ici_res, log)

    # ---------------- write tables ----------------
    demo_rows = []
    for name, res in [("FLIR", flir_res), ("ICI", ici_res)]:
        d = res["d_sub"]
        for col in ["Gender", "Ethnicity", "age_band"]:
            for val, n in d[col].value_counts().items():
                demo_rows.append({"device": name, "variable": col, "value": val,
                                   "n": n, "pct": round(100 * n / len(d), 2)})
    pd.DataFrame(demo_rows).to_csv(OUT_DIR / "table1_demographics.csv", index=False)

    flir_reg = flir_res["reg_table"].copy(); flir_reg["device"] = "FLIR"
    ici_reg = ici_res["reg_table"].copy(); ici_reg["device"] = "ICI"
    paired_reg = paired_table.copy(); paired_reg["device"] = "Paired"
    reg_all = pd.concat([flir_reg, ici_reg, paired_reg], axis=0)
    reg_all.to_csv(OUT_DIR / "table2_regression_coefficients.csv")

    metrics_all = pd.concat([
        flir_res["metrics_eth"], flir_res["metrics_sex"],
        ici_res["metrics_eth"], ici_res["metrics_sex"],
    ], axis=0, ignore_index=True)
    metrics_all.to_csv(OUT_DIR / "table3_subgroup_metrics.csv", index=False)

    flir_res["d_model"].to_csv(OUT_DIR / "residuals_flir.csv", index=False)
    ici_res["d_model"].to_csv(OUT_DIR / "residuals_ici.csv", index=False)

    with open(OUT_DIR / "report.txt", "w") as f:
        f.write("\n".join(log))

    print(f"\nDone. Report and evidence files written to: {OUT_DIR.resolve()}\n")
    print("Files:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
