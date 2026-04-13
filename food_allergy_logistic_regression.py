import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MPL_DIR = OUTPUT_DIR / "mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def load_child_data(csv_path: str | Path = "child24.csv") -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / csv_path)
    variables = [
        "AGEP_C",
        "CURFOOD_C",
        "ANXFREQ_C",
        "DEPFREQ_C",
        "SCHDYMSSTC_C",
        "FDSCAT4_C",
        "MAXPAREDUP_C",
        "HICOV_C",
        "FSNAP12M_C",
    ]
    return df[variables].copy()


def recode_parent_education(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out[s.between(0, 4)] = "Low"
    out[s.between(5, 7)] = "Medium"
    out[s.between(8, 10)] = "High"
    return out


def prepare_analysis_data(child_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = child_data.copy()

    age = pd.to_numeric(df["AGEP_C"], errors="coerce")
    df = df[age.between(5, 17)].copy()

    curfood = pd.to_numeric(df["CURFOOD_C"], errors="coerce")
    df["food_allergy"] = np.where(curfood == 1, 1, np.where(curfood == 2, 0, np.nan))

    anxiety = pd.to_numeric(df["ANXFREQ_C"], errors="coerce")
    df["high_anxiety"] = np.where(
        anxiety.isin([1, 2]),
        1,
        np.where(anxiety.isin([3, 4, 5]), 0, np.nan),
    )

    depression = pd.to_numeric(df["DEPFREQ_C"], errors="coerce")
    df["high_depression"] = np.where(
        depression.isin([1, 2]),
        1,
        np.where(depression.isin([3, 4, 5]), 0, np.nan),
    )

    school_days = pd.to_numeric(df["SCHDYMSSTC_C"], errors="coerce")
    df["school_absence_any"] = np.where(
        school_days.eq(0),
        0,
        np.where(school_days.between(1, 95), 1, np.nan),
    )

    food_security = pd.to_numeric(df["FDSCAT4_C"], errors="coerce")
    df["food_insecure"] = np.where(
        food_security.isin([3, 4]),
        1,
        np.where(food_security.isin([1, 2]), 0, np.nan),
    )

    df["parent_education"] = recode_parent_education(df["MAXPAREDUP_C"])

    insurance = pd.to_numeric(df["HICOV_C"], errors="coerce")
    df["insured"] = np.where(insurance == 1, 1, np.where(insurance == 2, 0, np.nan))

    snap = pd.to_numeric(df["FSNAP12M_C"], errors="coerce")
    df["snap_receive"] = np.where(snap == 1, 1, np.where(snap == 2, 0, np.nan))

    analysis_columns = [
        "food_allergy",
        "high_anxiety",
        "high_depression",
        "school_absence_any",
        "food_insecure",
        "parent_education",
        "insured",
        "snap_receive",
    ]

    analysis_df = df[analysis_columns].dropna().copy()
    analysis_df["food_allergy"] = analysis_df["food_allergy"].astype(int)
    analysis_df["high_anxiety"] = analysis_df["high_anxiety"].astype(int)
    analysis_df["high_depression"] = analysis_df["high_depression"].astype(int)
    analysis_df["school_absence_any"] = analysis_df["school_absence_any"].astype(int)
    analysis_df["food_insecure"] = analysis_df["food_insecure"].astype(int)
    analysis_df["insured"] = analysis_df["insured"].astype(int)
    analysis_df["snap_receive"] = analysis_df["snap_receive"].astype(int)
    analysis_df["parent_education"] = pd.Categorical(
        analysis_df["parent_education"],
        categories=["High", "Medium", "Low"],
        ordered=True,
    )

    predictors = pd.get_dummies(
        analysis_df[
            [
                "high_anxiety",
                "high_depression",
                "school_absence_any",
                "food_insecure",
                "parent_education",
                "insured",
                "snap_receive",
            ]
        ],
        columns=["parent_education"],
        drop_first=True,
        dtype=float,
    )
    predictors = sm.add_constant(predictors, has_constant="add")
    outcome = analysis_df["food_allergy"]

    return analysis_df, predictors, outcome


def fit_logit_model(X: pd.DataFrame, y: pd.Series):
    model = sm.Logit(y, X)
    return model.fit(disp=False, maxiter=200)


def build_results_table(result) -> pd.DataFrame:
    conf = result.conf_int()
    summary_df = pd.DataFrame(
        {
            "variable": result.params.index,
            "beta": result.params.values,
            "odds_ratio": np.exp(result.params.values),
            "ci_lower": np.exp(conf[0].values),
            "ci_upper": np.exp(conf[1].values),
            "p_value": result.pvalues.values,
        }
    )
    summary_df = summary_df[summary_df["variable"] != "const"].copy()
    summary_df["significant"] = summary_df["p_value"] < 0.05
    summary_df["abs_log_or"] = np.abs(np.log(summary_df["odds_ratio"]))
    summary_df = summary_df.sort_values(
        by=["abs_log_or", "p_value"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return summary_df


def prettify_variable_name(name: str) -> str:
    mapping = {
        "high_anxiety": "High Anxiety",
        "high_depression": "High Depression",
        "school_absence_any": "Any School Absence",
        "food_insecure": "Food Insecure",
        "insured": "Insured",
        "snap_receive": "SNAP Recipient",
        "parent_education_Low": "Parent Education: Low vs High",
        "parent_education_Medium": "Parent Education: Medium vs High",
    }
    return mapping.get(name, name.replace("_", " ").title())


def create_forest_plot(results_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = results_df.iloc[::-1].copy()
    plot_df["label"] = plot_df["variable"].map(prettify_variable_name)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(plot_df))
    lower_err = plot_df["odds_ratio"] - plot_df["ci_lower"]
    upper_err = plot_df["ci_upper"] - plot_df["odds_ratio"]
    colors = np.where(plot_df["significant"], "#D55E00", "#7F7F7F")

    for i, row in enumerate(plot_df.itertuples(index=False)):
        color = "#D55E00" if row.significant else "#7F7F7F"
        ax.errorbar(
            row.odds_ratio,
            y_pos[i],
            xerr=[[lower_err.iloc[i]], [upper_err.iloc[i]]],
            fmt="o",
            color="black",
            ecolor=color,
            elinewidth=2,
            capsize=4,
            markersize=7,
        )
        ax.scatter(row.odds_ratio, y_pos[i], color=color, s=70, zorder=3)
    ax.axvline(1.0, color="#1F77B4", linestyle="--", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_ylabel("")
    ax.set_title("Food Allergy Logistic Regression: Forest Plot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_bar_chart(results_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = results_df.copy()
    plot_df["label"] = plot_df["variable"].map(prettify_variable_name)
    plot_df["color"] = np.where(plot_df["significant"], "#E45756", "#B0B0B0")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df["label"], plot_df["odds_ratio"], color=plot_df["color"])
    ax.axhline(1.0, color="#1F77B4", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Odds Ratio")
    ax.set_xlabel("")
    ax.set_title("Food Allergy Logistic Regression: Odds Ratios")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_interpretation(results_df: pd.DataFrame) -> list[str]:
    statements = []
    for row in results_df.itertuples(index=False):
        label = prettify_variable_name(row.variable)
        if row.odds_ratio >= 1:
            direction = "higher"
            effect = row.odds_ratio
        else:
            direction = "lower"
            effect = 1 / row.odds_ratio

        significance = "significantly associated with" if row.significant else "not significantly associated with"
        statement = (
            f"{label} is {significance} food allergy "
            f"(OR {row.odds_ratio:.2f}, 95% CI {row.ci_lower:.2f}-{row.ci_upper:.2f}, p={row.p_value:.3g}). "
            f"This corresponds to {effect:.2f} times {direction} odds of food allergy."
        )
        statements.append(statement)
    return statements


def main() -> None:
    child_data = load_child_data()
    analysis_df, X, y = prepare_analysis_data(child_data)
    result = fit_logit_model(X, y)
    results_df = build_results_table(result)
    results_df["display_name"] = results_df["variable"].map(prettify_variable_name)

    csv_path = OUTPUT_DIR / "food_allergy_logit_results.csv"
    forest_path = OUTPUT_DIR / "food_allergy_forest_plot.png"
    bar_path = OUTPUT_DIR / "food_allergy_odds_ratio_bar_chart.png"

    results_df.to_csv(csv_path, index=False)
    create_forest_plot(results_df, forest_path)
    create_bar_chart(results_df, bar_path)

    significant_df = results_df[results_df["significant"]].copy()
    top_df = results_df.head(5).copy()
    interpretations = generate_interpretation(results_df)

    print("\nFood Allergy Logistic Regression")
    print("-" * 80)
    print(f"Analysis sample size: {len(analysis_df):,}")
    print(f"Number with food allergy: {int(y.sum()):,}")
    print(f"Number without food allergy: {int((1 - y).sum()):,}")
    print("\nModel coefficients and odds ratios:")
    print(
        results_df[
            ["display_name", "beta", "odds_ratio", "ci_lower", "ci_upper", "p_value", "significant"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nTop variables by effect size (|log(OR)|):")
    print(
        top_df[["display_name", "odds_ratio", "ci_lower", "ci_upper", "p_value"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nStatistically significant variables (p < 0.05):")
    if significant_df.empty:
        print("None")
    else:
        print(
            significant_df[
                ["display_name", "odds_ratio", "ci_lower", "ci_upper", "p_value"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\nInterpretation:")
    for sentence in interpretations:
        print(f"- {sentence}")

    print("\nSaved outputs:")
    print(f"- Results table: {csv_path}")
    print(f"- Forest plot: {forest_path}")
    print(f"- Bar chart: {bar_path}")


if __name__ == "__main__":
    main()
