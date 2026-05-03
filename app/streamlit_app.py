"""Streamlit app for the loan credit risk model.

Run locally::

    streamlit run app/streamlit_app.py

Tabs:
- Single application: form inputs -> probability, decision, score band, SHAP drivers.
- Batch scoring: upload a CSV with the same schema, download scored output.
- Model details: metrics, calibration plot, feature schema, link to model card.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data as data_mod, explain, modeling, scoring  # noqa: E402

# Short tooltips (hover the ⓘ next to each label in the form).
FEATURE_HELP: dict[str, str] = {
    "loanAmount": "Total loan principal requested, in dollars.",
    "interestRate": "Annual interest rate on the loan (APR), as a percentage.",
    "monthlyPayment": "Scheduled monthly payment amount, in dollars.",
    "term": "Repayment term length (e.g. 36 / 48 / 60 months).",
    "grade": "Underwriting risk grade (A1–E3); higher letter / number usually means higher risk.",
    "purpose": "Stated use of the loan (e.g. debt consolidation, home improvement).",
    "isJointApplication": "1 if the application includes a co-borrower, else 0.",
    "annualIncome": "Borrower’s self-reported annual income, in dollars.",
    "dtiRatio": "Debt-to-income ratio: monthly debt payments as a share of income (percentage).",
    "lengthCreditHistory": "Length of credit history on file, in years.",
    "homeOwnership": "Housing status: rent, own, or mortgage.",
    "yearsEmployment": "How long the borrower has been in their current employment band.",
    "residentialState": "Two-letter US state code for the borrower’s residence.",
    "incomeVerified": "1 if income was verified (e.g. with documents), else 0.",
    "numTotalCreditLines": "Total number of credit lines ever on the credit file.",
    "numOpenCreditLines": "Number of currently open credit lines.",
    "numOpenCreditLines1Year": "New credit lines opened in the last 12 months.",
    "revolvingBalance": "Total outstanding balance on revolving accounts (e.g. cards), in dollars.",
    "revolvingUtilizationRate": "Revolving balance divided by limits, as a percentage (credit usage).",
    "numDerogatoryRec": "Count of serious negative items (e.g. collections, public records).",
    "numDelinquency2Years": "Number of delinquencies in the last 24 months.",
    "numChargeoff1year": "Number of charge-offs in the last 12 months.",
    "numInquiries6Mon": "Hard credit inquiries in the last 6 months.",
}

st.set_page_config(
    page_title="Loan credit risk",
    page_icon="ring",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_artifacts() -> dict:
    """Load both calibrated models, score bands, thresholds, metadata."""
    artifacts: dict = {}
    artifacts["lr_model"] = modeling.load_model("lr_model")
    artifacts["lgbm_model"] = modeling.load_model("lgbm_model")
    artifacts["lr_band"] = scoring.load_score_bands(config.DATA_ARTIFACTS / "lr_score_bands.json")
    artifacts["lgbm_band"] = scoring.load_score_bands(config.DATA_ARTIFACTS / "lgbm_score_bands.json")

    thr_path = config.DATA_ARTIFACTS / "thresholds.json"
    if thr_path.is_file():
        artifacts["thresholds"] = json.loads(thr_path.read_text(encoding="utf-8"))
    else:
        artifacts["thresholds"] = {"lr": 0.5, "lgbm": 0.5}

    meta_path = config.DATA_ARTIFACTS / "metadata.json"
    artifacts["metadata"] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    )

    metrics_path = config.OUTPUTS_REPORTS / "metrics.json"
    artifacts["metrics"] = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    )
    return artifacts


@st.cache_data
def load_example_row() -> dict:
    """Pre-fill the form with one real (test-set) record so reviewers can click Predict immediately."""
    df = data_mod.build_modeling_frame().drop(columns=[config.TARGET])
    return df.sample(n=1, random_state=config.RANDOM_STATE).iloc[0].to_dict()


@st.cache_data
def load_categorical_options() -> dict[str, list[str]]:
    df = data_mod.build_modeling_frame()
    return {
        "purpose": sorted(df["purpose"].dropna().unique().tolist()),
        "term": sorted(df["term"].dropna().unique().tolist()),
        "grade": sorted(df["grade"].dropna().unique().tolist()),
        "residentialState": sorted(df["residentialState"].dropna().unique().tolist()),
        "yearsEmployment": [
            "< 1 year",
            "1 year",
            "2-5 years",
            "6-9 years",
            "10+ years",
        ],
        "homeOwnership": sorted(df["homeOwnership"].dropna().unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Sidebar (model + threshold)
# ---------------------------------------------------------------------------


def sidebar_controls(artifacts: dict) -> tuple[str, float]:
    st.sidebar.header("Settings")
    model_label = st.sidebar.radio(
        "Model",
        options=["LightGBM (recommended)", "Logistic regression (baseline)"],
        index=0,
        help="LightGBM is the stronger ranker; logistic regression is a transparent baseline on binned features.",
    )
    model_key = "lgbm" if model_label.startswith("LightGBM") else "lr"
    suggested = float(artifacts["thresholds"].get(model_key, 0.5))
    threshold = st.sidebar.slider(
        "Decision threshold (decline if probability >= threshold)",
        min_value=0.01,
        max_value=0.99,
        value=round(suggested, 3),
        step=0.005,
        help="Predicted probability of default (bad loan). At or above this cutoff we label the application as decline; below is approve. Tune for your risk appetite.",
    )
    st.sidebar.caption(f"KS-optimal threshold for selected model: {suggested:.3f}")
    return model_key, threshold


# ---------------------------------------------------------------------------
# Tab 1: single application
# ---------------------------------------------------------------------------


def render_single_tab(artifacts: dict, model_key: str, threshold: float) -> None:
    st.subheader("Single application")
    st.caption(
        "Default values come from a real record in the held-out test set, "
        "so you can press Predict immediately or override any field."
    )
    with st.expander("How to read the inputs"):
        st.markdown(
            "Hover the **ⓘ** icon next to any field label for a one-line definition. "
            "All fields match the synthetic Microsoft loan + borrower sample used to train the models."
        )
    example = load_example_row()
    options = load_categorical_options()

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Loan**")
            loan_amount = st.number_input(
                "loanAmount ($)",
                value=float(example.get("loanAmount", 15000.0)),
                min_value=0.0,
                step=500.0,
                help=FEATURE_HELP["loanAmount"],
            )
            interest_rate = st.number_input(
                "interestRate (%)",
                value=float(example.get("interestRate", 10.0)),
                min_value=0.0,
                max_value=40.0,
                step=0.1,
                help=FEATURE_HELP["interestRate"],
            )
            monthly_payment = st.number_input(
                "monthlyPayment ($)",
                value=float(example.get("monthlyPayment", 400.0)),
                min_value=0.0,
                step=10.0,
                help=FEATURE_HELP["monthlyPayment"],
            )
            term = st.selectbox(
                "term",
                options=options["term"],
                index=options["term"].index(example.get("term") or options["term"][0]),
                help=FEATURE_HELP["term"],
            )
            grade = st.selectbox(
                "grade",
                options=options["grade"],
                index=options["grade"].index(example.get("grade") or options["grade"][0]),
                help=FEATURE_HELP["grade"],
            )
            purpose = st.selectbox(
                "purpose",
                options=options["purpose"],
                index=options["purpose"].index(example.get("purpose") or options["purpose"][0]),
                help=FEATURE_HELP["purpose"],
            )
            is_joint = st.selectbox(
                "isJointApplication",
                options=[0, 1],
                index=int(example.get("isJointApplication") or 0),
                help=FEATURE_HELP["isJointApplication"],
            )

        with c2:
            st.markdown("**Borrower**")
            annual_income = st.number_input(
                "annualIncome ($)",
                value=float(example.get("annualIncome", 60000.0)),
                min_value=0.0,
                step=1000.0,
                help=FEATURE_HELP["annualIncome"],
            )
            dti_ratio = st.number_input(
                "dtiRatio",
                value=float(example.get("dtiRatio", 20.0)),
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                help=FEATURE_HELP["dtiRatio"],
            )
            length_credit = st.number_input(
                "lengthCreditHistory (years)",
                value=int(example.get("lengthCreditHistory", 10)),
                min_value=0,
                step=1,
                help=FEATURE_HELP["lengthCreditHistory"],
            )
            home_ownership = st.selectbox(
                "homeOwnership",
                options=options["homeOwnership"],
                index=options["homeOwnership"].index(example.get("homeOwnership") or options["homeOwnership"][0]),
                help=FEATURE_HELP["homeOwnership"],
            )
            years_employment = st.selectbox(
                "yearsEmployment",
                options=options["yearsEmployment"],
                index=options["yearsEmployment"].index(example.get("yearsEmployment") or options["yearsEmployment"][0])
                if (example.get("yearsEmployment") in options["yearsEmployment"]) else 0,
                help=FEATURE_HELP["yearsEmployment"],
            )
            residential_state = st.selectbox(
                "residentialState",
                options=options["residentialState"],
                index=options["residentialState"].index(example.get("residentialState") or options["residentialState"][0])
                if (example.get("residentialState") in options["residentialState"]) else 0,
                help=FEATURE_HELP["residentialState"],
            )
            income_verified = st.selectbox(
                "incomeVerified",
                options=[0, 1],
                index=int(example.get("incomeVerified") or 0),
                help=FEATURE_HELP["incomeVerified"],
            )

        with c3:
            st.markdown("**Credit history**")
            num_total = st.number_input(
                "numTotalCreditLines",
                value=int(example.get("numTotalCreditLines", 10)),
                min_value=0,
                step=1,
                help=FEATURE_HELP["numTotalCreditLines"],
            )
            num_open = st.number_input(
                "numOpenCreditLines",
                value=int(example.get("numOpenCreditLines", 7) or 7),
                min_value=0,
                step=1,
                help=FEATURE_HELP["numOpenCreditLines"],
            )
            num_open_1y = st.number_input(
                "numOpenCreditLines1Year",
                value=int(example.get("numOpenCreditLines1Year", 3)),
                min_value=0,
                step=1,
                help=FEATURE_HELP["numOpenCreditLines1Year"],
            )
            revolving_balance = st.number_input(
                "revolvingBalance ($)",
                value=int(example.get("revolvingBalance", 15000)),
                min_value=0,
                step=500,
                help=FEATURE_HELP["revolvingBalance"],
            )
            revolving_util = st.number_input(
                "revolvingUtilizationRate (%)",
                value=float(example.get("revolvingUtilizationRate", 50.0)),
                min_value=0.0,
                max_value=200.0,
                step=0.5,
                help=FEATURE_HELP["revolvingUtilizationRate"],
            )
            num_derog = st.number_input(
                "numDerogatoryRec",
                value=int(example.get("numDerogatoryRec", 0)),
                min_value=0,
                step=1,
                help=FEATURE_HELP["numDerogatoryRec"],
            )
            num_delinq = st.number_input(
                "numDelinquency2Years",
                value=int(example.get("numDelinquency2Years", 0)),
                min_value=0,
                step=1,
                help=FEATURE_HELP["numDelinquency2Years"],
            )
            num_chargeoff = st.number_input(
                "numChargeoff1year",
                value=int(example.get("numChargeoff1year", 0)),
                min_value=0,
                step=1,
                help=FEATURE_HELP["numChargeoff1year"],
            )
            num_inq = st.number_input(
                "numInquiries6Mon",
                value=int(example.get("numInquiries6Mon", 0)),
                min_value=0,
                step=1,
                help=FEATURE_HELP["numInquiries6Mon"],
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    row = {
        "loanAmount": loan_amount,
        "interestRate": interest_rate,
        "monthlyPayment": monthly_payment,
        "annualIncome": annual_income,
        "dtiRatio": dti_ratio,
        "lengthCreditHistory": length_credit,
        "numTotalCreditLines": num_total,
        "numOpenCreditLines": num_open,
        "numOpenCreditLines1Year": num_open_1y,
        "revolvingBalance": revolving_balance,
        "revolvingUtilizationRate": revolving_util,
        "numDerogatoryRec": num_derog,
        "numDelinquency2Years": num_delinq,
        "numChargeoff1year": num_chargeoff,
        "numInquiries6Mon": num_inq,
        "purpose": purpose,
        "term": term,
        "grade": grade,
        "residentialState": residential_state,
        "yearsEmployment": years_employment,
        "homeOwnership": home_ownership,
        "isJointApplication": is_joint,
        "incomeVerified": income_verified,
    }

    model = artifacts[f"{model_key}_model"]
    band = artifacts[f"{model_key}_band"]
    explainer = explain.top_drivers if model_key == "lgbm" else None

    pred = scoring.score_one(
        model,
        row,
        threshold=threshold,
        band=band,
        explainer=explainer,
        n_drivers=6,
    )

    color = "red" if pred.decision == "decline" else "green"
    st.markdown("### Result")
    cols = st.columns(4)
    cols[0].metric("Probability of default", f"{pred.probability:.3%}")
    cols[1].markdown(f"**Decision** :{color}[**{pred.decision.upper()}**]")
    cols[2].metric("Threshold", f"{pred.threshold:.3f}")
    cols[3].metric(
        "Score band bad rate",
        f"{pred.band_bad_rate:.2%}" if pred.band_bad_rate == pred.band_bad_rate else "n/a",
    )

    if pred.drivers:
        st.markdown("### Top drivers (LightGBM, SHAP)")
        drivers_df = pd.DataFrame(pred.drivers)
        drivers_df["abs_shap"] = drivers_df["shap_value"].abs()
        drivers_df = drivers_df.sort_values("abs_shap", ascending=True)
        chart = drivers_df.set_index("feature")["shap_value"]
        st.bar_chart(chart, horizontal=True)
        st.caption(
            "Positive (red on the right) pushes probability toward bad-loan. "
            "Negative (left) pushes toward good-loan."
        )
    elif model_key == "lr":
        st.info("SHAP drivers are shown for LightGBM only (LR is interpretable from coefficients).")


# ---------------------------------------------------------------------------
# Tab 2: batch scoring
# ---------------------------------------------------------------------------


def render_batch_tab(artifacts: dict, model_key: str, threshold: float) -> None:
    st.subheader("Batch scoring")
    st.caption(
        "Upload a CSV with the same columns as the modeling frame. "
        "Output adds `probability`, `decision`, `band_index`, and `band_bad_rate` columns."
    )

    expected = list(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + config.BINARY_FEATURES)
    with st.expander("Expected schema"):
        st.code(", ".join(expected), language="text")

    uploaded = st.file_uploader(
        "CSV file",
        type=["csv"],
        help="Must include the same columns as the single-application form (see Expected schema). One row per loan application.",
    )
    if uploaded is None:
        st.info("Tip: download a sample from `data/processed/Loan.csv` joined with `Borrower.csv`.")
        return

    df = pd.read_csv(uploaded)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    model = artifacts[f"{model_key}_model"]
    band = artifacts[f"{model_key}_band"]
    out = scoring.score_batch(model, df, threshold=threshold, band=band)

    st.success(f"Scored {len(out):,} rows.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Decline rate", f"{(out['decision'] == 'decline').mean():.1%}")
    c2.metric("Mean probability", f"{out['probability'].mean():.3%}")
    c3.metric("p95 probability", f"{out['probability'].quantile(0.95):.3%}")

    st.dataframe(out.head(50), use_container_width=True)
    st.download_button(
        label="Download scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="scored.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Tab 3: model details
# ---------------------------------------------------------------------------


def render_details_tab(artifacts: dict) -> None:
    st.subheader("Model details")

    metadata = artifacts.get("metadata", {})
    metrics = artifacts.get("metrics", {})

    if metadata:
        st.markdown("**Training metadata**")
        small = {
            k: metadata.get(k)
            for k in (
                "created_at",
                "git_sha",
                "random_state",
                "n_train",
                "n_test",
                "train_pos_rate",
                "test_pos_rate",
                "python_version",
                "sklearn_version",
                "lightgbm_version",
            )
        }
        st.json(small)

    if metrics:
        st.markdown("**Held-out metrics**")
        rows = []
        for name, m in metrics.items():
            rows.append(
                {
                    "model": name,
                    "AUC": round(m["auc"], 4),
                    "PR-AUC": round(m["pr_auc"], 4),
                    "KS": round(m["ks"], 4),
                    "Brier": round(m["brier"], 4),
                    "KS threshold": round(m["threshold"], 4),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("**Calibration (LightGBM)**")
        lgbm = metrics.get("lgbm")
        if lgbm:
            cal = pd.DataFrame(
                {
                    "predicted": lgbm["calibration"]["prob_pred"],
                    "observed": lgbm["calibration"]["prob_true"],
                }
            )
            st.line_chart(cal.set_index("predicted"))

        st.markdown("**Lift by decile (LightGBM)**")
        if lgbm:
            lift = pd.DataFrame(
                {
                    "decile": list(range(1, len(lgbm["lift_by_decile"]) + 1)),
                    "lift": lgbm["lift_by_decile"],
                    "bad_rate": lgbm["bad_rate_by_decile"],
                }
            ).set_index("decile")
            st.bar_chart(lift["lift"])

    st.markdown(
        "**Feature schema** lives in [`src/config.py`](https://github.com/buyangdaffa/r-server-loan-credit-risk/blob/master/src/config.py); "
        "the [model card](https://github.com/buyangdaffa/r-server-loan-credit-risk/blob/master/docs/MODEL_CARD.md) "
        "covers data, label definition, and limitations."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Loan credit risk")
    st.caption(
        "Demo for default-risk scoring on the synthetic Microsoft loan dataset. "
        "Logistic regression baseline + calibrated LightGBM, SHAP explanations, score bands."
    )

    artifacts = load_artifacts()
    model_key, threshold = sidebar_controls(artifacts)

    tab_single, tab_batch, tab_details = st.tabs(
        ["Single application", "Batch scoring", "Model details"]
    )

    with tab_single:
        render_single_tab(artifacts, model_key, threshold)
    with tab_batch:
        render_batch_tab(artifacts, model_key, threshold)
    with tab_details:
        render_details_tab(artifacts)


if __name__ == "__main__":
    main()
