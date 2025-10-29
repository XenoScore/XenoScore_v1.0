import io
import yaml
import pandas as pd
import streamlit as st
from xenoscore.config import load_component_config, load_weights_config
from xenoscore.data.validation import validate_dataframe
from xenoscore.scoring.weighted import WeightedScoreEngine
from xenoscore.scoring.model import ModelScoreEngine
from xenoscore.ml.featurize import featurize

st.set_page_config(page_title="XenoScore", layout="wide")
st.title("🧪 XenoScore — Xenotransplantation Outcome Scoring (Prototype)")

st.sidebar.header("Configuration")
use_model = st.sidebar.checkbox("Use trained ML model (joblib)", value=False)

comp_yaml_file = st.sidebar.file_uploader("Upload component YAML", type=["yaml","yml"])
weights_yaml_file = None
model_file = None

if use_model:
    model_file = st.sidebar.file_uploader("Upload trained model (.joblib)", type=["joblib"])
else:
    weights_yaml_file = st.sidebar.file_uploader("Upload weights YAML", type=["yaml","yml"])

# Example files for download
with st.expander("📥 Download example files"):
    st.download_button("Download example dataset (CSV)", data=open("examples/example_dataset.csv","rb").read(),
                       file_name="example_dataset.csv")
    st.download_button("Download default component config (YAML)", data=open("configs/default_components.yaml","rb").read(),
                       file_name="default_components.yaml")
    st.download_button("Download example weights (YAML)", data=open("configs/weights.example.yaml","rb").read(),
                       file_name="weights.example.yaml")

data_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if data_file and comp_yaml_file and ((use_model and model_file) or (not use_model and weights_yaml_file)):
    df = pd.read_csv(data_file)
    comp_cfg = yaml.safe_load(comp_yaml_file.read())["components"]
    df_valid, errors = validate_dataframe(df)
    if errors:
        st.warning(f"Validation warnings for {len(errors)} rows. Showing first 5.")
        for idx, err in errors[:5]:
            st.code(f"Row {idx}: {err}")

    if use_model:
        st.info("Using ML model to predict probability of adverse outcome.")
        eng = ModelScoreEngine(comp_cfg, model_file)
        preds = eng.predict_proba(df_valid)
    else:
        weights = yaml.safe_load(weights_yaml_file.read())
        eng = WeightedScoreEngine(comp_cfg, weights.get("weights", {}))
        preds = eng.score_dataframe(df_valid)

    out = pd.concat([df_valid, preds], axis=1)
    st.dataframe(out.head(50))
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download predictions CSV", data=csv, file_name="xenoscore_predictions.csv")
else:
    st.info("Upload dataset + configs in the sidebar to score.")

st.markdown("---")
st.caption("Research prototype. Not for clinical use.")
