import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import base64

st.set_page_config(page_title="Exoplanet Classifier", layout="wide")

# -------------------------------
# Local Background Image (Base64 Method)
# -------------------------------
bg_image_path = "Files/bgimg.jpg"  # ensure this exists

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if os.path.exists(bg_image_path):
    bin_str = get_base64_of_bin_file(bg_image_path)
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
else:
    st.warning("⚠️ Background image not found. Please add 'Files/bgimg.jpg' in your folder.")

# ---------- COMPARISON SECTION ----------
st.header("📊 Comparison with Research Paper - Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification")
st.write("Welcome to the research focused page of your Streamlit app!")

comparison_text = """
### Paper vs Our Work (CatBoost & LightGBM)

| Metric | Paper (Best: Stacking Improved) | LightGBM | CatBoost | Δ vs Paper (LightGBM) | Δ vs Paper (CatBoost) |
|:-------|:-------------------------------:|:----------------:|:----------------:|:----------------------:|:----------------------:|
| **Accuracy** | **83.08%** | **83.64%** | **83.35%** | **+0.56%** | **+0.27%** |
| **F1 Score** | **82.41%** | **84.24%** | **83.91%** | **+1.83%** | **+1.50%** |
| **ROC-AUC** | *Not reported* | **90.50%** | **90.64%** | — | — |

**Interpretation:**  
Both **LightGBM** and **CatBoost** outperform the best model reported in the MDPI paper (*Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification*).  
- **LightGBM** achieves the highest overall accuracy (83.64%) and F1-score (84.24%), showing slightly better balance between precision and recall.  
- **CatBoost** achieves the highest ROC-AUC (0.9064), indicating stronger class separation and ranking performance.  

These improvements likely stem from:
- **Feature engineering:** creation of domain-specific features (ratios, logs, etc.) capturing astrophysical relationships.
- **Robust preprocessing:** outlier clipping and median imputation for stability.
- **Cross-validation:** 5-fold CV ensures a more reliable performance estimate.
- **Modern boosting frameworks:** both LightGBM and CatBoost leverage gradient-boosting innovations (leaf-wise growth and ordered boosting, respectively), improving generalization compared to the ensembles used in the paper.
"""

st.markdown(comparison_text, unsafe_allow_html=True)

st.divider()


# ---------- ABOUT CATBOOST ----------

st.header("CatBoost")

catboost_info = """
CatBoost (**Categorical Boosting**) is a high-performance gradient boosting algorithm 
developed by Yandex. It is specifically optimized to handle **categorical features** 
and to prevent **prediction shift** and **target leakage** during training.

Key innovations that distinguish CatBoost:

1. **Ordered Boosting** — instead of standard gradient boosting, CatBoost uses a permutation-driven, 
   order-aware boosting process that prevents target leakage and produces more reliable estimates 
   for small datasets or when categories are strongly correlated with the label.
2. **Efficient Categorical Encoding** — categorical variables are converted to numerical statistics 
   using permutation-based encoding, allowing CatBoost to natively handle categorical data 
   without the need for one-hot encoding or manual preprocessing.

Practical advantages:
- **Automatic handling of missing values** in both numerical and categorical features.
- **Reduced overfitting** through built-in regularization, ordered boosting, and early stopping.
- **Fast inference** due to optimized tree structures and parallel computation.
- **Native support for GPU acceleration**, making it scalable to large datasets.

In the **KOI exoplanet dataset**, CatBoost achieved a **ROC-AUC of 0.9064** and an **F1-score of 0.8391**, 
performing on par with or slightly better than LightGBM.  
Its ability to integrate domain-specific engineered features and manage mixed data types 
makes CatBoost particularly well-suited for astrophysical classification problems 
such as distinguishing confirmed exoplanets from candidates.
"""

st.markdown(catboost_info)
st.divider()

# ---------- ABOUT LIGHTGBM ----------
st.header("LightGBM")

lightgbm_info = """
LightGBM (**Light Gradient Boosting Machine**) is another high-performance gradient boosting framework 
based on decision trees, developed by Microsoft Research.  
It is optimized for **speed**, **memory efficiency**, and **scalability** on large tabular datasets.

Key innovations that distinguish LightGBM:

1. **Leaf-wise growth strategy** — instead of growing trees level-by-level, LightGBM grows them 
   leaf-by-leaf, choosing the leaf with the highest loss reduction at each step.  
   This allows the model to achieve **lower loss with fewer iterations**, 
   though it may require regularization to prevent overfitting.
2. **Histogram-based algorithm** — continuous features are bucketed into discrete bins, 
   significantly reducing memory usage and computation time.
3. **Support for categorical features** — LightGBM can handle categorical features natively using 
   gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB), 
   which improve efficiency without sacrificing accuracy.

Practical advantages:
- Extremely fast training and prediction times.
- Built-in regularization parameters (`lambda_l1`, `lambda_l2`, `min_data_in_leaf`) to control overfitting.
- Highly effective for imbalanced datasets when combined with class-weighting or boosting type adjustments.

In the **KOI exoplanet dataset**, LightGBM achieved the **highest accuracy (83.64%)** 
and **F1-score (84.24%)** among all compared models.  
Its efficiency and precision make it a robust choice for exoplanet candidate classification, 
especially when rapid experimentation is required.
"""

st.markdown(lightgbm_info)
st.divider()


# ---------- DEMO SECTION ----------
st.header("🚀 Try the Model (Interactive Demo)")
st.title("🔭 Exoplanet Classifier using our trained models")
st.markdown("Select model and provide feature values.")

# EXPLANATIONS = {
#     # Main KOI features
#     "Orbital Period (days)": "How many days the planet takes to go around its star. Like Earth takes 365 days to go around the Sun!",
#     "Orbital Period Error (+)": "How much we might be off when guessing the orbit time.",
#     "Time of Transit (BJD)": "The exact moment when the planet walks in front of its star—like crossing in front of a flashlight.",
#     "Time of Transit Error (+)": "How much our 'crossing time' guess could be off.",
#     "Transit Duration (hrs)": "How many hours the planet is blocking the star. Short peek 👀 or long peek!",
#     "Transit Duration Error (+)": "How much our timing could be off for how long the planet blocks the star.",
#     "Transit Depth (ppm)": "How much dimmer the star looks when the planet blocks it. Big dip = bigger planet.",
#     "Transit Depth Error (+)": "How much our dimness guess could be off.",
#     "Planet Radius (Earth radii)": "How big the planet is compared to Earth. 2 means it's twice as big as Earth 🌍.",
#     "Planet Radius Error (+)": "How much we might be off when guessing the planet’s size (too big?).",
#     "Planet Radius Error (-)": "How much we might be off when guessing the planet’s size (too small?).",
#     "Equilibrium Temperature (K)": "How hot or cold the planet might be if it had no air blanket. 🔥🥶",
#     "Insolation Flux (Earth flux)": "How much starlight the planet gets compared to Earth. More light = toastier planet.",
#     "Insolation Flux Error (+)": "How much our 'sunlight guess' could be off.",
#     "Transit Model SNR": "How clear the planet signal is compared to noise. High SNR = strong signal 💡.",
#     "Stellar Effective Temp (K)": "How hot the star’s surface is. Hotter star = brighter and bluer.",
#     "Stellar Effective Temp Error (+)": "How much we might be off in the star’s hotness (too hot?).",
#     "Stellar Effective Temp Error (-)": "How much we might be off in the star’s hotness (too cold?).",
#     "Surface Gravity (log g)": "How strong the star’s gravity is. Strong pull = you’d feel super heavy!",
#     "Surface Gravity Error (+)": "How much we could be wrong in guessing how strong the star pulls (too strong?).",
#     "Surface Gravity Error (-)": "How much we could be wrong in guessing how strong the star pulls (too weak?).",
#     "Stellar Radius Error (+)": "How much bigger the star might really be than our guess.",
#     "Stellar Radius Error (-)": "How much smaller the star might really be than our guess.",
#     "RA (deg)": "Where the star sits left-right on the sky map. Like longitude for stars.",
#     "Dec (deg)": "Where the star sits up-down on the sky map. Like latitude for stars.",
#     "Kepler Magnitude": "How bright the star looks to the Kepler telescope 👀.",

#     # Extra KOI parameters
#     "Impact Parameter": "How much the planet’s path misses the middle of the star. 0 = right in front, 1 = just skimming the edge.",
#     "Impact Parameter Error (+)": "How much our guess could be too high for that skim factor.",
#     "Impact Parameter Error (-)": "How much our guess could be too low for that skim factor.",

#     # Engineered features (your extra math tricks)
#     "Depth / Stellar Radius": "How big the dip in light is compared to the star’s size. Bigger dip = bigger planet relative to star.",
#     "Planet / Stellar Radius Ratio": "How big the planet is compared to its star. Like saying 'the planet is a marble, the star is a beach ball'.",
#     "Period / Impact Parameter": "A mix of how long the orbit takes and how much the planet misses the center. More mathy combo ⚡.",
#     "Log(1 + Insolation Flux)": "A math trick to shrink sunlight values into a friendlier scale.",
#     "Log(1 + Transit SNR)": "A math trick to shrink signal-to-noise into a tidier number.",
# }


# Mapping: Pretty names -> Model feature names
NAME_MAP = {
    "Orbital Period (days)": "koi_period",
    "Orbital Period Error (+)": "koi_period_err1",
    "Time of Transit (BJD)": "koi_time0bk",
    "Transit Duration (hrs)": "koi_duration",
    "Transit Depth (ppm)": "koi_depth",
    "Planet Radius (Earth radii)": "koi_prad",
    "Equilibrium Temperature (K)": "koi_teq",
    "Insolation Flux (Earth flux)": "koi_insol",
    "Stellar Effective Temp (K)": "koi_steff",
    "Surface Gravity (log g)": "koi_slogg",
    "Stellar Radius Error (+)": "koi_srad_err1",
    "RA (deg)": "ra",
    "Dec (deg)": "dec",
    "Kepler Magnitude": "koi_kepmag",

    # Extra KOI parameters
    "Time of Transit Error (+)": "koi_time0bk_err1",
    "Impact Parameter": "koi_impact",
    "Impact Parameter Error (+)": "koi_impact_err1",
    "Impact Parameter Error (-)": "koi_impact_err2",
    "Transit Duration Error (+)": "koi_duration_err1",
    "Transit Depth Error (+)": "koi_depth_err1",
    "Planet Radius Error (+)": "koi_prad_err1",
    "Planet Radius Error (-)": "koi_prad_err2",
    "Insolation Flux Error (+)": "koi_insol_err1",
    "Transit Model SNR": "koi_model_snr",
    "Stellar Effective Temp Error (+)": "koi_steff_err1",
    "Stellar Effective Temp Error (-)": "koi_steff_err2",
    "Surface Gravity Error (+)": "koi_slogg_err1",
    "Surface Gravity Error (-)": "koi_slogg_err2",
    "Stellar Radius Error (-)": "koi_srad_err2",

    # Engineered features:
    "Depth / Stellar Radius": "depth_to_srad",
    "Planet / Stellar Radius Ratio": "prad_to_srad_ratio",
    "Period / Impact Parameter": "period_to_impact",
    "Log(1 + Insolation Flux)": "log_insol",
    "Log(1 + Transit SNR)": "log_snr",
}

# Define feature ranges (UI side, pretty names)
FEATURE_RANGES = {
    "Orbital Period (days)": (-0.389486, 3.884597),
    "Orbital Period Error (+)": (-0.249980, 3.912221),
    "Time of Transit (BJD)": (-0.509194, 3.841147),
    "Time of Transit Error (+)": (-0.651946, 3.741984),
    "Impact Parameter": (-0.630537, 3.631575),
    "Impact Parameter Error (+)": (-0.666667, 3.568627),
    "Impact Parameter Error (-)": (-3.567234, 0.7243362),
    "Transit Duration (hrs)": (-1.031911, 3.611988),
    "Transit Duration Error (+)": (-0.692735, 3.733467),
    "Transit Depth (ppm)": (-0.572557, 3.788191),
    "Transit Depth Error (+)": (-0.723214, 3.843750),
    "Planet Radius (Earth radii)": (-1.142857, 3.691814),
    "Planet Radius Error (+)": (-0.723404, 3.553191),
    "Planet Radius Error (-)": (-3.689655, 0.7241379),
    "Equilibrium Temperature (K)": (-1.433444, 3.570820),
    "Insolation Flux (Earth flux)": (-0.312844, 3.735998),
    "Insolation Flux Error (+)": (-0.261479, 3.751848),
    "Transit Model SNR": (-0.675000, 3.850000),
    "Stellar Effective Temp (K)": (-3.652231, 3.522310),
    "Stellar Effective Temp Error (+)": (-1.500000, 3.550000),
    "Stellar Effective Temp Error (-)": (-3.494382, 1.483146),
    "Surface Gravity (log g)": (-3.687500, 3.363971),
    "Surface Gravity Error (+)": (-0.693333, 3.658667),
    "Surface Gravity Error (-)": (-3.685185, 1.037037),
    "Stellar Radius Error (+)": (-0.941748, 3.543689),
    "Stellar Radius Error (-)": (-3.696721, 0.8032787),
    "RA (deg)": (-1.479197, 1.324565),
    "Dec (deg)": (-1.327027, 1.416869),
    "Kepler Magnitude": (-3.649838, 1.684677),

    # Engineered features:
    "Depth / Stellar Radius": (-0.479075, 30.32855),
    "Planet / Stellar Radius Ratio": (-1.007994, 13.79850),
    "Period / Impact Parameter": (-0.239422, 8.018737e9),
    "Log(1 + Insolation Flux)": (-1.665119, 0.9453637),
    "Log(1 + Transit SNR)": (-1.799029, 1.569439),
}

 
# Initialize session state for feature values 
if "feature_values" not in st.session_state: 
    st.session_state.feature_values = { 
        k: float((v[0] + v[1]) / 2.0) for k, v in FEATURE_RANGES.items() 
    } 
 
# Utility functions 
def random_value_between(key): 
    lo, hi = FEATURE_RANGES[key] 
    return float(np.random.uniform(lo, hi)) 
 
def assign_random(feature): 
    st.session_state.feature_values[feature] = random_value_between(feature) 
 
def assign_random_all(): 
    for k in FEATURE_RANGES.keys(): 
        st.session_state.feature_values[k] = random_value_between(k) 
 
# Dropdown to select the model
model_choice = st.selectbox("🧠 Select Model", ["CatBoost", "LightGBM"])

# Load selected model

model = None

model_paths = {
    "CatBoost": "catboost.pkl",
    "LightGBM": "lightgbm.pkl"
}

selected_model_path = model_paths.get(model_choice)

if os.path.exists(selected_model_path):
    model = joblib.load(selected_model_path)
    st.success(f"✅ Loaded {model_choice} model successfully.")
else:
    st.error(f"❌ Could not find `{selected_model_path}`. Please make sure the file exists.")
    st.write("📂 Current working directory:", os.getcwd())
 
# Input grid 
st.header("🛠️ Feature Inputs") 
 
# make 2 columns (so features split nicely) 
cols = st.columns(2) 
 
for i, (feat, (lo, hi)) in enumerate(FEATURE_RANGES.items()): 
    col = cols[i % 2]  # Alternate between columns
    
    val = col.number_input( 
        label=feat, 
        value=float(st.session_state.feature_values.get(feat, (lo + hi) / 2.0)), 
        format="%.6g", 
        key=f"input_{feat}", 
    ) 
 
    # Save immediately 
    st.session_state.feature_values[feat] = float(val) 
 
    # Validation with clear range message
    if val < lo or val > hi: 
        col.error(f"⚠️ **{feat}** is out of valid range: [{lo:.6g}, {hi:.6g}]")
 
 
# Prepare DataFrame for model 
X_dict = {} 
for pretty_name, value in st.session_state.feature_values.items(): 
    raw_name = NAME_MAP.get(pretty_name, pretty_name)  # fall back to same if not found 
    X_dict[raw_name] = [float(value)] 
 
X_df = pd.DataFrame.from_dict(X_dict) 
 
st.subheader("📊 Input Preview (Model Input)") 
st.dataframe(X_df, use_container_width=True) 
 
# Predict

if model is not None and st.button("🚀 Predict"):
    try:
        # Align features if needed
        if hasattr(model, "feature_names_"):
            expected_features = model.feature_names_
            X_df = X_df.reindex(columns=expected_features)
        elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_name"):
            expected_features = model.get_booster().feature_name()
            X_df = X_df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        preds = model.predict(X_df)

        # Get probability (if available)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_df)

            # Find probability of predicted class
            if hasattr(model, "classes_"):
                class_idx = list(model.classes_).index(preds[0])
                pred_prob = float(probs[0, class_idx])
            else:
                pred_prob = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])

            # Display as percentage
            st.success(f"✅ Prediction: {preds[0]} ({pred_prob * 100:.2f}%)")
        else:
            st.success(f"✅ Prediction: {preds[0]} (probability unavailable)")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")


