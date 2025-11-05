import gradio as gr
from risk_screener import calculate_risk_score
import pandas as pd
import joblib
import numpy as np
import logging
import uuid
from typing import Dict, Any, List, Tuple
import boto3
import os
from io import BytesIO

# -------------------------
# S3 Configuration
# -------------------------
S3_BUCKET = "clinovia.ai"
S3_PREFIX = "models/"
LOCAL_MODEL_DIR = "/tmp/models"  # Temporary storage on EC2

# Initialize S3 client
s3_client = boto3.client('s3')

def download_model_from_s3(s3_key: str, local_path: str):
    """Download a model file from S3 to local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        print(f"Downloading {s3_key} from S3...")
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        print(f"✅ Downloaded {s3_key}")
    except Exception as e:
        print(f"❌ Error downloading {s3_key}: {e}")
        raise

def load_model_from_s3(model_name: str):
    """Load a model directly from S3 into memory."""
    s3_key = f"{S3_PREFIX}{model_name}"
    try:
        print(f"Loading {model_name} from S3...")
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        model = joblib.load(BytesIO(obj['Body'].read()))
        print(f"✅ Loaded {model_name}")
        return model
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        raise

# -------------------------
# Load Models & Preprocessors
# -------------------------
CLASS_NAMES = ["CN", "MCI", "AD"]

print("🔄 Loading models from S3...")

# Primary classifier (RandomForest pipeline saved with preprocessor inside)
classifier_1_model = load_model_from_s3("alz_model.joblib")

# Basic classifier model + scaler (saved separately)
classifier_basic_model = load_model_from_s3("classifier_basic_model.pkl")
classifier_basic_preprocessor = load_model_from_s3("classifier_scaler_basic.pkl")

# Advanced classifier + scaler
classifier_advanced_model = load_model_from_s3("classifier_advanced_model.pkl")
classifier_advanced_preprocessor = load_model_from_s3("classifier_scaler_advanced.pkl")

# Progression models
progress_basic_model = load_model_from_s3("progress_basic_RandomForest_best_model.pkl")
progress_basic_scaler = load_model_from_s3("progress_basic_scaler.pkl")

progress_advanced_model = load_model_from_s3("progress_advanced_RandomForest_best_model.pkl")
progress_advanced_scaler = load_model_from_s3("progress_advanced_scaler.pkl")

# Load saved feature orders (ensures inference uses exact training column order)
try:
    BASIC_FEATURE_ORDER: List[str] = load_model_from_s3("basic_features.pkl")
except Exception:
    # fallback consistent ordering if file missing
    BASIC_FEATURE_ORDER = [
        "AGE", "MMSE_bl", "CDRSB_bl", "FAQ_bl", "PTEDUCAT",
        "PTGENDER", "APOE4", "RAVLT_immediate_bl", "MOCA_bl", "ADAS13_bl"
    ]

try:
    ADVANCED_FEATURE_ORDER: List[str] = load_model_from_s3("advanced_features.pkl")
except Exception:
    ADVANCED_FEATURE_ORDER = BASIC_FEATURE_ORDER + [
        "Hippocampus_bl", "Ventricles_bl", "WholeBrain_bl", "Entorhinal_bl",
        "FDG_bl", "AV45_bl", "PIB_bl", "FBB_bl", "ABETA_bl", "TAU_bl",
        "PTAU_bl", "mPACCdigit_bl", "mPACCtrailsB_bl"
    ]

# First classifier features (the simpler tests-only classifier that uses pipeline model)
FIRST_CLASSIFIER_FEATURES = [
    "age", "education_years", "moca_score",
    "adas13_score", "cdr_sum", "faq_total", "gender", "race"
]

# -----------------------------
# 📘 Load Progression Feature Metadata
# -----------------------------

# Load progression basic features metadata
try:
    PROGRESS_BASIC_FEATURES = load_model_from_s3("progress_basic_features.pkl")
    PROGRESS_BASIC_NUMERIC_COLS = load_model_from_s3("progress_basic_numeric_cols.pkl")
    PROGRESS_BASIC_CATEGORICAL_COLS = load_model_from_s3("progress_basic_categorical_cols.pkl")
    print(f"✅ Loaded progression basic metadata: {len(PROGRESS_BASIC_FEATURES)} features")
except Exception as e:
    print(f"⚠️ Could not load progression basic metadata: {e}")
    # Fallback values
    PROGRESS_BASIC_FEATURES = [
        "AGE", "PTGENDER", "PTEDUCAT", "ADAS13", "MOCA", 
        "CDRSB", "FAQ", "APOE4_count", "GDTOTAL"
    ]
    PROGRESS_BASIC_NUMERIC_COLS = [c for c in PROGRESS_BASIC_FEATURES if c != "PTGENDER"]
    PROGRESS_BASIC_CATEGORICAL_COLS = ["PTGENDER"]

# Load progression advanced features metadata
try:
    PROGRESS_ADVANCED_FEATURES = load_model_from_s3("progress_advanced_features.pkl")
    PROGRESS_ADVANCED_NUMERIC_COLS = load_model_from_s3("progress_advanced_numeric_cols.pkl")
    PROGRESS_ADVANCED_CATEGORICAL_COLS = load_model_from_s3("progress_advanced_categorical_cols.pkl")
    print(f"✅ Loaded progression advanced metadata: {len(PROGRESS_ADVANCED_FEATURES)} features")
except Exception as e:
    print(f"⚠️ Could not load progression advanced metadata: {e}")
    # Fallback values
    PROGRESS_ADVANCED_FEATURES = [
        "AGE", "PTGENDER", "PTEDUCAT", "ADAS13", "MOCA", "CDRSB", "FAQ",
        "APOE4_count", "GDTOTAL", "ABETA", "TAU", "PTAU", 
        "FDG", "PIB", "AV45", "FBB"
    ]
    PROGRESS_ADVANCED_NUMERIC_COLS = [c for c in PROGRESS_ADVANCED_FEATURES if c != "PTGENDER"]
    PROGRESS_ADVANCED_CATEGORICAL_COLS = ["PTGENDER"]

print("✅ All models loaded successfully!")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(filename="usage.log", level=logging.INFO, format="%(asctime)s | %(message)s")

def log_usage(tool_name: str, inputs: Any, outputs: Any = None):
    session_id = str(uuid.uuid4())[:8]
    logging.info(f"session={session_id} tool={tool_name} inputs={inputs} outputs={outputs}")

# -----------------------------
# Helpers
# -----------------------------
def encode_gender(value: Any) -> int:
    """Map gender to training encoding: Male -> 1, Female -> 0. Accept strings or numeric."""
    if isinstance(value, str):
        v = value.strip().lower()
        return 1 if v in ("male", "m") else 0
    try:
        return int(value)
    except Exception:
        return 0

def build_df_from_order(values: List[Any], columns: List[str]) -> pd.DataFrame:
    """Build DataFrame in exact column order. values should match len(columns)."""
    if len(values) != len(columns):
        raise ValueError(f"Expected {len(columns)} values but got {len(values)}")
    return pd.DataFrame([values], columns=columns)

def preprocess_for_prediction(input_dict: dict, 
                              feature_order: List[str],
                              numeric_cols: List[str],
                              categorical_cols: List[str],
                              scaler) -> np.ndarray:
    """
    Preprocess input data for model prediction.
    
    Args:
        input_dict: Dictionary mapping feature names to values
        feature_order: Expected order of features (as used in training)
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        scaler: Fitted StandardScaler object
    
    Returns:
        Preprocessed feature array ready for model.predict()
    """
    # Build DataFrame with exact column order
    values = [input_dict[col] for col in feature_order]
    X_df = pd.DataFrame([values], columns=feature_order)
    
    # Extract numeric features
    X_numeric = X_df[numeric_cols]
    
    # Scale numeric features
    X_numeric_scaled = scaler.transform(X_numeric)
    
    # Convert back to DataFrame
    X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_cols)
    
    # Add back categorical features (unscaled)
    for col in categorical_cols:
        X_numeric_scaled_df[col] = X_df[col].values
    
    # Reorder to match training order
    X_final = X_numeric_scaled_df[feature_order]
    
    return X_final.values

# -----------------------------
# Core Functions
# -----------------------------
def run_risk_screener(age, gender, education_years, apoe4_status, memory_score, hippocampal_volume, patient_id=None):
    try:
        inputs = {
            "age": age,
            "gender": gender,
            "education_years": education_years,
            "apoe4_status": apoe4_status,
            "memory_score": memory_score,
            "hippocampal_volume": hippocampal_volume,
        }

        risk_result = calculate_risk_score(inputs)
        log_usage("risk_screener", inputs, {"risk_result": risk_result})

        score = risk_result.get("risk_score", None)
        category = risk_result.get("risk_category", "Unknown")
        rec = risk_result.get("recommendation", "")

        if score is not None:
            return f"🧠 Estimated Alzheimer risk: {score:.2f}% ({category})\n💡 Recommendation: {rec}"
        else:
            return "⚠️ Unable to compute risk score."

    except Exception as e:
        log_usage("risk_screener_error", {}, {"error": str(e)})
        return f"⚠️ Error: {e}"

def run_first_classifier(age, gender, education, adas13, moca, cdrsb, faq, race):
    """Use pipeline model (includes preprocessing internally)."""
    try:
        # normalize input names to training schema
        gender_enc = gender  # pipeline will expect whatever was used in training; training used 'male'/'female' lowercase
        if isinstance(gender, str):
            gender_enc = gender.lower()

        df = pd.DataFrame(
            [[age, education, moca, adas13, cdrsb, faq, gender_enc, race]],
            columns=FIRST_CLASSIFIER_FEATURES
        )

        # classifier_1_model is a Pipeline: pass raw df, pipeline will preprocess
        probs = classifier_1_model.predict_proba(df)[0]
        idx = int(np.argmax(probs))

        result = (
            f"🧠 Predicted: {CLASS_NAMES[idx]} ({probs[idx]:.2f} confidence)\n"
            f"📊 Probabilities: {dict(zip(CLASS_NAMES, map(lambda p: round(float(p), 3), probs)))}"
        )

        log_usage("first_classifier", df.to_dict(orient='records')[0], result)
        return result

    except Exception as e:
        log_usage("first_classifier_error", {}, {"error": str(e)})
        return f"⚠️ Error: {e}"

def run_classifier_basic(age, mmse, cdrsb, faq, education, gender, apoe4, ravlt, moca, adas13):
    """
    Basic classifier expects the exact feature order used during preprocessing.
    """
    try:
        input_vals = [
            age,
            mmse,
            cdrsb,
            faq,
            education,
            encode_gender(gender),
            apoe4 if apoe4 is not None else -1,
            ravlt,
            moca,
            adas13
        ]

        # Build DataFrame with exact column ordering
        X_df = build_df_from_order(input_vals, BASIC_FEATURE_ORDER)

        # Identify numeric columns that scaler was trained on (training excluded PTGENDER and APOE4)
        numeric_cols = [c for c in BASIC_FEATURE_ORDER if c not in ("PTGENDER", "APOE4")]

        # Scale numeric columns
        X_num = X_df[numeric_cols]
        X_num_scaled = classifier_basic_preprocessor.transform(X_num)

        # Extract categorical columns in expected order
        X_cat = X_df[["PTGENDER", "APOE4"]].to_numpy()

        # Recombine scaled numeric + categorical columns in the original training order
        X_scaled = np.hstack([X_num_scaled, X_cat])

        probs = classifier_basic_model.predict_proba(X_scaled)[0]
        idx = int(np.argmax(probs))
        result = (
            f"🧠 Predicted: {CLASS_NAMES[idx]} ({probs[idx]:.2f} confidence)\n"
            f"📊 Probabilities: {dict(zip(CLASS_NAMES, map(lambda p: round(float(p), 3), probs)))}"
        )

        log_usage("classifier_basic", {"features": dict(zip(BASIC_FEATURE_ORDER, input_vals))}, result)
        return result

    except Exception as e:
        log_usage("classifier_basic_error", {}, {"error": str(e)})
        return f"⚠️ Error: {e}"

def run_classifier_advanced(
    age, mmse, cdrsb, faq, education, gender, apoe4,
    ravlt, moca, adas13,
    Hippocampus_bl, Ventricles_bl, WholeBrain_bl, Entorhinal_bl,
    FDG_bl, AV45_bl, PIB_bl, FBB_bl,
    ABETA_bl, TAU_bl, PTAU_bl, mPACCdigit_bl, mPACCtrailsB_bl
):
    try:
        # --- Construct DataFrame ---
        columns = [
            "AGE", "MMSE_bl", "CDRSB_bl", "FAQ_bl", "PTEDUCAT", "PTGENDER",
            "APOE4", "RAVLT_immediate_bl", "MOCA_bl", "ADAS13_bl",
            "Hippocampus_bl", "Ventricles_bl", "WholeBrain_bl", "Entorhinal_bl",
            "FDG_bl", "AV45_bl", "PIB_bl", "FBB_bl",
            "ABETA_bl", "TAU_bl", "PTAU_bl", "mPACCdigit_bl", "mPACCtrailsB_bl"
        ]

        # --- Gender encoding ---
        gender_numeric = 1 if gender == "Male" else 0

        # --- Input array in correct order ---
        X = np.array([[
            age, mmse, cdrsb, faq, education, gender_numeric,
            apoe4, ravlt, moca, adas13,
            Hippocampus_bl, Ventricles_bl, WholeBrain_bl, Entorhinal_bl,
            FDG_bl, AV45_bl, PIB_bl, FBB_bl,
            ABETA_bl, TAU_bl, PTAU_bl, mPACCdigit_bl, mPACCtrailsB_bl
        ]])

        X_df = pd.DataFrame(X, columns=columns)

        # --- Separate numeric vs categorical ---
        num_cols = [c for c in X_df.columns if c not in ["PTGENDER", "APOE4"]]
        X_num = X_df[num_cols]
        X_cat = X_df[["PTGENDER", "APOE4"]]

        # --- Apply scaler only to numeric features ---
        X_num_scaled = classifier_advanced_preprocessor.transform(X_num)

        # --- Combine scaled numeric + categorical ---
        X_scaled = np.hstack([X_num_scaled, X_cat.values])

        # --- Prediction ---
        probs = classifier_advanced_model.predict_proba(X_scaled)[0]
        idx = np.argmax(probs)

        result = (
            f"🧠 Predicted: {CLASS_NAMES[idx]} ({probs[idx]:.2f} confidence)\n"
            f"📊 Probabilities: {dict(zip(CLASS_NAMES, map(float, probs)))}"
        )

        log_usage("classifier_advanced", {"features": X_df.to_dict(orient='records')[0]}, result)
        return result

    except Exception as e:
        log_usage("classifier_advanced_error", {}, {"error": str(e)})
        return f"⚠️ Error: {e}"

# -----------------------------
# 🧩 Progression Predictors
# -----------------------------

def run_progress_basic(age, gender, education, adas13, moca, cdrsb, faq, apoe4_count, gdtotal):
    """
    Predicts Alzheimer's disease progression (e.g., 2-year conversion) using basic clinical features.
    """
    try:
        # Encode gender
        gender_enc = encode_gender(gender)
        
        # Build input dictionary matching training feature names
        input_dict = {
            "AGE": age,
            "PTGENDER": gender_enc,
            "PTEDUCAT": education,
            "ADAS13": adas13,
            "MOCA": moca,
            "CDRSB": cdrsb,
            "FAQ": faq,
            "APOE4_count": apoe4_count,
            "GDTOTAL": gdtotal
        }
        
        # Preprocess input
        X_processed = preprocess_for_prediction(
            input_dict=input_dict,
            feature_order=PROGRESS_BASIC_FEATURES,
            numeric_cols=PROGRESS_BASIC_NUMERIC_COLS,
            categorical_cols=PROGRESS_BASIC_CATEGORICAL_COLS,
            scaler=progress_basic_scaler
        )
        
        # Predict probability of progression
        probs = progress_basic_model.predict_proba(X_processed)[0]
        prog_prob = float(probs[1])  # Assuming binary: 0=stable, 1=progress
        
        # Format result
        result = (
            f"🧩 Predicted 2-year progression risk: {prog_prob*100:.1f}%\n"
            f"📊 Probabilities: Stable={probs[0]:.3f}, Progress={probs[1]:.3f}"
        )
        
        log_usage("progress_basic", {"features": input_dict}, result)
        return result
        
    except Exception as e:
        error_msg = f"⚠️ Error in progression prediction: {str(e)}"
        log_usage("progress_basic_error", {}, {"error": str(e)})
        return error_msg

def run_progress_advanced(age, gender, education, adas13, moca, cdrsb, faq, 
                         apoe4_count, gdtotal, abeta, tau, ptau, fdg, pib, av45, fbb):
    """
    Predicts Alzheimer's disease progression using advanced multimodal biomarkers.
    """
    try:
        # Encode gender
        gender_enc = encode_gender(gender)
        
        # Build input dictionary
        input_dict = {
            "AGE": age,
            "PTGENDER": gender_enc,
            "PTEDUCAT": education,
            "ADAS13": adas13,
            "MOCA": moca,
            "CDRSB": cdrsb,
            "FAQ": faq,
            "APOE4_count": apoe4_count,
            "GDTOTAL": gdtotal,
            "ABETA": abeta,
            "TAU": tau,
            "PTAU": ptau,
            "FDG": fdg,
            "PIB": pib,
            "AV45": av45,
            "FBB": fbb
        }
        
        # Preprocess
        X_processed = preprocess_for_prediction(
            input_dict=input_dict,
            feature_order=PROGRESS_ADVANCED_FEATURES,
            numeric_cols=PROGRESS_ADVANCED_NUMERIC_COLS,
            categorical_cols=PROGRESS_ADVANCED_CATEGORICAL_COLS,
            scaler=progress_advanced_scaler
        )
        
        # Predict
        probs = progress_advanced_model.predict_proba(X_processed)[0]
        prog_prob = float(probs[1])
        
        result = (
            f"🧬 Predicted 2-year progression risk (Advanced): {prog_prob*100:.1f}%\n"
            f"📊 Probabilities: Stable={probs[0]:.3f}, Progress={probs[1]:.3f}"
        )
        
        log_usage("progress_advanced", {"features": input_dict}, result)
        return result
        
    except Exception as e:
        error_msg = f"⚠️ Error in advanced progression prediction: {str(e)}"
        log_usage("progress_advanced_error", {}, {"error": str(e)})
        return error_msg

# -----------------------------
# Gradio App
# -----------------------------
with gr.Blocks(title="🧬 Alzheimer Diagnosis & Prognosis Tool (RUO)") as demo:
    gr.Markdown(
        """
        ## 🧠 Alzheimer Diagnostic & Prognostic Suite — *Research Use Only*

        Using **ADNI data** (Alzheimer's Disease Neuroimaging Initiative), a longitudinal multicenter study
        that collects clinical, imaging, genetic, and biomarker data to better understand the progression of Alzheimer's disease.
        """
    )
    with gr.Tabs():
        # Risk Screener
        with gr.TabItem("Risk Screener"):
            age = gr.Slider(40, 90, 65, label="Age")
            gender = gr.Dropdown(["Male", "Female"], label="Gender")
            education = gr.Slider(0, 25, 16, label="Years of Education")
            apoe4 = gr.Slider(0, 2, 1, step=1, label="APOE4 Allele Count")
            memory_score = gr.Slider(0, 30, 25, label="Memory Score")
            hippocampus = gr.Slider(1000, 6000, 3500, label="Hippocampal Volume (mm³)")
            result = gr.Textbox(label="Result", lines=2)
            gr.Button("Estimate Risk").click(
                run_risk_screener,
                [age, gender, education, apoe4, memory_score, hippocampus],
                result,
            )

        # First Classifier (Pipeline model: tests-only)
        with gr.TabItem("Classifier (Tests Only)"):
            age_f = gr.Slider(40, 90, 65, label="Age")
            gender_f = gr.Dropdown(["Male", "Female"], label="Gender")
            education_f = gr.Slider(0, 25, 16, label="Education (Years)")
            adas13 = gr.Slider(0, 85, 20, label="ADAS13")
            moca = gr.Slider(0, 30, 25, label="MoCA")
            cdrsb = gr.Slider(0, 18, 3, label="CDRSB")
            faq = gr.Slider(0, 30, 5, label="FAQ")
            race = gr.Slider(0, 7, 1, step=1, label="Race (Encoded)")
            result1 = gr.Textbox(label="Result", lines=3)
            gr.Button("Classify").click(
                run_first_classifier,
                [age_f, gender_f, education_f, adas13, moca, cdrsb, faq, race],
                result1,
            )

        # Classifier Basic
        with gr.TabItem("Classifier (Basic Features)"):
            age_b = gr.Slider(40, 90, 65, label="Age")
            mmse = gr.Slider(0, 30, 26, label="MMSE")
            cdrsb_b = gr.Slider(0, 18, 3, label="CDRSB")
            faq_b = gr.Slider(0, 30, 5, label="FAQ")
            education_b = gr.Slider(0, 25, 16, label="Education (Years)")
            gender_b = gr.Dropdown(["Male", "Female"], label="Gender")
            apoe4_b = gr.Slider(0, 2, 1, step=1, label="APOE4 Allele Count")
            ravlt = gr.Slider(0, 75, 45, label="RAVLT Immediate")
            moca_b = gr.Slider(0, 30, 25, label="MOCA")
            adas13_b = gr.Slider(0, 85, 20, label="ADAS13")
            result2 = gr.Textbox(label="Result", lines=3)
            gr.Button("Run Basic Classifier").click(
                run_classifier_basic,
                [age_b, mmse, cdrsb_b, faq_b, education_b, gender_b, apoe4_b, ravlt, moca_b, adas13_b],
                result2,
            )

        # Classifier Advanced
        with gr.TabItem("Classifier (Advanced Features)"):
            gr.Markdown("### 🧬 Advanced Alzheimer Classifier\n"
                        "This mode builds upon core clinical features with biomarkers and neuroimaging metrics "
                        "to provide a more comprehensive prediction. "
                        "_Adjust clinical and biomarker inputs below._")

            # --- Clinical (basic) features reused ---
            gr.Markdown("#### 🧠 Clinical & Cognitive Features")
            age_a = gr.Slider(40, 90, 65, label="Age")
            mmse_a = gr.Slider(0, 30, 26, label="MMSE")
            cdrsb_a = gr.Slider(0, 18, 3, label="CDRSB")
            faq_a = gr.Slider(0, 30, 5, label="FAQ")
            education_a = gr.Slider(0, 25, 16, label="Education (Years)")
            gender_a = gr.Dropdown(["Male", "Female"], label="Gender")
            apoe4_a = gr.Slider(0, 2, 1, step=1, label="APOE4 Allele Count")
            ravlt_a = gr.Slider(0, 75, 45, label="RAVLT Immediate")
            moca_a = gr.Slider(0, 30, 25, label="MoCA")
            adas13_a = gr.Slider(0, 85, 20, label="ADAS13")

            # --- Biomarkers & Imaging features ---
            gr.Markdown("#### 🧬 Biomarker & Imaging Features (Default values approximate normal ranges)")
            default_biomarker_values = {
                "Hippocampus_bl": 4000,
                "Ventricles_bl": 30000,
                "WholeBrain_bl": 1100000,
                "Entorhinal_bl": 3800,
                "FDG_bl": 1.2,
                "AV45_bl": 1.3,
                "PIB_bl": 1.1,
                "FBB_bl": 1.0,
                "ABETA_bl": 800,
                "TAU_bl": 300,
                "PTAU_bl": 30,
                "mPACCdigit_bl": 0.5,
                "mPACCtrailsB_bl": 0.4,
            }

            biomarker_inputs = [
                gr.Number(label=feature, value=value)
                for feature, value in default_biomarker_values.items()
            ]

            result3 = gr.Textbox(label="Result", lines=4)

            gr.Button("Run Advanced Classifier").click(
                run_classifier_advanced,
                [age_a, mmse_a, cdrsb_a, faq_a, education_a, gender_a,
                apoe4_a, ravlt_a, moca_a, adas13_a] + biomarker_inputs,
                result3
            )

        with gr.TabItem("Progression Predictor (Basic)"):
            age_pb = gr.Slider(40, 90, 65, label="Age")
            gender_pb = gr.Dropdown(["Male", "Female"], label="Gender")
            education_pb = gr.Slider(0, 25, 16, label="Education (Years)")
            adas13_pb = gr.Slider(0, 85, 20, label="ADAS13")
            moca_pb = gr.Slider(0, 30, 25, label="MOCA")
            cdrsb_pb = gr.Slider(0, 18, 3, label="CDRSB")
            faq_pb = gr.Slider(0, 30, 5, label="FAQ")
            apoe4_pb = gr.Slider(0, 2, 1, step=1, label="APOE4 Allele Count")
            gdtotal_pb = gr.Slider(0, 10, 3, label="GDTOTAL")
            result_pb = gr.Textbox(label="Result", lines=3)
            gr.Button("Run Progression (Basic)").click(
                run_progress_basic,
                [age_pb, gender_pb, education_pb, adas13_pb, moca_pb, cdrsb_pb, faq_pb, apoe4_pb, gdtotal_pb],
                result_pb,
            )

        with gr.TabItem("Progression Predictor (Advanced)"):
            gr.Markdown("### 🧠 Predict Alzheimer's progression using advanced multimodal biomarkers")

            age_pa = gr.Slider(40, 90, 65, label="Age")
            gender_pa = gr.Dropdown(["Male", "Female"], label="Gender")
            education_pa = gr.Slider(0, 25, 16, label="Education (Years)")
            adas13_pa = gr.Slider(0, 85, 20, label="ADAS13")
            moca_pa = gr.Slider(0, 30, 25, label="MOCA")
            cdrsb_pa = gr.Slider(0, 18, 3, label="CDRSB")
            faq_pa = gr.Slider(0, 30, 5, label="FAQ")
            apoe4_pa = gr.Slider(0, 2, 1, step=1, label="APOE4 Allele Count")
            gdtotal_pa = gr.Slider(0, 10, 3, label="GDTOTAL")

            gr.Markdown("#### 🧪 Biomarkers")
            abeta_pa = gr.Slider(label="ABETA (CSF Aβ42)", minimum=200, maximum=2000, step=10, value=1000)
            tau_pa = gr.Slider(label="TAU (CSF Total Tau)", minimum=50, maximum=1500, step=10, value=300)
            ptau_pa = gr.Slider(label="PTAU (CSF Phospho-Tau)", minimum=10, maximum=200, step=1, value=50)
            fdg_pa = gr.Slider(label="FDG (PET Metabolism)", minimum=0.5, maximum=2.0, step=0.01, value=1.0)
            pib_pa = gr.Slider(label="PIB (Amyloid PET)", minimum=0.5, maximum=2.5, step=0.01, value=1.0)
            av45_pa = gr.Slider(label="AV45 (Amyloid PET Tracer)", minimum=0.5, maximum=2.5, step=0.01, value=1.0)
            fbb_pa = gr.Slider(label="FBB (Florbetaben PET)", minimum=0.5, maximum=2.5, step=0.01, value=1.0)

            result_pa = gr.Textbox(label="Result", lines=3)
            gr.Button("Run Progression (Advanced)").click(
                run_progress_advanced,
                [
                    age_pa, gender_pa, education_pa, adas13_pa, moca_pa, cdrsb_pa, faq_pa,
                    apoe4_pa, gdtotal_pa, abeta_pa, tau_pa, ptau_pa, fdg_pa, pib_pa, av45_pa, fbb_pa
                ],
                result_pa,
            )

# -----------------------------
# 🔄 Warm-up / Preload Function
# -----------------------------
def warm_up_models():
    """
    Perform a dummy prediction for each model to initialize internal structures.
    Prevents first-click errors in Gradio.
    """
    print("⚡ Warming up models...")

    # First classifier (pipeline)
    try:
        dummy_df = pd.DataFrame([[65, 16, 25, 20, 3, 5, "male", 1]], columns=FIRST_CLASSIFIER_FEATURES)
        _ = classifier_1_model.predict_proba(dummy_df)
        print("✅ First classifier warmed up")
    except Exception as e:
        print(f"⚠️ First classifier warm-up failed: {e}")

    # Basic classifier
    try:
        dummy_vals = [65, 26, 3, 5, 16, 1, 1, 45, 25, 20]
        X_df = build_df_from_order(dummy_vals, BASIC_FEATURE_ORDER)
        numeric_cols = [c for c in BASIC_FEATURE_ORDER if c not in ("PTGENDER", "APOE4")]
        X_scaled = np.hstack([classifier_basic_preprocessor.transform(X_df[numeric_cols]), X_df[["PTGENDER", "APOE4"]].to_numpy()])
        _ = classifier_basic_model.predict_proba(X_scaled)
        print("✅ Basic classifier warmed up")
    except Exception as e:
        print(f"⚠️ Basic classifier warm-up failed: {e}")

    # Advanced classifier
    try:
        dummy_input = np.zeros((1, len(ADVANCED_FEATURE_ORDER)))
        X_num = dummy_input[:, :-2]  # exclude categorical
        X_cat = dummy_input[:, -2:]  # last 2 are categorical
        X_scaled = np.hstack([classifier_advanced_preprocessor.transform(X_num), X_cat])
        _ = classifier_advanced_model.predict_proba(X_scaled)
        print("✅ Advanced classifier warmed up")
    except Exception as e:
        print(f"⚠️ Advanced classifier warm-up failed: {e}")

    # Progression basic
    try:
        dummy_dict = {k: 0 for k in PROGRESS_BASIC_FEATURES}
        dummy_dict["PTGENDER"] = 1
        X_processed = preprocess_for_prediction(dummy_dict, PROGRESS_BASIC_FEATURES, PROGRESS_BASIC_NUMERIC_COLS, PROGRESS_BASIC_CATEGORICAL_COLS, progress_basic_scaler)
        _ = progress_basic_model.predict_proba(X_processed)
        print("✅ Progression basic model warmed up")
    except Exception as e:
        print(f"⚠️ Progression basic warm-up failed: {e}")

    # Progression advanced
    try:
        dummy_dict = {k: 0 for k in PROGRESS_ADVANCED_FEATURES}
        dummy_dict["PTGENDER"] = 1
        X_processed = preprocess_for_prediction(dummy_dict, PROGRESS_ADVANCED_FEATURES, PROGRESS_ADVANCED_NUMERIC_COLS, PROGRESS_ADVANCED_CATEGORICAL_COLS, progress_advanced_scaler)
        _ = progress_advanced_model.predict_proba(X_processed)
        print("✅ Progression advanced model warmed up")
    except Exception as e:
        print(f"⚠️ Progression advanced warm-up failed: {e}")

    print("⚡ All models warmed up successfully!")

# -----------------------------
# Main Launch
# -----------------------------
if __name__ == "__main__":
    # Warm up models BEFORE launching Gradio
    warm_up_models()

    # Launch Gradio app
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
