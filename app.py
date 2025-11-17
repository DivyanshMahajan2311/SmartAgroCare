# app.py (merged)
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, redirect, url_for, render_template_string

# TensorFlow/Keras loader fallback
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import MeanSquaredError
except Exception:
    try:
        from keras.models import load_model
        from keras.losses import MeanSquaredError
    except Exception:
        load_model = None
        MeanSquaredError = None

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV_PATH = BASE_DIR / "Amritsar_Weather_1Year.csv"
FORECAST_DAYS = 7

# Model file locations (exact names you provided)
FILES = {
    "wind": {
        "lstm": BASE_DIR / "wind-speed_lstm_model.h5",
        "svm": BASE_DIR / "wind-speed_svm_model.pkl",
        "scaler_X": BASE_DIR / "wind-speed_scaler_X.pkl",
        "scaler_y": BASE_DIR / "wind-speed_scaler_y.pkl",
    },
    "precip": {
        "lstm": BASE_DIR / "prec_lstm_model.h5",
        "svm": BASE_DIR / "prec_svm_model.pkl",
        "scaler": BASE_DIR / "prec_scaler.pkl",
    },
    "temp": {
        "combined": BASE_DIR / "min-max_temp.pkl",
        "temp_min_lstm": BASE_DIR / "temp_min_lstm_model.h5",
        "temp_min_svm": BASE_DIR / "temp_min_svm_model.pkl",
        "temp_min_scaler_X": BASE_DIR / "temp_min_scaler_X.pkl",
        "temp_min_scaler_y": BASE_DIR / "temp_min_scaler_y.pkl",
        "temp_max_lstm": BASE_DIR / "temp_max_lstm_model.h5",
        "temp_max_svm": BASE_DIR / "temp_max_svm_model.pkl",
        "temp_max_scaler_X": BASE_DIR / "temp_max_scaler_X.pkl",
        "temp_max_scaler_y": BASE_DIR / "temp_max_scaler_y.pkl",
    }
}

# Crop & fertilizer model filenames (updated to the separate crop + season models)
CROP_FILES = {
    "crop_model": BASE_DIR / "crop_model.pkl",            # classifier with predict_proba
    "season_model": BASE_DIR / "season_model.pkl",        # season predictor
    "crop_encoder": BASE_DIR / "crop_label_encoder.pkl",
    "season_encoder": BASE_DIR / "season_label_encoder.pkl",
    "crop_time_csv": BASE_DIR / "balanced_crop_dataset.csv"  # dataset for growing times
}

FERT_FILES = {
    "model": BASE_DIR / "fertilizer_rf_model.pkl",
    "label_encoder": BASE_DIR / "label_encoder.pkl",
    "crop_encoder": BASE_DIR / "crop_encoder.pkl"
}

# ---------- Flask app ----------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- Utilities ----------
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

class IdentityScaler:
    """No-op scaler for fallback"""
    def __init__(self, n_features_in_=1):
        self.n_features_in_ = n_features_in_

    def transform(self, X):
        import numpy as _np
        X = _np.asarray(X)
        return X

    def inverse_transform(self, X):
        import numpy as _np
        X = _np.asarray(X)
        return X

    def fit(self, X, y=None):
        import numpy as _np
        self.n_features_in_ = _np.asarray(X).shape[1]
        return self

    def __repr__(self):
        return f"IdentityScaler(n_features_in_={getattr(self,'n_features_in_',None)})"

def parse_unit_object(unit_obj):
    """Parse dict/path/object into {lstm, svm, scaler_X, scaler_y}"""
    out = {"lstm": None, "svm": None, "scaler_X": None, "scaler_y": None}
    if unit_obj is None:
        return out
    if isinstance(unit_obj, dict):
        # lstm can be path or model
        lstm_val = unit_obj.get("lstm") or unit_obj.get("model") or unit_obj.get("lstm_path")
        if isinstance(lstm_val, (str, Path)):
            p = Path(lstm_val)
            if p.exists() and load_model is not None:
                out["lstm"] = load_model(str(p), custom_objects={'mse': MeanSquaredError()})
        elif lstm_val is not None:
            out["lstm"] = lstm_val

        svm_val = unit_obj.get("svm")
        if isinstance(svm_val, (str, Path)):
            p = Path(svm_val)
            if p.exists():
                out["svm"] = load_pickle(p)
        elif svm_val is not None:
            out["svm"] = svm_val

        sx = unit_obj.get("scaler_X") or unit_obj.get("scalerX") or unit_obj.get("scaler")
        sy = unit_obj.get("scaler_y") or unit_obj.get("scalerY") or unit_obj.get("scaler")
        if isinstance(sx, (str, Path)):
            psx = Path(sx)
            if psx.exists():
                out["scaler_X"] = load_pickle(psx)
        elif sx is not None:
            out["scaler_X"] = sx

        if isinstance(sy, (str, Path)):
            psy = Path(sy)
            if psy.exists():
                out["scaler_y"] = load_pickle(psy)
        elif sy is not None:
            out["scaler_y"] = sy

        if out["scaler_X"] is None and out["scaler_y"] is not None:
            out["scaler_X"] = out["scaler_y"]
        if out["scaler_y"] is None and out["scaler_X"] is not None:
            out["scaler_y"] = out["scaler_X"]
        return out

    if isinstance(unit_obj, (str, Path)):
        p = Path(unit_obj)
        if p.exists():
            loaded = load_pickle(p)
            return parse_unit_object(loaded)
    return out

def align_features_to_scaler(X, scaler):
    import numpy as _np
    n_in = X.shape[1]
    n_expected = getattr(scaler, "n_features_in_", None)
    if n_expected is None:
        return X
    if n_in == n_expected:
        return X
    if n_in < n_expected:
        pad = _np.zeros((1, n_expected - n_in), dtype=X.dtype)
        return _np.hstack([X, pad])
    return X[:, :n_expected]

# ------------------ Organic prepare_input helper (same behavior as single-file app) ------------------
def prepare_input_organic(payload, feature_cols):
    """
    Convert incoming payload (strings or types) to DataFrame with same columns/order as training.
    Expects keys: Crop, pH, N_kg_per_ha, P_kg_per_ha, K_kg_per_ha, Moisture_percent
    """
    import pandas as _pd
    sample = {
        "pH": float(payload.get("pH", 7.0)),
        "N_kg_per_ha": float(payload.get("N_kg_per_ha", 0.0)),
        "P_kg_per_ha": float(payload.get("P_kg_per_ha", 0.0)),
        "K_kg_per_ha": float(payload.get("K_kg_per_ha", 0.0)),
        "Moisture_percent": float(payload.get("Moisture_percent", 0.0)),
        "Crop": payload.get("Crop", "")
    }
    df = _pd.DataFrame([sample])
    df = _pd.get_dummies(df, columns=["Crop"], drop_first=True)
    # add missing features as zeros and order columns to feature_cols
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_cols].astype(float)
    return df
# ---------------------------------------------------------------------------------------------------

# ---------- Load crop & fertilizer models ----------
MODEL_CACHE = {}
LOAD_ERRORS = {}

def startup_load_basic_models():
    # Crop (separate crop_model + season_model + time CSV)
    try:
        # crop classifier
        if not CROP_FILES["crop_model"].exists():
            raise FileNotFoundError(f"Missing crop model: {CROP_FILES['crop_model']}")
        crop_model = load_pickle(CROP_FILES["crop_model"])

        # season predictor
        if not CROP_FILES["season_model"].exists():
            raise FileNotFoundError(f"Missing season model: {CROP_FILES['season_model']}")
        season_model = load_pickle(CROP_FILES["season_model"])

        crop_encoder = load_pickle(CROP_FILES["crop_encoder"]) if CROP_FILES["crop_encoder"].exists() else None
        season_encoder = load_pickle(CROP_FILES["season_encoder"]) if CROP_FILES["season_encoder"].exists() else None

        # load CSV used for average time lookup (optional but recommended)
        time_df = None
        if CROP_FILES.get("crop_time_csv") and CROP_FILES["crop_time_csv"].exists():
            try:
                time_df = pd.read_csv(CROP_FILES["crop_time_csv"])
            except Exception as ex:
                # non-fatal; keep time_df as None but record warning
                LOAD_ERRORS.setdefault("crop_warnings", []).append(f"Could not read crop time CSV: {ex}")

        MODEL_CACHE["crop"] = {
            "crop_model": crop_model,
            "season_model": season_model,
            "crop_encoder": crop_encoder,
            "season_encoder": season_encoder,
            "time_df": time_df
        }
    except Exception as e:
        LOAD_ERRORS["crop"] = str(e)

    # Fertilizer
    try:
        if not FERT_FILES["model"].exists():
            raise FileNotFoundError(f"Missing fertilizer model: {FERT_FILES['model']}")
        fert_model = load_pickle(FERT_FILES["model"])
        label_encoder = load_pickle(FERT_FILES["label_encoder"]) if FERT_FILES["label_encoder"].exists() else None
        crop_encoder = load_pickle(FERT_FILES["crop_encoder"]) if FERT_FILES["crop_encoder"].exists() else None
        MODEL_CACHE["fertilizer"] = {"model": fert_model, "label_encoder": label_encoder, "crop_encoder": crop_encoder}
    except Exception as e:
        LOAD_ERRORS["fertilizer"] = str(e)
        
    # ---------- Optional: load organic fertilizer artifacts (single-file app artifacts) ----------
    ORGANIC_ARTIFACTS = {
        "model": BASE_DIR / "fertilizer_model.pkl",
        "mlb": BASE_DIR / "fertilizer_label_encoder.pkl",
        "feature_cols": BASE_DIR / "fertilizer_feature_columns.pkl"
    }

    try:
        if (ORGANIC_ARTIFACTS["model"].exists() and 
            ORGANIC_ARTIFACTS["mlb"].exists() and 
            ORGANIC_ARTIFACTS["feature_cols"].exists()):
            
            import joblib
            organic_model = joblib.load(str(ORGANIC_ARTIFACTS["model"]))
            organic_mlb = joblib.load(str(ORGANIC_ARTIFACTS["mlb"]))
            organic_feature_cols = joblib.load(str(ORGANIC_ARTIFACTS["feature_cols"]))

            MODEL_CACHE["fertilizer_organic"] = {
                "model": organic_model,
                "mlb": organic_mlb,
                "feature_cols": organic_feature_cols
            }
    except Exception as e:
        LOAD_ERRORS["fertilizer_organic"] = str(e)

startup_load_basic_models()

# ---------- Weather model loaders (same logic as your weather app) ----------
def load_wind_models():
    f = FILES["wind"]
    for k, p in f.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing wind file: {p}")
    lstm = load_model(str(f["lstm"]), custom_objects={'mse': MeanSquaredError()})
    svm = load_pickle(f["svm"])
    scaler_X = load_pickle(f["scaler_X"])
    scaler_y = load_pickle(f["scaler_y"])
    return {"lstm": lstm, "svm": svm, "scaler_X": scaler_X, "scaler_y": scaler_y}

def load_precip_models():
    f = FILES["precip"]
    for k,p in f.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing precip file: {p}")
    lstm = load_model(str(f["lstm"]), custom_objects={'mse': MeanSquaredError()})
    svm = load_pickle(f["svm"])
    sc = load_pickle(f["scaler"])
    if isinstance(sc, dict):
        scaler_X = sc.get("scaler_X") or sc.get("scalerX") or sc.get("scaler")
        scaler_y = sc.get("scaler_y") or sc.get("scalerY") or sc.get("scaler")
        if scaler_X is None and scaler_y is not None:
            scaler_X = scaler_y
        if scaler_y is None and scaler_X is not None:
            scaler_y = scaler_X
    else:
        scaler_X = sc
        scaler_y = IdentityScaler(n_features_in_=1)
    return {"lstm": lstm, "svm": svm, "scaler_X": scaler_X, "scaler_y": scaler_y}

def load_temp_models():
    tfiles = FILES["temp"]
    min_files = [
        tfiles["temp_min_lstm"], tfiles["temp_min_svm"],
        tfiles["temp_min_scaler_X"], tfiles["temp_min_scaler_y"]
    ]
    max_files = [
        tfiles["temp_max_lstm"], tfiles["temp_max_svm"],
        tfiles["temp_max_scaler_X"], tfiles["temp_max_scaler_y"]
    ]
    min_exist = all(p.exists() for p in min_files)
    max_exist = all(p.exists() for p in max_files)
    if min_exist and max_exist:
        lstm_min = load_model(str(tfiles["temp_min_lstm"]), custom_objects={'mse': MeanSquaredError()})
        svm_min = load_pickle(tfiles["temp_min_svm"])
        scaler_min_X = load_pickle(tfiles["temp_min_scaler_X"])
        scaler_min_y = load_pickle(tfiles["temp_min_scaler_y"])

        lstm_max = load_model(str(tfiles["temp_max_lstm"]), custom_objects={'mse': MeanSquaredError()})
        svm_max = load_pickle(tfiles["temp_max_svm"])
        scaler_max_X = load_pickle(tfiles["temp_max_scaler_X"])
        scaler_max_y = load_pickle(tfiles["temp_max_scaler_y"])

        return {
            "temp_min": {"lstm": lstm_min, "svm": svm_min, "scaler_X": scaler_min_X, "scaler_y": scaler_min_y},
            "temp_max": {"lstm": lstm_max, "svm": svm_max, "scaler_X": scaler_max_X, "scaler_y": scaler_max_y}
        }

    p = tfiles["combined"]
    if not p.exists():
        raise FileNotFoundError(f"Missing temperature pickle: {p} and individual temp_* files not found.")
    content = load_pickle(p)
    from sklearn.svm import SVR as _SVR
    if isinstance(content, _SVR):
        n_feat = getattr(content, "n_features_in_", None)
        if n_feat is None:
            n_feat = 20
        parsed = {
            "lstm": None,
            "svm": content,
            "scaler_X": IdentityScaler(n_features_in_=n_feat),
            "scaler_y": IdentityScaler(n_features_in_=1)
        }
        import copy
        return {"temp_min": parsed, "temp_max": copy.deepcopy(parsed)}
    if isinstance(content, dict) and ("temp_min" in content or "temp_max" in content):
        tmin = parse_unit_object(content.get("temp_min"))
        tmax = parse_unit_object(content.get("temp_max"))
        return {"temp_min": tmin, "temp_max": tmax}
    if isinstance(content, dict):
        lowermap = {k.lower(): k for k in content.keys()}
        tmin_key = lowermap.get('temp_min') or lowermap.get('tmin') or lowermap.get('min_temp')
        tmax_key = lowermap.get('temp_max') or lowermap.get('tmax') or lowermap.get('max_temp')
        if tmin_key and tmax_key:
            return {'temp_min': parse_unit_object(content[tmin_key]), 'temp_max': parse_unit_object(content[tmax_key])}
    parsed = parse_unit_object(content)
    if any(parsed.values()):
        return {'temp_min': parsed, 'temp_max': parsed}
    raise ValueError(
        "min-max_temp.pkl not in expected format. It should be a dict with 'temp_min' and 'temp_max', "
    )

# ---------- Feature builder & iterative forecast ----------
def build_feature_row(buffer_y, ds):
    buf = list(buffer_y[-7:]) if len(buffer_y) >= 7 else ([0] * (7 - len(buffer_y)) + list(buffer_y))
    feat = {}
    for i in range(1, 8):
        feat[f"lag_{i}"] = buf[-i]
    feat["var_7d_mean"] = float(np.mean(buf))
    feat["var_7d_std"] = float(np.std(buf))
    feat["day_of_week"] = int(ds.weekday())
    feat["month"] = int(ds.month)
    return feat

def iterative_forecast_models(models, history_series, days=7):
    lstm = models.get("lstm")
    svm = models.get("svm")
    scaler_X = models.get("scaler_X")
    scaler_y = models.get("scaler_y")
    if svm is None or scaler_X is None or scaler_y is None:
        raise ValueError("One of the required model components is missing (svm/scaler_X/scaler_y).")

    buffer = list(history_series.dropna().astype(float).tolist()[-7:])
    today = datetime.now().date()
    forecasts = []
    for i in range(1, days + 1):
        ds = datetime.combine(today + timedelta(days=i), datetime.min.time())
        feat = build_feature_row(buffer, ds)
        X = [feat[f"lag_{j}"] for j in range(1, 8)]
        X += [feat["var_7d_mean"], feat["var_7d_std"], feat["day_of_week"], feat["month"]]
        X = np.array(X).reshape(1, -1)

        X_aligned = align_features_to_scaler(X, scaler_X)
        Xs = scaler_X.transform(X_aligned)

        if lstm is not None:
            Xl = Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))
            pred_lstm_scaled = lstm.predict(Xl, verbose=0)
            try:
                pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled.reshape(-1, 1)).flatten()[0]
            except Exception:
                pred_lstm = float(np.asarray(pred_lstm_scaled).reshape(-1)[0])
        else:
            pred_lstm = 0.0

        residual_pred = float(svm.predict(Xs.reshape(1, -1))[0])
        hybrid_pred = float(pred_lstm + residual_pred)
        hybrid_pred = max(hybrid_pred, 0.0)
        forecasts.append({"date": ds.date().isoformat(), "value": round(hybrid_pred, 4)})
        buffer.append(hybrid_pred)

    return forecasts

# ---------- CSV loader ----------
def load_csv():
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {DATA_CSV_PATH}. Place your CSV there or change DATA_CSV_PATH.")
    df = pd.read_csv(DATA_CSV_PATH)
    if "time" in df.columns and "ds" not in df.columns:
        df = df.rename(columns={"time": "ds"})
    df["ds"] = pd.to_datetime(df["ds"], format="%d-%b-%y", errors="coerce")
    df = df.dropna(subset=["ds"])
    df = df.set_index("ds").sort_index()
    return df

# ---------- Startup load heavy weather models ----------
def startup_load_weather():
    try:
        MODEL_CACHE["wind"] = load_wind_models()
    except Exception as e:
        LOAD_ERRORS["wind"] = str(e)
    try:
        MODEL_CACHE["precip"] = load_precip_models()
    except Exception as e:
        LOAD_ERRORS["precip"] = str(e)
    try:
        MODEL_CACHE["temp"] = load_temp_models()
    except Exception as e:
        LOAD_ERRORS["temp"] = str(e)

# Only attempt to load weather models if load_model is available
if load_model is None:
    LOAD_ERRORS["weather"] = "Keras/TensorFlow not found; weather models cannot be loaded."
else:
    startup_load_weather()

# ---------- Routes ----------
@app.route("/")
def index():
    today_str = datetime.now().strftime("%A, %d %B %Y")
    # Note: index.html you provided expects to be rendered — pass load_errors optionally
    return render_template("index.html", today=today_str, load_errors=LOAD_ERRORS)

# Simple About & Privacy (templates not provided—render strings to avoid 404)
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/privacy")
def privacy():
    return render_template_string("""
      <!doctype html><title>Privacy</title>
      <h1>Privacy</h1>
      <p>This demo stores no personal data. Models and CSV remain local to your deployment.</p>
      <p><a href="/">Home</a></p>
    """)

# ---------- Crop endpoints ----------
@app.route("/crop", methods=["GET"])
def crop_page():
    return render_template("crop.html")

@app.route("/crop/predict", methods=["GET", "POST"]) 
def crop_predict():
    if request.method == "GET":
        return render_template_string("<p>This endpoint accepts POST from the crop form.</p>")
    else:
        # Use the improved top-3 crop + season logic (predict_proba) with ndarray-first approach
        if "crop" in LOAD_ERRORS:
            return render_template_string(f"<pre>Load error: {LOAD_ERRORS['crop']}</pre>"), 500
        try:
            data = request.form
            nitrogen = float(data.get('nitrogen'))
            phosphorus = float(data.get('phosphorus'))
            potassium = float(data.get('potassium'))
            temperature = float(data.get('temperature'))
            humidity = float(data.get('humidity'))
            ph_val = float(data.get('ph'))
            rainfall = float(data.get('rainfall'))

            features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall]])
            crop_entry = MODEL_CACHE.get("crop")
            if crop_entry is None:
                raise RuntimeError("Crop models not loaded.")
            crop_model = crop_entry.get("crop_model")
            season_model = crop_entry.get("season_model")
            crop_encoder = crop_entry.get("crop_encoder")
            season_encoder = crop_entry.get("season_encoder")
            time_df = crop_entry.get("time_df")  # may be None

            # --- Predict top-3 crops (try ndarray first to avoid DataFrame feature-name checks) ---
            X_arr = features  # (1,7)
            top_labels = None
            try:
                if hasattr(crop_model, "predict_proba"):
                    probs = crop_model.predict_proba(X_arr)[0]
                    classes = np.array(crop_model.classes_)
                    top_idx = np.argsort(probs)[::-1][:3]
                    top_labels = classes[top_idx]
                else:
                    preds = crop_model.predict(X_arr)
                    arr = np.asarray(preds).reshape(-1)
                    top_labels = arr[:3] if arr.size >= 3 else np.pad(arr, (0, max(0, 3 - arr.size)),
                                                                    constant_values=arr[-1] if arr.size > 0 else 0)
            except Exception as e_nd:
                # ndarray approach failed; try DataFrame with expected feature names
                try:
                    if hasattr(crop_model, "feature_names_in_"):
                        cols = list(getattr(crop_model, "feature_names_in_"))
                    else:
                        cols = ["N", "P", "K", "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (in cm)"]
                    X_df = pd.DataFrame(X_arr, columns=cols)
                    if hasattr(crop_model, "predict_proba"):
                        probs = crop_model.predict_proba(X_df)[0]
                        classes = np.array(crop_model.classes_)
                        top_idx = np.argsort(probs)[::-1][:3]
                        top_labels = classes[top_idx]
                    else:
                        preds = crop_model.predict(X_df)
                        arr = np.asarray(preds).reshape(-1)
                        top_labels = arr[:3] if arr.size >= 3 else np.pad(arr, (0, max(0, 3 - arr.size)),
                                                                        constant_values=arr[-1] if arr.size > 0 else 0)
                except Exception as e_df:
                    raise RuntimeError(f"Both ndarray and DataFrame prediction failed: ndarray error: {e_nd}; dataframe attempt error: {e_df}")

            # decode top labels to human names (handle encoders that expect ints)
            try:
                main_crop = crop_encoder.inverse_transform([int(top_labels[0])])[0] if crop_encoder is not None else str(top_labels[0])
                mix_crop_1 = crop_encoder.inverse_transform([int(top_labels[1])])[0] if crop_encoder is not None else str(top_labels[1])
                mix_crop_2 = crop_encoder.inverse_transform([int(top_labels[2])])[0] if crop_encoder is not None else str(top_labels[2])
            except Exception:
                # labels are likely already strings
                main_crop = str(top_labels[0])
                mix_crop_1 = str(top_labels[1])
                mix_crop_2 = str(top_labels[2])

            # --- Season prediction (try ndarray first, fallback to DataFrame) ---
            try:
                season_pred = season_model.predict(X_arr)
                try:
                    season_name = season_encoder.inverse_transform(np.asarray(season_pred).reshape(-1))[0] if season_encoder is not None else str(season_pred[0])
                except Exception:
                    season_name = str(np.asarray(season_pred).reshape(-1)[0])
            except Exception as e_season:
                # fallback to DataFrame approach
                try:
                    if hasattr(crop_model, "feature_names_in_"):
                        cols = list(getattr(crop_model, "feature_names_in_"))
                    else:
                        cols = ["N", "P", "K", "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (in cm)"]
                    X_df = pd.DataFrame(X_arr, columns=cols)
                    season_pred = season_model.predict(X_df)
                    try:
                        season_name = season_encoder.inverse_transform(np.asarray(season_pred).reshape(-1))[0] if season_encoder is not None else str(season_pred[0])
                    except Exception:
                        season_name = str(np.asarray(season_pred).reshape(-1)[0])
                except Exception as e2:
                    raise RuntimeError(f"Season prediction failed (ndarray err: {e_season}; df attempt err: {e2})")

            # helper to get avg time from time_df
            def get_time_for_crop(crop_name):
                if time_df is None:
                    return None
                subset = time_df[time_df["Crop"] == crop_name]
                col_name = "Time Required to grow (in months)"
                if col_name in subset.columns and not subset[col_name].dropna().empty:
                    avg_time = subset[col_name].astype(float).mean()
                    return round(float(avg_time), 1)
                return None

            main_time = get_time_for_crop(main_crop)
            mix_time_1 = get_time_for_crop(mix_crop_1)
            mix_time_2 = get_time_for_crop(mix_crop_2)

            result = {
                "main_crop": main_crop,
                "main_time": main_time,
                "mix_crop_1": mix_crop_1,
                "mix_time_1": mix_time_1,
                "mix_crop_2": mix_crop_2,
                "mix_time_2": mix_time_2,
                "season": season_name
            }

            # Render result.html with the richer result object (match your crop result UI)
            return render_template("result.html", result=result)

        except Exception as e:
            return render_template_string(f"<pre>Error: {e}</pre>"), 500


# ---------- Fertilizer endpoints ----------
@app.route("/fertilizer", methods=["GET"])
def fertilizer_page():
    # fertilizer.html expects 'prediction' variable when POST returned — here GET shows form
    return render_template("fertilizer.html", prediction=None)

@app.route("/fertilizer/predict", methods=["GET", "POST"])
def fertilizer_predict():
    if request.method == "GET":
        return render_template("fertilizer.html", prediction=None)
    else: 
        if "fertilizer" in LOAD_ERRORS:
            return render_template_string(f"<pre>Load error: {LOAD_ERRORS['fertilizer']}</pre>"), 500
        try:
            data = request.form
            # fertilizer type radio expected as 'fert_type' with values 'organic' or 'inorganic'
            fert_type = data.get("fert_type", "inorganic").strip().lower()

            # ---------- ORGANIC branch (fallback to inorganic if organic model missing) ----------
            if fert_type == "organic":
                organic_entry = MODEL_CACHE.get("fertilizer_organic")
                fallback_to_inorganic = False

                if organic_entry is None:
                    # fallback instead of erroring
                    fallback_to_inorganic = True
                    fert_entry = MODEL_CACHE.get("fertilizer")
                else:
                    fert_entry = organic_entry

                if fert_entry is None:
                    # still nothing available
                    return render_template_string("<pre>Error: Organic model not available on server and no inorganic fallback present.</pre>"), 500

                # If using organic artifacts (joblib multi-label model) -> expect keys: model, mlb, feature_cols
                if (not fallback_to_inorganic) and ("mlb" in fert_entry and "feature_cols" in fert_entry):
                    # Build payload expected by organic prepare_input helper
                    payload = {
                        "Crop": data.get("crop", "").strip() or data.get("Crop", ""),
                        "pH": data.get("pH", data.get("pH", 7.0)),
                        "N_kg_per_ha": data.get("N", data.get("N_kg_per_ha", 0.0)),
                        "P_kg_per_ha": data.get("P", data.get("P_kg_per_ha", 0.0)),
                        "K_kg_per_ha": data.get("K", data.get("K_kg_per_ha", 0.0)),
                        "Moisture_percent": data.get("moisture", data.get("Moisture_percent", 0.0))
                    }

                    # coerce numeric inputs
                    try:
                        payload["pH"] = float(payload["pH"])
                    except Exception:
                        payload["pH"] = 7.0
                    for k in ("N_kg_per_ha", "P_kg_per_ha", "K_kg_per_ha", "Moisture_percent"):
                        try:
                            payload[k] = float(payload[k])
                        except Exception:
                            payload[k] = 0.0

                    organic_model = fert_entry["model"]
                    organic_mlb = fert_entry["mlb"]
                    organic_feature_cols = fert_entry["feature_cols"]

                    # prepare features and predict
                    Xorg = prepare_input_organic(payload, organic_feature_cols)
                    y_pred = organic_model.predict(Xorg)
                    try:
                        ferts = list(organic_mlb.inverse_transform(y_pred)[0])
                    except Exception:
                        try:
                            ferts = list(organic_mlb.inverse_transform(y_pred))
                        except Exception:
                            ferts = []

                    prediction = ", ".join(ferts) if ferts else "No recommendation"
                    return render_template("fertilizer.html", prediction=prediction)

                else:
                    # Either organic artifacts not present or we intentionally fell back to inorganic.
                    # Use the inorganic prediction path (reuse existing inorganic logic).
                    # Build same inputs as inorganic branch and return below.
                    crop = data.get("crop", "").lower().strip()
                    pH = float(data.get("pH"))
                    N = float(data.get("N"))
                    P = float(data.get("P"))
                    K = float(data.get("K"))

                    fert_entry = MODEL_CACHE.get("fertilizer")
                    if fert_entry is None:
                        return render_template_string("<pre>Error: Fertilizer model not loaded.</pre>"), 500
                    model = fert_entry["model"]
                    label_encoder = fert_entry.get("label_encoder")
                    crop_encoder = fert_entry.get("crop_encoder")

                    try:
                        crop_encoded = crop_encoder.transform([crop])[0] if crop_encoder is not None else 0
                    except Exception:
                        crop_encoded = 0

                    input_data = np.array([[crop_encoded, pH, N, P, K]])
                    pred_encoded = model.predict(input_data)
                    pred_encoded = np.asarray(pred_encoded).reshape(-1)
                    pred_val = pred_encoded[0]

                    if label_encoder is not None:
                        raw_pred = label_encoder.inverse_transform([pred_val])[0]
                    else:
                        raw_pred = str(pred_val)

                    _label_map = {
                        "fym": "urea dap mop combo",
                        "other fertilizers": "Soil Amendment (Lime / Gypsum / Micronutrients)",
                        "routine fertilizer schedule": "Follow crop-specific fertilizer schedule"
                    }

                    pred_key = str(raw_pred).strip().lower()
                    prediction = _label_map.get(pred_key, raw_pred)

                    return render_template("fertilizer.html", prediction=prediction)

            # ---------- INORGANIC branch (existing code path unchanged) ----------
            # handle inorganic (original model)
            crop = data.get("crop", "").lower().strip()
            pH = float(data.get("pH"))
            N = float(data.get("N"))
            P = float(data.get("P"))
            K = float(data.get("K"))

            fert_entry = MODEL_CACHE.get("fertilizer")
            if fert_entry is None:
                raise RuntimeError("Fertilizer model not loaded.")
            model = fert_entry["model"]
            label_encoder = fert_entry.get("label_encoder")
            crop_encoder = fert_entry.get("crop_encoder")

            try:
                crop_encoded = crop_encoder.transform([crop])[0] if crop_encoder is not None else 0
            except Exception:
                crop_encoded = 0

            input_data = np.array([[crop_encoded, pH, N, P, K]])
            pred_encoded = model.predict(input_data)
            # model.predict may return array shape (1,) or (1,1)
            pred_encoded = np.asarray(pred_encoded).reshape(-1)
            pred_val = pred_encoded[0]

            # decode predicted label (existing)
            if label_encoder is not None:
                raw_pred = label_encoder.inverse_transform([pred_val])[0]
            else:
                raw_pred = str(pred_val)

            # map model labels to user-friendly / substituted strings
            _label_map = {
                "fym": "urea dap mop combo",
                "other fertilizers": "Soil Amendment (Lime / Gypsum)",
                "routine fertilizer schedule": "Follow crop-specific fertilizer schedule"
            }

            # normalize key for lookup and fallback to original raw_pred when not found
            pred_key = str(raw_pred).strip().lower()
            prediction = _label_map.get(pred_key, raw_pred)

            return render_template("fertilizer.html", prediction=prediction)

        except Exception as e:
            return render_template_string(f"<pre>Error: {e}</pre>"), 500









# @app.route("/fertilizer/predict", methods=["POST"])
# def fertilizer_predict():
#     if "fertilizer" in LOAD_ERRORS:
#         return render_template_string(f"<pre>Load error: {LOAD_ERRORS['fertilizer']}</pre>"), 500
#     try:
#         data = request.form
#         # fertilizer type radio expected as 'fert_type' with values 'organic' or 'inorganic'
#         fert_type = data.get("fert_type", "inorganic").strip().lower()

#         # ---------- ORGANIC branch ----------
#         if fert_type == "organic":
#             organic_entry = MODEL_CACHE.get("fertilizer_organic")
#             if organic_entry is None:
#                 return render_template_string("<pre>Error: Organic model not available on server.</pre>"), 500

#             # Build payload expected by organic prepare_input helper
#             payload = {
#                 "Crop": data.get("crop", "").strip() or data.get("Crop", ""),
#                 "pH": data.get("pH", data.get("pH", 7.0)),
#                 "N_kg_per_ha": data.get("N", data.get("N_kg_per_ha", 0.0)),
#                 "P_kg_per_ha": data.get("P", data.get("P_kg_per_ha", 0.0)),
#                 "K_kg_per_ha": data.get("K", data.get("K_kg_per_ha", 0.0)),
#                 "Moisture_percent": data.get("moisture", data.get("Moisture_percent", 0.0))
#             }

#             # coerce numeric inputs
#             try:
#                 payload["pH"] = float(payload["pH"])
#             except Exception:
#                 payload["pH"] = 7.0
#             for k in ("N_kg_per_ha", "P_kg_per_ha", "K_kg_per_ha", "Moisture_percent"):
#                 try:
#                     payload[k] = float(payload[k])
#                 except Exception:
#                     payload[k] = 0.0

#             organic_model = organic_entry["model"]
#             organic_mlb = organic_entry["mlb"]
#             organic_feature_cols = organic_entry["feature_cols"]

#             # prepare features and predict
#             # NOTE: your codebase must provide prepare_input_organic(payload, feature_cols)
#             Xorg = prepare_input_organic(payload, organic_feature_cols)
#             y_pred = organic_model.predict(Xorg)
#             try:
#                 ferts = list(organic_mlb.inverse_transform(y_pred)[0])
#             except Exception:
#                 # fallback in case shape differs
#                 try:
#                     ferts = list(organic_mlb.inverse_transform(y_pred))
#                 except Exception:
#                     ferts = []

#             prediction = ", ".join(ferts) if ferts else "No recommendation"
#             return render_template("fertilizer.html", prediction=prediction)

#         # ---------- INORGANIC branch (existing code path unchanged) ----------
#         # else handle inorganic (original model)
#         crop = data.get("crop", "").lower().strip()
#         pH = float(data.get("pH"))
#         N = float(data.get("N"))
#         P = float(data.get("P"))
#         K = float(data.get("K"))

#         fert_entry = MODEL_CACHE.get("fertilizer")
#         if fert_entry is None:
#             raise RuntimeError("Fertilizer model not loaded.")
#         model = fert_entry["model"]
#         label_encoder = fert_entry.get("label_encoder")
#         crop_encoder = fert_entry.get("crop_encoder")

#         try:
#             crop_encoded = crop_encoder.transform([crop])[0] if crop_encoder is not None else 0
#         except Exception:
#             crop_encoded = 0

#         input_data = np.array([[crop_encoded, pH, N, P, K]])
#         pred_encoded = model.predict(input_data)
#         # model.predict may return array shape (1,) or (1,1)
#         pred_encoded = np.asarray(pred_encoded).reshape(-1)
#         pred_val = pred_encoded[0]

#         if label_encoder is not None:
#             raw_pred = label_encoder.inverse_transform([pred_val])[0]
#         else:
#             raw_pred = str(pred_val)

#         # user-friendly substitution map (keeps your mapping)
#         _label_map = {
#             "fym": "urea dap mop combo",
#             "other fertilizers": "Soil Amendment (Lime / Gypsum)",
#             "routine fertilizer schedule": "Follow crop-specific fertilizer schedule"
#         }

#         pred_key = str(raw_pred).strip().lower()
#         prediction = _label_map.get(pred_key, raw_pred)

#         return render_template("fertilizer.html", prediction=prediction)

#     except Exception as e:
#         return render_template_string(f"<pre>Error: {e}</pre>"), 500













# @app.route("/fertilizer/predict", methods=["POST"])
# def fertilizer_predict():
#     if "fertilizer" in LOAD_ERRORS:
#         return render_template_string(f"<pre>Load error: {LOAD_ERRORS['fertilizer']}</pre>"), 500
#     try:
#         data = request.form
#         crop = data.get("crop", "").lower().strip()
#         pH = float(data.get("pH"))
#         N = float(data.get("N"))
#         P = float(data.get("P"))
#         K = float(data.get("K"))

#         fert_entry = MODEL_CACHE.get("fertilizer")
#         if fert_entry is None:
#             raise RuntimeError("Fertilizer model not loaded.")
#         model = fert_entry["model"]
#         label_encoder = fert_entry.get("label_encoder")
#         crop_encoder = fert_entry.get("crop_encoder")

#         try:
#             crop_encoded = crop_encoder.transform([crop])[0] if crop_encoder is not None else 0
#         except Exception:
#             crop_encoded = 0

#         input_data = np.array([[crop_encoded, pH, N, P, K]])
#         pred_encoded = model.predict(input_data)
#         # model.predict may return array shape (1,) or (1,1)
#         pred_encoded = np.asarray(pred_encoded).reshape(-1)
#         pred_val = pred_encoded[0]
#         # prediction = label_encoder.inverse_transform([pred_val])[0] if label_encoder is not None else str(pred_val)
#         # return render_template("fertilizer.html", prediction=prediction)
        
#         # decode predicted label (existing)
#         if label_encoder is not None:
#             raw_pred = label_encoder.inverse_transform([pred_val])[0]
#         else:
#             raw_pred = str(pred_val)

#         # map model labels to user-friendly / substituted strings
#         _label_map = {
#             "fym": "urea dap mop combo",
#             "other fertilizers": "Soil Amendment (Lime / Gypsum / Micronutrients)",
#             "routine fertilizer schedule": "Follow crop-specific fertilizer schedule"
#         }

#         # normalize key for lookup and fallback to original raw_pred when not found
#         pred_key = str(raw_pred).strip().lower()
#         prediction = _label_map.get(pred_key, raw_pred)

#         return render_template("fertilizer.html", prediction=prediction)
    
#     except Exception as e:
#         return render_template_string(f"<pre>Error: {e}</pre>"), 500

# ---------- Weather endpoints (keep /predict for ajax) ----------
@app.route("/weather")
def weather_page():
    today_str = datetime.now().strftime("%A, %d %B %Y")
    return render_template("weather.html", today=today_str)

@app.route("/predict")
def predict():
    # use load_csv and model cache created earlier
    try:
        df = load_csv()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    out = {}

    # WIND
    if "wind" in LOAD_ERRORS:
        out["wind"] = {"error": LOAD_ERRORS["wind"]}
    else:
        model_wind = MODEL_CACHE.get("wind")
        if model_wind is None:
            out["wind"] = {"error": "wind models not loaded"}
        else:
            col = next((c for c in ["wind_speed_10m_max", "wind", "wind_speed"] if c in df.columns), None)
            if col is None:
                out["wind"] = {"error": "wind column not found in CSV"}
            else:
                try:
                    out["wind"] = iterative_forecast_models(model_wind, df[col], days=FORECAST_DAYS)
                except Exception as e:
                    out["wind"] = {"error": str(e)}

    # PRECIP
    if "precip" in LOAD_ERRORS:
        out["precip"] = {"error": LOAD_ERRORS["precip"]}
    else:
        model_prec = MODEL_CACHE.get("precip")
        if model_prec is None:
            out["precip"] = {"error": "precip models not loaded"}
        else:
            col = next((c for c in ["precipitation_sum", "precipitation", "prec", "y"] if c in df.columns), None)
            if col is None:
                out["precip"] = {"error": "precip column not found in CSV"}
            else:
                try:
                    out["precip"] = iterative_forecast_models(model_prec, df[col], days=FORECAST_DAYS)
                except Exception as e:
                    out["precip"] = {"error": str(e)}

    # TEMP
    if "temp" in LOAD_ERRORS:
        out["temp"] = {"error": LOAD_ERRORS["temp"]}
    else:
        temp_models = MODEL_CACHE.get("temp")
        if temp_models is None:
            out["temp"] = {"error": "temp models not loaded"}
        else:
            col_min = next((c for c in ["temperature_2m_min", "temp_min", "tmin"] if c in df.columns), None)
            col_max = next((c for c in ["temperature_2m_max", "temp_max", "tmax"] if c in df.columns), None)
            if col_min is None or col_max is None:
                out["temp"] = {"error": "temp_min/temp_max columns not found in CSV"}
            else:
                try:
                    out_min = iterative_forecast_models(temp_models["temp_min"], df[col_min], days=FORECAST_DAYS)
                    out_max = iterative_forecast_models(temp_models["temp_max"], df[col_max], days=FORECAST_DAYS)
                    out["temp"] = {"min": out_min, "max": out_max}
                except Exception as e:
                    out["temp"] = {"error": str(e)}

    return jsonify({"today": datetime.now().date().isoformat(), "forecast_days": FORECAST_DAYS, "data": out})

# ---------- Run ----------
if __name__ == "__main__":
    # debug True for development; change for production
    app.run(host="0.0.0.0", port=5000, debug=True)