from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'evaluation_metrics.json')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.json')
DATA_FILES = ['data/Crop_recommendation.csv', 'data/Crop_recommendation1.csv']

def load_model_and_scaler():
    try:
        logger.info("Loading model and scaler...")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            logger.warning("Model or scaler files not found.")
            return None, None
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Loaded model: {type(model)}, scaler: {type(scaler)}")
        return model, scaler
    except Exception as e:
        logger.error(f"Failed to load model or scaler: {e}", exc_info=True)
        return None, None

def add_features(df):
    try:
        df = df.copy()
        df['NP_ratio'] = df['N'] / df['P']
        df['NK_ratio'] = df['N'] / df['K']
        df['PK_ratio'] = df['P'] / df['K']
        df['temperature_squared'] = df['temperature'] ** 2
        df['humidity_squared'] = df['humidity'] ** 2
        df['rainfall_squared'] = df['rainfall'] ** 2
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['temp_rainfall'] = df['temperature'] * df['rainfall']

        feature_columns = [
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
            'NP_ratio', 'NK_ratio', 'PK_ratio',
            'temperature_squared', 'humidity_squared', 'rainfall_squared',
            'temp_humidity', 'temp_rainfall'
        ]

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for features: {missing}")
        return df[feature_columns]
    except Exception as e:
        logger.error(f"Error in add_features: {e}", exc_info=True)
        raise

def train_model():
    logger.info("Starting model training...")
    try:
        dfs = []
        for file in DATA_FILES:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Training data file not found: {file}")
            dfs.append(pd.read_csv(file))
        df = pd.concat(dfs, ignore_index=True).drop_duplicates()

        logger.info(f"Loaded combined dataset with {len(df)} samples and {df['label'].nunique()} unique crops.")

        X = add_features(df)
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save feature names
        with open(FEATURE_NAMES_PATH, 'w') as f:
            json.dump(X.columns.tolist(), f)

        param_grid = {
            'n_estimators': [300, 400, 500],
            'max_depth': [20, 25, 30],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None]
        }

        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=10)

        feature_importance = dict(zip(X.columns, best_model.feature_importances_))
        feature_importance_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        metrics = {
            'accuracy': accuracy,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance_sorted,
            'best_parameters': grid_search.best_params_,
            'dataset_info': {
                'total_samples': len(df),
                'unique_crops': int(df['label'].nunique()),
                'crop_distribution': df['label'].value_counts().to_dict()
            }
        }

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f)

        logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")

        return metrics
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise

# Load or train model on startup
model, scaler = load_model_and_scaler()
if model is None or scaler is None:
    logger.info("Model/scaler missing, starting training...")
    train_model()
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        logger.error("Failed to load model after training. Exiting.")
        exit(1)

@app.route('/')
def home():
    try:
        logger.info("Serving home page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {e}", exc_info=True)
        return "Internal Server Error", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503

        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'JSON body required'}), 400

        data = request.get_json()
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing field: {field}'}), 400
            try:
                float(data[field])
            except ValueError:
                return jsonify({'status': 'error', 'message': f'Invalid value for {field}'}), 400

        features = pd.DataFrame([[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall']),
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        features = add_features(features)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [{'crop': model.classes_[i], 'confidence': float(probabilities[i])} for i in top_indices]

        return jsonify({
            'status': 'success',
            'best_match': {'crop': prediction, 'confidence': float(np.max(probabilities))},
            'predictions': top_predictions
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Prediction failed'}), 500

@app.route('/model-metrics')
def model_metrics():
    try:
        if not os.path.exists(METRICS_PATH):
            return jsonify({'status': 'error', 'message': 'Metrics file not found'}), 404
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error reading metrics: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Failed to read metrics'}), 500

if __name__ == '__main__':
    # Allow configurable debug mode and port for easy deployment
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
