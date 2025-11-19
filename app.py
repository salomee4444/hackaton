"""Simple Flask app to serve the interactive HTML and provide a predict endpoint
for the front-end to request probability values from a trained pipeline.

Behavior:
- Serves `index.html` at `/` (index.html should be at repo root)
- POST /predict-grid accepts a JSON payload with a list of grid points and returns P(panic)
- On startup, tries to load `best_panic_pipeline.joblib`; if missing, trains a small fallback model
  from `avalon_nuclear.csv` and saves it as `best_panic_pipeline.joblib`.

This keeps the service self-contained for Render deployments.
"""

import os
import traceback
from flask import Flask, send_from_directory, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='.')

MODEL_PATH = 'best_panic_pipeline.joblib'
TRAIN_CSV = 'avalon_nuclear.csv'


def train_fallback_model():
    """Train a quick RandomForest pipeline if no saved model exists."""
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        df = pd.read_csv(TRAIN_CSV)
        # define panic_mode if missing
        if 'panic_mode' not in df.columns:
            if 'true_risk_level' in df.columns:
                df['panic_mode'] = (((df.get('avalon_evac_recommendation', 0) == 1) | (df.get('avalon_shutdown_recommendation', 0) == 1)) & (df['true_risk_level'] <= 2)).astype(int)
            else:
                df['panic_mode'] = (((df.get('avalon_evac_recommendation', 0) == 1) | (df.get('avalon_shutdown_recommendation', 0) == 1))).astype(int)

        features = ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score', 'population_within_30km']
        for c in features:
            if c not in df.columns:
                df[c] = 0

        X = df[features].fillna(df[features].median())
        y = df['panic_mode'].fillna(0).astype(int)

        pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
        pipe.fit(X, y)
        joblib.dump(pipe, MODEL_PATH)
        return pipe
    except Exception:
        traceback.print_exc()
        return None


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception:
            traceback.print_exc()
    # fallback: train if csv exists
    if os.path.exists(TRAIN_CSV):
        return train_fallback_model()
    return None


model = load_or_train_model()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/predict-grid', methods=['POST'])
def predict_grid():
    if model is None:
        return jsonify({'error': 'Model not available on server.'}), 500

    payload = request.get_json(force=True)
    grid = payload.get('grid')
    if grid is None:
        return jsonify({'error': 'Missing grid in JSON body.'}), 400

    try:
        df_grid = pd.DataFrame(grid)
        # Ensure expected columns exist
        expected = ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score', 'population_within_30km']
        for c in expected:
            if c not in df_grid.columns:
                df_grid[c] = 0

        # If model is a pipeline with preproc and classifier inside, handle accordingly
        try:
            # many pipelines are imblearn Pipeline with named steps
            if hasattr(model, 'named_steps') and 'preproc' in model.named_steps and 'clf' in model.named_steps:
                preproc = model.named_steps['preproc']
                clf = model.named_steps['clf']
                X_trans = preproc.transform(df_grid[expected])
                probas = clf.predict_proba(X_trans)[:, 1].tolist()
            else:
                # assume estimator accepts raw numeric array
                probas = model.predict_proba(df_grid[expected].fillna(0))[:, 1].tolist()
        except Exception:
            # fallback: try pipeline predict_proba directly
            probas = model.predict_proba(df_grid[expected].fillna(0))[:, 1].tolist()

        return jsonify({'probas': probas})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
