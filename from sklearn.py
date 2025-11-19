"""
Comprehensive pipeline for 'panic_mode' modelling and tipping-point analysis.

This script will:
 - load the dataset `avalon_nuclear.csv` (expected in the same folder)
 - perform light EDA and define the 'panic_mode' target
 - build a preprocessing pipeline (imputation, scaling, one-hot encoding)
 - handle class imbalance with SMOTE inside an imblearn Pipeline
 - tune a GradientBoostingClassifier with RandomizedSearchCV
 - evaluate with PR AUC / ROC AUC and choose an operational threshold
 - compute SHAP explanations (if shap is available) and estimate tipping points
 - save the best model and a CSV with predicted probabilities for inspection

Comments are in English. Adjust paths and hyperparameters as needed.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Visualization imports (optional for EDA) ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Scikit-learn and imbalanced-learn ---
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, precision_recall_curve,
                             classification_report)

from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint, uniform

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib

# Try to import shap (optional). If not available, continue without shap outputs.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def load_data(csv_path='avalon_nuclear.csv'):
    """Load dataset from CSV file path (default expects file in script folder).
    Returns a pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Update path accordingly.")
    df = pd.read_csv(csv_path)
    return df


def define_panic_mode(df):
    """Define 'panic_mode' = 1 when Avalon recommends evacuation or shutdown
    while true_risk_level is low (0,1,2). Otherwise 0.
    """
    cond_evac = (df['avalon_evac_recommendation'] == 1) & (df['true_risk_level'].isin([0, 1, 2]))
    cond_shutdown = (df['avalon_shutdown_recommendation'] == 1) & (df['true_risk_level'].isin([0, 1, 2]))
    df['panic_mode'] = (cond_evac | cond_shutdown).astype(int)
    return df


def build_preprocessor(numeric_features, categorical_features):
    """Build a ColumnTransformer preprocessor for numeric and categorical features.
    Numeric: median imputation + StandardScaler
    Categorical: constant imputation + OneHotEncoder
    """
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Build OneHotEncoder in a way that's compatible with multiple sklearn versions
    # (older versions use `sparse=False`, newer use `sparse_output=False`).
    from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
    ohe_kwargs = {'handle_unknown': 'ignore'}
    try:
        # Try the older kwarg first
        _OneHotEncoder(**{**ohe_kwargs, 'sparse': False})
        ohe = _OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # Fallback for newer sklearn versions
        ohe = _OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', ohe)
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ], remainder='drop')

    return preprocessor


def get_feature_names(preprocessor, numeric_features, categorical_features):
    """Produce transformed feature names after ColumnTransformer and OneHotEncoder.
    Note: requires scikit-learn 1.0+ for get_feature_names_out on OneHotEncoder.
    """
    cat_ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    try:
        cat_names = list(cat_ohe.get_feature_names_out(categorical_features))
    except Exception:
        # Older sklearn: fallback to manual names
        cat_categories = cat_ohe.categories_
        cat_names = []
        for col, cats in zip(categorical_features, cat_categories):
            cat_names += [f"{col}_{str(c)}" for c in cats]

    feature_names = list(numeric_features) + cat_names
    return feature_names


def estimate_threshold(feature_name, X_reference, pipeline, model_step_name='clf', preproc_step_name='preproc', grid=None, prob_threshold=0.5):
    """Estimate tipping point threshold for a single feature by varying it while fixing
    other variables to medians. Returns (vals, probs, tipping_value_or_None).
    pipeline: the imblearn Pipeline (fitted). X_reference: DataFrame (train/valid)
    """
    if grid is None:
        vals = np.linspace(X_reference[feature_name].min(), X_reference[feature_name].max(), 200)
    else:
        vals = np.asarray(grid)

    median_row = X_reference.median(numeric_only=True).to_frame().T
    # replicate median row
    X_tmp = pd.concat([median_row]*len(vals), ignore_index=True)
    X_tmp[feature_name] = vals

    # transform with internal preprocessor
    preproc = pipeline.named_steps[preproc_step_name]
    X_trans = preproc.transform(X_tmp)
    clf = pipeline.named_steps[model_step_name]
    probs = clf.predict_proba(X_trans)[:, 1]
    idx = np.where(probs >= prob_threshold)[0]
    tipping_value = vals[idx[0]] if len(idx) > 0 else None
    return vals, probs, tipping_value


def main():
    # 1. Load data
    df = load_data('avalon_nuclear.csv')

    # 2. Light EDA (printouts)
    print('Data shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('\nFirst 5 rows:')
    print(df.head())
    print('\nInfo:')
    print(df.info())
    print('\nDescribe:')
    print(df.describe())

    # 3. Define panic_mode target
    df = define_panic_mode(df)
    print('\npanic_mode value counts:')
    print(df['panic_mode'].value_counts())

    # 4. Feature selection (adjust lists according to your dataset)
    numerical_features = [
        'public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score',
        'reactor_age_years', 'reactor_nominal_power_mw', 'population_within_30km',
        'seismic_activity_index', 'cyber_attack_score'
    ]
    categorical_features = ['country', 'reactor_type_code']

    # Ensure selected features exist in DataFrame
    for col in numerical_features + categorical_features:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataset.")

    X = df[numerical_features + categorical_features].copy()
    y = df['panic_mode'].copy()

    # 5. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # 6. Preprocessor and imbalanced pipeline
    preproc = build_preprocessor(numerical_features, categorical_features)

    # Imbalanced-learn pipeline: preproc -> SMOTE -> classifier
    gbc = GradientBoostingClassifier(random_state=RANDOM_STATE)

    pipeline = ImbPipeline([
        ('preproc', preproc),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', gbc)
    ])

    # 7. RandomizedSearchCV for GradientBoosting (parameters use 'clf__' prefix)
    param_dist = {
        'clf__n_estimators': randint(100, 400),
        'clf__learning_rate': uniform(0.01, 0.2),
        'clf__max_depth': randint(3, 8),
        'clf__subsample': uniform(0.7, 0.3)
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
    n_iter=12,
        cv=cv,
        scoring='average_precision',  # PR AUC is useful for imbalanced classes
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # 8. Fit randomized search
    print('\nFitting RandomizedSearchCV (this may take a while)...')
    random_search.fit(X_train, y_train)
    print('\nBest params:', random_search.best_params_)
    print('Best CV avg precision (PR AUC):', random_search.best_score_)

    best_pipeline = random_search.best_estimator_

    # 9. Evaluate on test set
    # transform test set and predict probabilities
    X_test_trans = best_pipeline.named_steps['preproc'].transform(X_test)
    clf = best_pipeline.named_steps['clf']
    y_proba = clf.predict_proba(X_test_trans)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    print('\nTest ROC AUC:', roc_auc_score(y_test, y_proba))
    print('Test Average Precision (PR AUC):', average_precision_score(y_test, y_proba))
    print('\nClassification report (threshold=0.5):')
    print(classification_report(y_test, y_pred_default, zero_division=0))

    # 10. Choose threshold using F1 on PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    # thresholds has length len(precision)-1, protect indexing
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = 0.5
    print(f"Selected threshold (max F1 on PR curve): {best_thresh:.4f}")
    y_pred_tuned = (y_proba >= best_thresh).astype(int)
    print('\nClassification report (tuned threshold):')
    print(classification_report(y_test, y_pred_tuned, zero_division=0))

    # --- Plots for model evaluation (save PNGs for presentation) ---
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
    from sklearn.calibration import calibration_curve

    # 1) Confusion matrix (tuned threshold)
    cm = confusion_matrix(y_test, y_pred_tuned)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax_cm)
    ax_cm.set_title('Confusion Matrix (threshold={:.2f})'.format(best_thresh))
    fig_cm.tight_layout()
    cm_path = 'confusion_matrix.png'
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)
    print(f"Saved confusion matrix to {cm_path}")

    # 2) ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax_roc.plot([0, 1], [0, 1], '--', color='gray')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    fig_roc.tight_layout()
    roc_path = 'roc_curve.png'
    fig_roc.savefig(roc_path)
    plt.close(fig_roc)
    print(f"Saved ROC curve to {roc_path}")

    # 3) Precision-Recall curve
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.plot(rec, prec, label=f'AP = {ap:.3f}')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(loc='lower left')
    fig_pr.tight_layout()
    pr_path = 'precision_recall_curve.png'
    fig_pr.savefig(pr_path)
    plt.close(fig_pr)
    print(f"Saved Precision-Recall curve to {pr_path}")

    # 4) Calibration plot
    try:
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
        ax_cal.plot(prob_pred, prob_true, marker='o', linewidth=1)
        ax_cal.plot([0, 1], [0, 1], '--', color='gray')
        ax_cal.set_xlabel('Mean predicted probability')
        ax_cal.set_ylabel('Fraction of positives')
        ax_cal.set_title('Calibration Curve')
        fig_cal.tight_layout()
        cal_path = 'calibration_curve.png'
        fig_cal.savefig(cal_path)
        plt.close(fig_cal)
        print(f"Saved calibration curve to {cal_path}")
    except Exception as e:
        print('Calibration plot failed:', e)

    # --- Additional visuals for presentation ---
    try:
        # 5) Feature importance bar chart (top 20)
        if 'importances' in locals():
            fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
            importances.head(20).sort_values().plot.barh(ax=ax_fi)
            ax_fi.set_title('Top 20 Feature Importances')
            fig_fi.tight_layout()
            fi_path = 'feature_importances.png'
            fig_fi.savefig(fi_path)
            plt.close(fig_fi)
            print(f"Saved feature importance plot to {fi_path}")

        # 6) 3D scatter: public_anxiety_index, social_media_rumour_index, predicted probability
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig3d = plt.figure(figsize=(9, 7))
            ax3d = fig3d.add_subplot(111, projection='3d')
            x = X_test['public_anxiety_index'].values
            y3 = X_test['social_media_rumour_index'].values
            z = y_proba
            cmap = plt.get_cmap('viridis')
            sc = ax3d.scatter(x, y3, z, c=z, cmap=cmap, s=20, alpha=0.8)
            ax3d.set_xlabel('Public Anxiety Index')
            ax3d.set_ylabel('Social Media Rumour Index')
            ax3d.set_zlabel('Predicted Probability (panic_mode)')
            fig3d.colorbar(sc, ax=ax3d, label='Pred Proba')
            fig3d.tight_layout()
            scatter3d_path = '3d_scatter_predprob.png'
            fig3d.savefig(scatter3d_path)
            plt.close(fig3d)
            print(f"Saved 3D scatter to {scatter3d_path}")
        except Exception as e:
            print('3D scatter failed:', e)

        # 7) Pairplot for key external features colored by panic_mode (sampled to speed up)
        try:
            sample_plot = df[[
                'public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score',
                'panic_mode'
            ]].sample(n=min(1000, len(df)), random_state=RANDOM_STATE)
            sns.pairplot(sample_plot, hue='panic_mode', corner=True)
            pairplot_path = 'pairplot_external_features.png'
            plt.savefig(pairplot_path)
            plt.close()
            print(f"Saved pairplot to {pairplot_path}")
        except Exception as e:
            print('Pairplot failed:', e)

        # 8) Boxplots of external factors by panic mode
        try:
            fig_box, axes_box = plt.subplots(1, 3, figsize=(18, 5))
            for i, feat in enumerate(['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']):
                sns.boxplot(data=df, x='panic_mode', y=feat, ax=axes_box[i])
                axes_box[i].set_title(f'{feat} by panic_mode')
                axes_box[i].set_xlabel('panic_mode')
            fig_box.tight_layout()
            boxplot_path = 'boxplots_external_by_panic.png'
            fig_box.savefig(boxplot_path)
            plt.close(fig_box)
            print(f"Saved boxplots to {boxplot_path}")
        except Exception as e:
            print('Boxplots failed:', e)

        # 9) Country-level violin plots for external factors (top 8 countries by samples)
        try:
            country_counts = df['country'].value_counts()
            top_countries = country_counts.head(8).index.tolist()
            sub = df[df['country'].isin(top_countries)].copy()
            fig_v, axes_v = plt.subplots(3, 1, figsize=(14, 12))
            for i, feat in enumerate(['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']):
                sns.violinplot(data=sub, x='country', y=feat, ax=axes_v[i], inner='quartile')
                axes_v[i].tick_params(axis='x', rotation=45)
                axes_v[i].set_title(f'{feat} by Country (top 8 countries)')
            fig_v.tight_layout()
            violins_path = 'violin_top_countries.png'
            fig_v.savefig(violins_path)
            plt.close(fig_v)
            print(f"Saved violin plots to {violins_path}")
        except Exception as e:
            print('Violin plots failed:', e)

    except Exception as e:
        print('Additional visuals failed:', e)


    # 11. Feature importance (map back to original feature names)
    try:
        feature_names = get_feature_names(best_pipeline.named_steps['preproc'], numerical_features, categorical_features)
        importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
        print('\nTop 20 feature importances:')
        print(importances.head(20))
    except Exception as e:
        print('Could not compute feature importances mapping:', e)

    # 12. SHAP explanations (optional)
    if SHAP_AVAILABLE:
        try:
            print('\nComputing SHAP values (may take some time)...')
            # Use a small background sample to speed up
            background = X_train.sample(n=min(500, len(X_train)), random_state=RANDOM_STATE)
            background_trans = best_pipeline.named_steps['preproc'].transform(background)
            explainer = shap.TreeExplainer(clf)
            # compute SHAP on transformed test sample
            sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
            sample_trans = best_pipeline.named_steps['preproc'].transform(sample)
            shap_values = explainer.shap_values(sample_trans)
            # summary plot
            try:
                shap.summary_plot(shap_values, sample_trans, feature_names=feature_names)
            except Exception:
                pass
        except Exception as e:
            print('Error while computing SHAP:', e)
    else:
        print('\nSHAP package not available; skip SHAP explanation step.')

    # 13. Estimate tipping points for chosen external factors
    for feat in ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']:
        try:
            vals, probs, tip = estimate_threshold(feat, X_train, best_pipeline, model_step_name='clf', preproc_step_name='preproc', prob_threshold=best_thresh)
            print(f"Tipping point for {feat}: {tip}")
        except Exception as e:
            print(f"Could not estimate tipping point for {feat}: {e}")

    # Additional analyses for presentation: per-country and per-reactor tipping points,
    # panic_mode frequencies, and optional SHAP-based archetype clustering.

    # Print overall panic_mode statistics
    total_cases = len(df)
    panic_counts = df['panic_mode'].sum()
    print(f"\nOverall panic_mode: {panic_counts} / {total_cases} ({panic_counts/total_cases:.3%})")

    # Cross-tabulations
    print('\nPanic mode counts by country (top 20):')
    ct_country = pd.crosstab(df['country'], df['panic_mode'])
    ct_country['panic_rate'] = ct_country.get(1, 0) / ct_country.sum(axis=1)
    print(ct_country.sort_values('panic_rate', ascending=False).head(20))

    print('\nPanic mode counts by reactor_type_code:')
    ct_reactor = pd.crosstab(df['reactor_type_code'], df['panic_mode'])
    ct_reactor['panic_rate'] = ct_reactor.get(1, 0) / ct_reactor.sum(axis=1)
    print(ct_reactor.sort_values('panic_rate', ascending=False))

    # Per-country tipping points (only for countries with enough samples)
    countries = X_train['country'].unique()
    tipping_by_country = []
    for country in countries:
        sub = X_train[X_train['country'] == country]
        if len(sub) < 30:
            continue
        row = {'country': country}
        for feat in ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']:
            try:
                _, _, tip = estimate_threshold(feat, sub, best_pipeline, prob_threshold=best_thresh)
                row[f'tipping_{feat}'] = tip
            except Exception:
                row[f'tipping_{feat}'] = None
        tipping_by_country.append(row)
    tipping_country_df = pd.DataFrame(tipping_by_country)
    if not tipping_country_df.empty:
        tipping_country_df.to_csv('tipping_points_by_country.csv', index=False)
        print('\nSaved tipping points by country to tipping_points_by_country.csv')
        print(tipping_country_df.head(10))
    else:
        print('\nNo country-level tipping points computed (insufficient data per country).')

    # Per-reactor-type tipping points
    reactor_types = X_train['reactor_type_code'].unique()
    tipping_by_reactor = []
    for rtype in reactor_types:
        sub = X_train[X_train['reactor_type_code'] == rtype]
        if len(sub) < 30:
            continue
        row = {'reactor_type_code': rtype}
        for feat in ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']:
            try:
                _, _, tip = estimate_threshold(feat, sub, best_pipeline, prob_threshold=best_thresh)
                row[f'tipping_{feat}'] = tip
            except Exception:
                row[f'tipping_{feat}'] = None
        tipping_by_reactor.append(row)
    tipping_reactor_df = pd.DataFrame(tipping_by_reactor)
    if not tipping_reactor_df.empty:
        tipping_reactor_df.to_csv('tipping_points_by_reactor_type.csv', index=False)
        print('\nSaved tipping points by reactor type to tipping_points_by_reactor_type.csv')
        print(tipping_reactor_df.head(10))
    else:
        print('\nNo reactor-type-level tipping points computed (insufficient data per type).')

    # SHAP-based archetype clustering (if available)
    if SHAP_AVAILABLE:
        try:
            print('\nRunning SHAP-based archetype clustering...')
            # Compute SHAP values on a sample of training data
            sample_X = X_train.sample(n=min(1000, len(X_train)), random_state=RANDOM_STATE)
            sample_trans = best_pipeline.named_steps['preproc'].transform(sample_X)
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(sample_trans)

            # Ensure shap_vals is array-like [n_samples, n_features]
            S = np.array(shap_vals)
            if S.ndim == 3:  # multiclass returns list; pick class 1 if present
                S = S[1]

            # cluster
            from sklearn.cluster import KMeans
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE).fit(S)
            labels = kmeans.labels_
            sample_out = sample_X.reset_index(drop=True).copy()
            sample_out['shap_cluster'] = labels

            # Describe clusters by mean of external features
            cluster_desc = sample_out.groupby('shap_cluster')[[
                'public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score',
                'population_within_30km'
            ]].mean()
            print('\nSHAP cluster description (means):')
            print(cluster_desc)
            sample_out.to_csv('shap_cluster_assignments.csv', index=False)
            cluster_desc.to_csv('shap_cluster_summary.csv')
            print('\nSaved SHAP cluster assignments and summary to CSV files.')
        except Exception as e:
            print('Error during SHAP clustering:', e)
    else:
        print('\nSHAP not available — skipping archetype clustering.')

    # 14. Save the best pipeline and a CSV with probabilities for manual inspection
    out_model_path = 'best_panic_pipeline.joblib'
    joblib.dump(best_pipeline, out_model_path)
    print(f"Saved best pipeline to {out_model_path}")

    # Save test set predictions and probabilities together with original meta columns
    preds_df = X_test.copy().reset_index(drop=True)
    preds_df['y_true'] = y_test.reset_index(drop=True)
    preds_df['y_proba'] = y_proba
    preds_df['y_pred_tuned'] = y_pred_tuned
    inspect_path = 'panic_predictions_inspect.csv'
    preds_df.to_csv(inspect_path, index=False)
    print(f"Saved test predictions to {inspect_path}")


if __name__ == '__main__':
    main()
