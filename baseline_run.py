import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report, precision_recall_curve, roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv(r'c:\Users\salom\OneDrive\Bureau\hackaton\avalon_nuclear.csv')
except Exception as e:
    print('Could not read CSV:', e)
    raise

# define panic_mode
df['panic_mode'] = ( ((df['avalon_evac_recommendation']==1) & (df['true_risk_level'].isin([0,1,2]))) | \
                   ((df['avalon_shutdown_recommendation']==1) & (df['true_risk_level'].isin([0,1,2]))) ).astype(int)

numeric = ['public_anxiety_index','social_media_rumour_index','regulator_scrutiny_score','reactor_age_years',
           'reactor_nominal_power_mw','population_within_30km','seismic_activity_index','cyber_attack_score']
cat = ['country','reactor_type_code']

X = df[numeric + cat]
y = df['panic_mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# handle OneHotEncoder API differences
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', ohe)
])

preproc = ColumnTransformer([
    ('num', num_pipe, numeric),
    ('cat', cat_pipe, cat)
], remainder='drop')

X_train_t = preproc.fit_transform(X_train)
X_test_t = preproc.transform(X_test)

clf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(X_train_t, y_train)

y_proba = clf.predict_proba(X_test_t)[:,1]
y_pred = (y_proba >= 0.5).astype(int)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_proba))
print('Average Precision (PR AUC):', average_precision_score(y_test, y_proba))
print('\nClassification report:')
print(classification_report(y_test, y_pred, zero_division=0))

# save confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Pred')
ax.set_ylabel('True')
fig.savefig('baseline_confusion_matrix.png')

# save ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig = plt.figure()
plt.plot(fpr, tpr, label='ROC')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend()
plt.savefig('baseline_roc.png')

# save PR
prec, rec, _ = precision_recall_curve(y_test, y_proba)
fig = plt.figure()
plt.plot(rec, prec, label='PR')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.legend()
plt.savefig('baseline_pr.png')

print('Saved baseline_confusion_matrix.png, baseline_roc.png, baseline_pr.png')
