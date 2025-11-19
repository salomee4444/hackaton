"""
Quick generator for an interactive 3D Plotly HTML with simple filters.

What it does:
- Loads `avalon_nuclear.csv` from the same folder
- Trains a quick RandomForest (class_weight='balanced') as a fallback to get P(panic)
- Builds a Plotly 3D scatter where:
  - X = public_anxiety_index, Y = social_media_rumour_index, Z = regulator_scrutiny_score
  - point size = population_within_30km, hover shows meta
  - color = predicted probability of panic
  - a dropdown lets you filter to top-8 countries ("All" available)
  - a slider (τ) updates which points are highlighted (P >= τ)

Outputs: `3d_interactive_with_filters.html`

This is a fast, self-contained fallback: it does not require shap or imbalanced-learn.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

RANDOM_STATE = 42


def load_and_prepare(csv_path='avalon_nuclear.csv'):
    df = pd.read_csv(csv_path)
    # define panic_mode as in pipeline if not present
    if 'panic_mode' not in df.columns:
        # safe fallback: use any recommendation flags combined with true risk <=2 when available
        if 'true_risk_level' in df.columns:
            df['panic_mode'] = (((df.get('avalon_evac_recommendation', 0) == 1) | (df.get('avalon_shutdown_recommendation', 0) == 1)) & (df['true_risk_level'] <= 2)).astype(int)
        else:
            df['panic_mode'] = (((df.get('avalon_evac_recommendation', 0) == 1) | (df.get('avalon_shutdown_recommendation', 0) == 1))).astype(int)

    # features to use for quick model
    feat_num = ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score', 'population_within_30km']
    for c in feat_num:
        if c not in df.columns:
            df[c] = np.nan

    return df, feat_num


def train_quick_model(df, feat_num):
    X = df[feat_num].copy()
    y = df['panic_mode'].copy()
    # simple pipeline: median impute + scale + RF
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(n_estimators=150, max_depth=8, n_jobs=-1, random_state=RANDOM_STATE, class_weight='balanced'))
    # drop rows with nan target
    mask = y.notna()
    pipe.fit(X[mask], y[mask])
    proba = pipe.predict_proba(X.fillna(X.median()))[:, 1]
    return proba, pipe


def build_plot(df, proba, out_html='3d_interactive_with_filters.html'):
    df = df.copy()
    df['proba_panic'] = proba
    df['pop_size'] = (df['population_within_30km'].fillna(0) / (df['population_within_30km'].max() + 1)) * 30 + 5

    # pick top countries to include as direct buttons
    top_countries = list(df['country'].value_counts().head(8).index)
    df['country_group'] = df['country'].where(df['country'].isin(top_countries), 'Other')
    country_keys = ['All'] + top_countries + ['Other']

    # create one trace per country_group
    traces = []
    groups_sorted = ['All'] + [g for g in country_keys if g in df['country_group'].unique() and g != 'All']

    for g in groups_sorted:
        if g == 'All':
            sub = df
        else:
            sub = df[df['country_group'] == g]
        trace = go.Scatter3d(
            x=sub['public_anxiety_index'],
            y=sub['social_media_rumour_index'],
            z=sub['regulator_scrutiny_score'],
            mode='markers',
            marker=dict(
                size=sub['pop_size'],
                color=sub['proba_panic'],
                colorscale='RdYlBu_r',
                cmin=0, cmax=1,
                colorbar=dict(title='P(panic)'),
                opacity=0.9,
                showscale=(g == 'All')
            ),
            hovertemplate=(
                'Country: %{customdata[0]}<br>'
                'Reactor type: %{customdata[1]}<br>'
                'Public anxiety: %{x:.1f}<br>'
                'Social rumour: %{y:.1f}<br>'
                'Regulator scrutiny: %{z:.1f}<br>'
                'Population: %{customdata[2]:.0f}<br>'
                'P(panic): %{marker.color:.2f}<br>'
                'panic_mode: %{customdata[3]}<extra></extra>'
            ),
            customdata=np.stack([sub['country'], sub.get('reactor_type_code', pd.Series(['NA']*len(sub))), sub['population_within_30km'].fillna(0), sub['panic_mode']], axis=1)
        )
        traces.append(trace)

    # Build frames for tau slider (21 steps)
    taus = np.linspace(0.0, 1.0, 21)
    frames = []
    for tau in taus:
        data_for_frame = []
        for g in groups_sorted:
            if g == 'All':
                sub = df
            else:
                sub = df[df['country_group'] == g]
            colors = ['red' if p >= tau else 'rgba(100,100,200,0.4)' for p in sub['proba_panic']]
            marker = dict(size=sub['pop_size'], color=colors)
            data_for_frame.append(go.Scatter3d(marker=marker))
        frames.append(go.Frame(data=data_for_frame, name=f"tau_{tau:.2f}"))

    layout = go.Layout(
        title='Interactive 3D: External signals (color=proba, size=population)',
        scene=dict(
            xaxis=dict(title='public_anxiety_index'),
            yaxis=dict(title='social_media_rumour_index'),
            zaxis=dict(title='regulator_scrutiny_score')
        ),
        updatemenus=[
            dict(
                type='dropdown',
                buttons=[
                    dict(label='All', method='update', args=[{'visible': [True]*len(traces)}, {'title': 'All countries'}])
                ] + [
                    dict(label=ct, method='update', args=[
                        {'visible': [ct == g for g in groups_sorted]},
                        {'title': f'Country = {ct}'}
                    ])
                    for ct in groups_sorted if ct != 'All'
                ],
                x=0, y=1.12
            )
        ],
        sliders=[{
            'active': 10,
            'currentvalue': {'prefix': 'τ = '},
            'pad': {'t': 50},
            'steps': [
                {'label': f'{tau:.2f}', 'method': 'animate', 'args': [[f'tau_{tau:.2f}'], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}]}
                for tau in taus
            ]
        }]
    )

    fig = go.Figure(data=traces, layout=layout, frames=frames)

    out_path = Path(out_html)
    fig.write_html(str(out_path), full_html=True, include_plotlyjs='cdn')
    print(f"Saved interactive plot to {out_path.resolve()}")


def main():
    df, feat_num = load_and_prepare('avalon_nuclear.csv')
    proba, model = train_quick_model(df, feat_num)
    build_plot(df, proba)


if __name__ == '__main__':
    main()
