import os
import pandas as pd
import plotly.express as px

ROOT = os.path.dirname(__file__)
CSV = os.path.join(ROOT, 'avalon_nuclear.csv')

def make_panic_label(df):
    df = df.copy()
    df['panic_mode'] = ((df['avalon_evac_recommendation'] == 1) | (df['avalon_shutdown_recommendation'] == 1)) & (df['true_risk_level'] <= 2)
    df['panic_mode'] = df['panic_mode'].astype(int)
    return df

def main():
    df = pd.read_csv(CSV)
    df = make_panic_label(df)

    fig = px.scatter_3d(
        df,
        x='public_anxiety_index',
        y='social_media_rumour_index',
        z='regulator_scrutiny_score',
        color=df['panic_mode'].map({0:'no_panic',1:'panic'}),
        size='population_within_30km',
        hover_data=['country','reactor_type_code','true_risk_level','avalon_raw_risk_score'],
        title='Interactive 3D: External signals (color=panic_mode, size=population)'
    )

    out = os.path.join(ROOT, '3d_interactive.html')
    fig.write_html(out, include_plotlyjs='cdn')
    print('Saved interactive 3D plot to', out)

if __name__ == '__main__':
    main()
