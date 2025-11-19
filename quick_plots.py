import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Try to use seaborn if available for nicer plots
try:
    import seaborn as sns
    sns.set(style='whitegrid')
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

ROOT = os.path.dirname(__file__)
CSV = os.path.join(ROOT, 'avalon_nuclear.csv')

def make_panic_label(df):
    # quick heuristic: panic if Avalon recommends evacuation or shutdown while true_risk_level is low/moderate
    df = df.copy()
    df['panic_mode'] = ((df['avalon_evac_recommendation'] == 1) | (df['avalon_shutdown_recommendation'] == 1)) & (df['true_risk_level'] <= 2)
    df['panic_mode'] = df['panic_mode'].astype(int)
    return df

def save_hist(df, col, fname):
    plt.figure(figsize=(6,4))
    if HAS_SEABORN:
        sns.histplot(df[col].dropna(), kde=True, color='C0', bins=35)
    else:
        plt.hist(df[col].dropna(), bins=35, color='C0', alpha=0.8)
    plt.title(col)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_box_by_panic(df, cols, fname):
    plt.figure(figsize=(8,5))
    if HAS_SEABORN:
        melted = df.melt(id_vars='panic_mode', value_vars=cols, var_name='feature', value_name='value')
        sns.boxplot(x='feature', y='value', hue='panic_mode', data=melted)
        plt.legend(title='panic_mode')
    else:
        # simple grouped boxplots
        data = [df[df['panic_mode']==0][c].dropna() for c in cols]
        data2 = [df[df['panic_mode']==1][c].dropna() for c in cols]
        plt.boxplot(data, positions=np.arange(len(cols))*2.0-0.4, widths=0.6)
        plt.boxplot(data2, positions=np.arange(len(cols))*2.0+0.4, widths=0.6)
        plt.xticks(np.arange(len(cols))*2.0, cols)
        plt.legend(['panic=0','panic=1'])
    plt.title('Boxplots by panic_mode')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_pairplot(df, cols, fname):
    if HAS_SEABORN:
        pp = sns.pairplot(df[cols + ['panic_mode']].dropna(), hue='panic_mode', corner=True, plot_kws={'s':15, 'alpha':0.6})
        pp.fig.suptitle('Pairplot (external signals)')
        pp.fig.tight_layout()
        pp.fig.subplots_adjust(top=0.95)
        pp.fig.savefig(fname, dpi=150)
        plt.close()
    else:
        # fallback: scatter matrix
        from pandas.plotting import scatter_matrix
        axes = scatter_matrix(df[cols].dropna(), alpha=0.3, figsize=(9,9))
        plt.suptitle('Scatter matrix')
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

def save_corr_heatmap(df, cols, fname):
    plt.figure(figsize=(6,5))
    corr = df[cols].corr()
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
    else:
        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(cols)), cols, rotation=90)
        plt.yticks(range(len(cols)), cols)
    plt.title('Correlation heatmap')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def main():
    df = pd.read_csv(CSV)
    df = make_panic_label(df)
    os.chdir(ROOT)

    # features to visualize
    ext_feats = ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']

    # Histograms
    for f in ext_feats:
        out = f.replace(' ', '_') + '_hist.png'
        try:
            save_hist(df, f, out)
        except Exception as e:
            print('hist failed', f, e)

    # Pairplot / scatter matrix
    try:
        save_pairplot(df, ext_feats, 'pairplot_external.png')
    except Exception as e:
        print('pairplot failed', e)

    # Boxplots by panic
    try:
        save_box_by_panic(df, ext_feats, 'box_by_panic.png')
    except Exception as e:
        print('boxplot failed', e)

    # Correlation heatmap
    try:
        save_corr_heatmap(df, ext_feats + ['avalon_raw_risk_score', 'true_risk_level'], 'corr_heatmap.png')
    except Exception as e:
        print('corr heatmap failed', e)

    # Small summary CSV with means by panic_mode
    summary = df.groupby('panic_mode')[ext_feats + ['avalon_raw_risk_score', 'true_risk_level']].median()
    summary.to_csv('visuals_summary_by_panic.csv')
    print('Saved visuals and summary CSV')

if __name__ == '__main__':
    main()
