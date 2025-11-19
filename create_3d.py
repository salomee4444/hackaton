import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

    x = df['public_anxiety_index']
    y = df['social_media_rumour_index']
    z = df['regulator_scrutiny_score']
    cat = df['panic_mode']

    # size by population within 30km (scaled)
    pop = df['population_within_30km'].fillna(df['population_within_30km'].median())
    sizes = (np.sqrt(pop) / np.sqrt(pop).max()) * 80 + 10

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # color map: panic=1 red, panic=0 blue (with some alpha)
    colors = np.where(cat==1, 'red', 'C0')

    sc = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.6, edgecolors='w', linewidth=0.2)

    ax.set_xlabel('public_anxiety_index')
    ax.set_ylabel('social_media_rumour_index')
    ax.set_zlabel('regulator_scrutiny_score')
    ax.set_title('3D scatter: External signals (color=panic_mode, size~population)')

    # create custom legend for panic_mode
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w', label='panic_mode=0', markerfacecolor='C0', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='panic_mode=1', markerfacecolor='red', markersize=8)]
    ax.legend(handles=legend_elems, loc='upper left')

    out = os.path.join(ROOT, '3d_scatter.png')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('Saved', out)

if __name__ == '__main__':
    main()
