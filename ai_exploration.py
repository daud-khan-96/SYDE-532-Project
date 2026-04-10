"""
Exploratory AI-Assisted Analyses -- Beyond Core Report Scope
SYDE 532 Final Project, Daud Khan

This script is SEPARATE from the main reproducible analysis pipeline.
It explores additional analyses, alternative parameters, and supplementary
visualisations that go beyond the required report. Outputs are saved to
a dedicated exploratory/ folder and do not overwrite main results.

Analyses included:
  1. Window-size sensitivity (2, 10, 15 min vs. the default 5 min)
  2. Betweenness and eigenvector centrality profiles for El Clasico 1
  3. Half-by-half structural comparison (first half vs. second half)
  4. Degree distribution shape analysis per outcome group
  5. Role stability heatmap (centrality rank changes across windows)
"""

import json
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'open-data', 'data')
EVENTS_DIR = os.path.join(DATA_DIR, 'events')
MATCHES_DIR = os.path.join(DATA_DIR, 'matches')
RESULTS_DIR = os.path.join(BASE, 'results')
EXPLORE_DIR = os.path.join(BASE, 'exploratory')
os.makedirs(EXPLORE_DIR, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.facecolor': 'white',
})

# Known name mapping (same as fix_and_regenerate.py)
KNOWN_NAMES = {
    "Lionel Andrés Messi Cuccittini": "Messi",
    "Neymar da Silva Santos Junior": "Neymar",
    "Luis Alberto Suárez Díaz": "Suárez",
    "Andrés Iniesta Luján": "Iniesta",
    "Sergio Busquets i Burgos": "Busquets",
    "Ivan Rakitić": "Rakitić",
    "Gerard Piqué Bernabéu": "Piqué",
    "Jordi Alba Ramos": "J. Alba",
    "Daniel Alves da Silva": "Dani Alves",
    "Javier Alejandro Mascherano": "Mascherano",
    "Claudio Andrés Bravo Muñoz": "Bravo",
    "Sergi Roberto Carnicer": "Sergi Roberto",
    "Cristiano Ronaldo dos Santos Aveiro": "Ronaldo",
    "Toni Kroos": "Kroos",
    "Luka Modrić": "Modrić",
    "Sergio Ramos García": "S. Ramos",
    "Karim Benzema": "Benzema",
    "Gareth Frank Bale": "Bale",
    "Marcelo Vieira da Silva Júnior": "Marcelo",
    "Daniel Carvajal Ramos": "Carvajal",
    "Francisco Román Alarcón Suárez": "Isco",
    "Carlos Henrique Casimiro": "Casemiro",
    "Kléper Laveran Lima Ferreira": "Pepe",
}


def short_name(full_name):
    if full_name in KNOWN_NAMES:
        return KNOWN_NAMES[full_name]
    parts = full_name.split()
    if len(parts) <= 2:
        return parts[-1]
    return parts[0]


def extract_passes(events, window_min=5):
    passes = []
    for e in events:
        if e.get('type', {}).get('name') == 'Pass' and 'pass' in e:
            p = e['pass']
            if 'recipient' in p and 'outcome' not in p:
                passes.append({
                    'from': e['player']['name'],
                    'to': p['recipient']['name'],
                    'team': e['team']['name'],
                    'minute': e['minute'],
                    'time_bin': e['minute'] // window_min,
                })
    return pd.DataFrame(passes)


def build_network(df_window):
    G = nx.DiGraph()
    for _, row in df_window.iterrows():
        u, v = row['from'], row['to']
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def network_metrics(G):
    if len(G.nodes()) < 2:
        return {'density': 0, 'clustering': 0, 'transitivity': 0, 'avg_degree': 0}
    return {
        'density': nx.density(G),
        'clustering': nx.average_clustering(G.to_undirected()),
        'transitivity': nx.transitivity(G.to_undirected()),
        'avg_degree': np.mean([d for _, d in G.degree()]),
    }


# ======================================================================
#  EXPLORATION 1: Window-size sensitivity
# ======================================================================

def explore_window_sensitivity():
    """Compare network metrics across different time-window sizes for EC1."""
    print("[1] Window-size sensitivity analysis...")

    match_id = 266424  # El Clasico 1
    with open(os.path.join(EVENTS_DIR, f'{match_id}.json')) as f:
        ev = json.load(f)

    windows = [2, 5, 10, 15]
    team = 'Barcelona'
    results = {}

    for w in windows:
        df = extract_passes(ev, window_min=w)
        df_t = df[df['team'] == team]
        bins = sorted(df_t['time_bin'].unique())
        metrics_per_bin = []
        for b in bins:
            G = build_network(df_t[df_t['time_bin'] == b])
            m = network_metrics(G)
            m['time'] = b * w
            metrics_per_bin.append(m)
        results[w] = pd.DataFrame(metrics_per_bin)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    metric_names = ['density', 'clustering', 'transitivity', 'avg_degree']
    metric_labels = ['Network Density', 'Clustering Coeff.', 'Transitivity', 'Average Degree']

    for ax, met, lab in zip(axes.flat, metric_names, metric_labels):
        for w in windows:
            df_w = results[w]
            ax.plot(df_w['time'], df_w[met], 'o-', label=f'{w}-min', ms=3, lw=1)
        ax.set_xlabel('Match Time (min)')
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)

    fig.suptitle('Window-Size Sensitivity: Barcelona in El Clasico 1',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB1_window_sensitivity.png'))
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB1_window_sensitivity.pdf'))
    plt.close(fig)
    print("  Saved figB1_window_sensitivity")


# ======================================================================
#  EXPLORATION 2: Betweenness and eigenvector centrality for EC1
# ======================================================================

def explore_centrality_profiles():
    """Plot betweenness and closeness centrality for top players in EC1."""
    print("[2] Extended centrality profiles...")

    match_id = 266424
    with open(os.path.join(EVENTS_DIR, f'{match_id}.json')) as f:
        ev = json.load(f)

    df = extract_passes(ev)
    team = 'Barcelona'
    df_t = df[df['team'] == team]
    bins = sorted(df_t['time_bin'].unique())

    # Collect betweenness per window
    betw_data = defaultdict(list)
    for b in bins:
        G = build_network(df_t[df_t['time_bin'] == b])
        if len(G.nodes()) < 3:
            continue
        betw = nx.betweenness_centrality(G, weight='weight')
        for player, val in betw.items():
            betw_data[player].append((b * 5, val))

    # Find top-3 by total betweenness
    totals = {p: sum(v for _, v in vals) for p, vals in betw_data.items()}
    top3 = sorted(totals, key=totals.get, reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#1E88E5', '#E53935', '#43A047']
    for p, c in zip(top3, colors):
        times = [t for t, _ in betw_data[p]]
        vals = [v for _, v in betw_data[p]]
        ax.plot(times, vals, 'o-', color=c, label=short_name(p), lw=1.2, ms=4)

    ax.set_xlabel('Match Time (min)')
    ax.set_ylabel('Betweenness Centrality')
    ax.set_title('Barcelona Betweenness Centrality Over Time (El Clasico 1)',
                 fontsize=10, fontweight='bold')
    ax.legend()
    ax.axvline(45, color='gray', ls=':', alpha=0.5)
    ax.axvline(83, color='red', ls='--', lw=1.5, alpha=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB2_betweenness_profile.png'))
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB2_betweenness_profile.pdf'))
    plt.close(fig)
    print(f"  Top-3 betweenness: {[short_name(p) for p in top3]}")
    print("  Saved figB2_betweenness_profile")


# ======================================================================
#  EXPLORATION 3: First half vs second half structural comparison
# ======================================================================

def explore_half_comparison():
    """Compare network metrics between first and second halves across season."""
    print("[3] Half-by-half structural comparison...")

    season_df = pd.read_csv(os.path.join(RESULTS_DIR, 'season_results.csv'))

    # We need to recompute from raw data for half-level granularity.
    # Instead, use the cached data and split by checking if we have
    # half-level info. Since we don't, use a representative sample.
    # Load 20 random matches and compute first-half vs second-half metrics.

    with open(os.path.join(MATCHES_DIR, '11', '27.json')) as f:
        matches = json.load(f)

    rng = np.random.RandomState(42)
    sample = rng.choice(matches, size=30, replace=False)

    half_data = []
    for m in sample:
        mid = m['match_id']
        fp = os.path.join(EVENTS_DIR, f'{mid}.json')
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            ev = json.load(f)
        df = extract_passes(ev)
        for team in df['team'].unique():
            df_t = df[df['team'] == team]
            for half_label, minute_range in [('1st Half', (0, 45)), ('2nd Half', (45, 200))]:
                df_h = df_t[(df_t['minute'] >= minute_range[0]) & (df_t['minute'] < minute_range[1])]
                if len(df_h) < 5:
                    continue
                G = build_network(df_h)
                m_dict = network_metrics(G)
                m_dict['half'] = half_label
                m_dict['team'] = team
                half_data.append(m_dict)

    hdf = pd.DataFrame(half_data)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    metrics = ['density', 'clustering', 'transitivity', 'avg_degree']
    labels = ['Density', 'Clustering', 'Transitivity', 'Avg Degree']

    for ax, met, lab in zip(axes, metrics, labels):
        data_1 = hdf[hdf['half'] == '1st Half'][met].dropna()
        data_2 = hdf[hdf['half'] == '2nd Half'][met].dropna()
        bp = ax.boxplot([data_1, data_2], labels=['1st', '2nd'],
                        patch_artist=True, notch=True, widths=0.5, showfliers=False)
        bp['boxes'][0].set_facecolor('#42A5F5')
        bp['boxes'][1].set_facecolor('#EF5350')
        for b in bp['boxes']:
            b.set_alpha(0.6)
        u_stat, p_val = stats.mannwhitneyu(data_1, data_2, alternative='two-sided')
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.set_title(f'{lab} ({sig})', fontsize=9)

    fig.suptitle('First Half vs. Second Half Network Metrics (30-match sample)',
                 fontsize=10, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB3_half_comparison.png'))
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB3_half_comparison.pdf'))
    plt.close(fig)
    print("  Saved figB3_half_comparison")


# ======================================================================
#  EXPLORATION 4: Degree distribution shape by outcome
# ======================================================================

def explore_degree_distributions():
    """Compare aggregated degree distributions by match outcome."""
    print("[4] Degree distribution by outcome...")

    with open(os.path.join(MATCHES_DIR, '11', '27.json')) as f:
        matches = json.load(f)

    rng = np.random.RandomState(42)
    sample = rng.choice(matches, size=40, replace=False)

    outcome_degrees = {'Win': [], 'Draw': [], 'Loss': []}

    for m in sample:
        mid = m['match_id']
        fp = os.path.join(EVENTS_DIR, f'{mid}.json')
        if not os.path.exists(fp):
            continue
        hs, aws = m['home_score'], m['away_score']
        with open(fp) as f:
            ev = json.load(f)
        df = extract_passes(ev)

        home = m['home_team']['home_team_name']
        away = m['away_team']['away_team_name']

        for team, score_for, score_against in [(home, hs, aws), (away, aws, hs)]:
            if score_for > score_against:
                outcome = 'Win'
            elif score_for == score_against:
                outcome = 'Draw'
            else:
                outcome = 'Loss'

            df_t = df[df['team'] == team]
            G = build_network(df_t)
            degrees = [d for _, d in G.degree()]
            outcome_degrees[outcome].extend(degrees)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    colors = {'Win': '#2e7d32', 'Draw': '#f9a825', 'Loss': '#c62828'}

    for ax, outcome in zip(axes, ['Win', 'Draw', 'Loss']):
        degs = outcome_degrees[outcome]
        if degs:
            ax.hist(degs, bins=range(0, max(degs)+2), density=True,
                    color=colors[outcome], alpha=0.7, edgecolor='white')
        ax.set_title(f'{outcome} (n={len(degs)})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Node Degree')
        if ax == axes[0]:
            ax.set_ylabel('Density')

    fig.suptitle('Degree Distributions by Match Outcome (40-match sample)',
                 fontsize=10, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB4_degree_distributions.png'))
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB4_degree_distributions.pdf'))
    plt.close(fig)
    print("  Saved figB4_degree_distributions")


# ======================================================================
#  EXPLORATION 5: Role stability heatmap for El Clasico 1
# ======================================================================

def explore_role_stability():
    """Heatmap of centrality rank changes across time windows."""
    print("[5] Role stability heatmap...")

    match_id = 266424
    with open(os.path.join(EVENTS_DIR, f'{match_id}.json')) as f:
        ev = json.load(f)

    df = extract_passes(ev)
    team = 'Barcelona'
    df_t = df[df['team'] == team]
    bins = sorted(df_t['time_bin'].unique())

    # Build centrality rank matrix
    rank_data = {}
    all_players = set()
    for b in bins:
        G = build_network(df_t[df_t['time_bin'] == b])
        deg = nx.degree_centrality(G)
        ranked = sorted(deg, key=deg.get, reverse=True)
        rank_data[b] = {p: i+1 for i, p in enumerate(ranked)}
        all_players.update(deg.keys())

    # Keep only players who appear in at least half the windows
    threshold = len(bins) // 2
    frequent = [p for p in all_players
                if sum(1 for b in bins if p in rank_data[b]) >= threshold]
    frequent.sort(key=lambda p: np.mean([rank_data[b].get(p, 15) for b in bins]))

    # Build matrix
    matrix = np.full((len(frequent), len(bins)), np.nan)
    for i, p in enumerate(frequent):
        for j, b in enumerate(bins):
            if p in rank_data[b]:
                matrix[i, j] = rank_data[b][p]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=1, vmax=12)
    ax.set_yticks(range(len(frequent)))
    ax.set_yticklabels([short_name(p) for p in frequent], fontsize=8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f'{b*5}' for b in bins], fontsize=7)
    ax.set_xlabel('Match Time (min)')
    ax.set_ylabel('Player')
    ax.set_title('Barcelona Centrality Rank Over Time (El Clasico 1)\n'
                 'Lower rank = more central; white = not active in window',
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Centrality Rank', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB5_role_stability.png'))
    fig.savefig(os.path.join(EXPLORE_DIR, 'figB5_role_stability.pdf'))
    plt.close(fig)
    print("  Saved figB5_role_stability")


# ======================================================================
#  MAIN
# ======================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("EXPLORATORY AI-ASSISTED ANALYSES")
    print("Beyond core report scope")
    print("=" * 60)

    explore_window_sensitivity()
    explore_centrality_profiles()
    explore_half_comparison()
    explore_degree_distributions()
    explore_role_stability()

    print("\n" + "=" * 60)
    print("ALL EXPLORATORY OUTPUTS SAVED TO exploratory/")
    print("=" * 60)
