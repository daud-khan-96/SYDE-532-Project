"""
Fix player naming and regenerate all El Clasico and community figures.
Also regenerate the season boxplots to align with text.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'open-data', 'data')
EVENTS_DIR = os.path.join(DATA_DIR, 'events')
MATCHES_DIR = os.path.join(DATA_DIR, 'matches')
FIG_DIR = os.path.join(BASE, 'figures')
RESULTS_DIR = os.path.join(BASE, 'results')

TIME_WINDOW = 5
np.random.seed(42)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

OUTCOME_COLORS = {'Win': '#2e7d32', 'Draw': '#f9a825', 'Loss': '#c62828'}

# Known name mapping for StatsBomb full names to common names
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
    "Jérémy Mathieu": "Mathieu",
    "Sergi Roberto Carnicer": "Sergi Roberto",
    "Munir El Haddadi Mohamed": "Munir",
    "Cristiano Ronaldo dos Santos Aveiro": "Ronaldo",
    "Toni Kroos": "Kroos",
    "Luka Modrić": "Modrić",
    "Sergio Ramos García": "S. Ramos",
    "Karim Benzema": "Benzema",
    "Gareth Frank Bale": "Bale",
    "Marcelo Vieira da Silva Júnior": "Marcelo",
    "Daniel Carvajal Ramos": "Carvajal",
    "Danilo Luiz da Silva": "Danilo",
    "Francisco Román Alarcón Suárez": "Isco",
    "James David Rodríguez Rubio": "James",
    "Raphaël Varane": "Varane",
    "Keylor Navas Gamboa": "Navas",
    "Carlos Henrique Casimiro": "Casemiro",
    "Lucas Vázquez Iglesias": "Lucas V.",
    "Kléper Laveran Lima Ferreira": "Pepe",
    "Jesé Rodríguez Ruiz": "Jesé",
    "Arda Turan": "Arda Turan",
}


def short_name(full_name):
    """Get a recognizable short name for a player."""
    if full_name in KNOWN_NAMES:
        return KNOWN_NAMES[full_name]
    parts = full_name.split()
    if len(parts) <= 2:
        return parts[-1]
    # For unknown players, try first and last
    return parts[0]


# ── Core functions (unchanged from analysis_main.py) ──

def extract_passes(events):
    passes = []
    for e in events:
        if e.get('type', {}).get('name') == 'Pass' and 'pass' in e:
            p = e['pass']
            if 'recipient' in p and 'outcome' not in p:
                loc = e.get('location', [None, None])
                end_loc = p.get('end_location', [None, None])
                passes.append({
                    'from': e['player']['name'],
                    'to': p['recipient']['name'],
                    'team': e['team']['name'],
                    'minute': e['minute'],
                    'second': e.get('second', 0),
                    'time_bin': e['minute'] // TIME_WINDOW,
                    'loc_x': loc[0] if loc else None,
                    'loc_y': loc[1] if loc else None,
                    'end_x': end_loc[0] if end_loc else None,
                    'end_y': end_loc[1] if end_loc else None,
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


def build_temporal_networks(df, team):
    df_t = df[df['team'] == team]
    return {tb: build_network(df_t[df_t['time_bin'] == tb])
            for tb in sorted(df_t['time_bin'].unique())}


def compute_centralities(G):
    if len(G.nodes()) < 2:
        return {}
    deg = nx.degree_centrality(G)
    total_deg = sum(deg.values())
    norm_deg = {n: v / total_deg if total_deg > 0 else 0 for n, v in deg.items()}
    result = {}
    for n in G.nodes():
        result[n] = {
            'degree': deg[n],
            'norm_degree': norm_deg[n],
        }
    return result


def network_metrics(G):
    if len(G.nodes()) < 2:
        return {'density': 0, 'clustering': 0, 'transitivity': 0, 'avg_degree': 0}
    return {
        'density': nx.density(G),
        'clustering': nx.average_clustering(G.to_undirected()),
        'transitivity': nx.transitivity(G.to_undirected()),
        'avg_degree': np.mean([d for _, d in G.degree()]),
    }


def detect_communities(G):
    if len(G.nodes()) < 3:
        return [], 0
    Gu = G.to_undirected()
    comms = list(nx.community.louvain_communities(Gu, seed=42))
    mod = nx.community.modularity(Gu, comms)
    return comms, mod


def get_top_players(cents_dict, metric, n=3):
    totals = defaultdict(float)
    for tb, c in cents_dict.items():
        for player, vals in c.items():
            totals[player] += vals.get(metric, 0)
    return sorted(totals, key=totals.get, reverse=True)[:n]


def get_red_cards(events):
    cards = []
    for e in events:
        if e.get('type', {}).get('name') == 'Foul Committed':
            fc = e.get('foul_committed', {})
            if 'card' in fc:
                card_name = fc['card']['name']
                if 'Red' in card_name or 'Second Yellow' in card_name:
                    cards.append({
                        'minute': e['minute'],
                        'player': e['player']['name'],
                        'team': e['team']['name'],
                        'card': card_name,
                    })
    return cards


def draw_pitch(ax, color='white', linewidth=1.5):
    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#2d572c')
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color=color, lw=linewidth)
    ax.plot([60, 60], [0, 80], color=color, lw=linewidth)
    circle = plt.Circle((60, 40), 9.15, fill=False, edgecolor=color, lw=linewidth)
    ax.add_patch(circle)
    ax.plot(60, 40, 'o', color=color, ms=3)
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], color=color, lw=linewidth)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], color=color, lw=linewidth)
    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], color=color, lw=linewidth)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], color=color, lw=linewidth)
    ax.plot(12, 40, 'o', color=color, ms=3)
    ax.plot(108, 40, 'o', color=color, ms=3)
    arc1 = Arc((12, 40), 18.3, 18.3, angle=0, theta1=-55, theta2=55, color=color, lw=linewidth)
    arc2 = Arc((108, 40), 18.3, 18.3, angle=0, theta1=125, theta2=235, color=color, lw=linewidth)
    ax.add_patch(arc1)
    ax.add_patch(arc2)


def plot_passing_network(df, team, ax=None, color='#1E88E5', min_passes=3, title_extra=""):
    df_t = df[df['team'] == team]
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    pos = df_t.groupby('from').agg(x=('loc_x', 'mean'), y=('loc_y', 'mean'), n=('loc_x', 'count'))
    G = build_network(df_t)
    if len(G.edges()) == 0:
        return ax
    max_w = max(d['weight'] for _, _, d in G.edges(data=True))
    for u, v, d in G.edges(data=True):
        if d['weight'] >= min_passes and u in pos.index and v in pos.index:
            w = d['weight']
            lw = 0.5 + (w / max_w) * 4
            alpha = 0.2 + (w / max_w) * 0.6
            ax.annotate('', xy=(pos.loc[v, 'x'], pos.loc[v, 'y']),
                        xytext=(pos.loc[u, 'x'], pos.loc[u, 'y']),
                        arrowprops=dict(arrowstyle='->', color='white', lw=lw,
                                        alpha=alpha, connectionstyle='arc3,rad=0.08'))
    for player in pos.index:
        size = 150 + pos.loc[player, 'n'] * 8
        ax.scatter(pos.loc[player, 'x'], pos.loc[player, 'y'],
                   s=size, c=color, edgecolors='white', linewidths=1.5, zorder=5)
        name = short_name(player)
        ax.annotate(name, (pos.loc[player, 'x'], pos.loc[player, 'y']),
                    xytext=(0, -16), textcoords='offset points', ha='center',
                    fontsize=7, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.75))
    ax.set_title(f'{team} {title_extra}', fontsize=11, fontweight='bold', color='white', pad=8)
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  REGENERATE EL CLASICO FIGURES (with fixed names)
# ═══════════════════════════════════════════════════════════════════════

def fig_clasico_temporal(match_id, match_label, filename_prefix):
    """Generate El Clasico temporal figure with correct player names."""
    with open(os.path.join(EVENTS_DIR, f'{match_id}.json')) as f:
        ev = json.load(f)
    with open(os.path.join(MATCHES_DIR, '11', '27.json')) as f:
        season = json.load(f)
    info = next(m for m in season if m['match_id'] == match_id)
    home = info['home_team']['home_team_name']
    away = info['away_team']['away_team_name']
    hs, aws = info['home_score'], info['away_score']
    teams_list = [home, away]

    df = extract_passes(ev)
    red_cards = get_red_cards(ev)
    rc_min = red_cards[0]['minute'] if red_cards else None
    rc_player = short_name(red_cards[0]['player']) if red_cards else None

    team_colors = ['#0067B1', '#A50044']
    if 'Barcelona' == home:
        team_colors = ['#A50044', '#0067B1']
    elif 'Barcelona' == away:
        team_colors = ['#0067B1', '#A50044']

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32)

    # (a) Pass volume
    ax_vol = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(teams_list):
        df_t = df[df['team'] == t]
        ppb = df_t.groupby('time_bin').size()
        lab = t
        ax_vol.plot(ppb.index * TIME_WINDOW, ppb.values, 'o-',
                    color=team_colors[i], label=lab, lw=1.5, ms=4)
    ax_vol.axvline(45, color='gray', ls=':', alpha=0.5, label='Half')
    if rc_min:
        ax_vol.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7,
                       label=f'Red ({rc_player})')
    ax_vol.set_xlabel('Match Time (min)')
    ax_vol.set_ylabel('Passes per Window')
    ax_vol.set_title('(a) Pass Volume', fontsize=9, fontweight='bold')
    ax_vol.legend(fontsize=6, loc='upper right')

    # (b) and (c) Centrality panels
    for idx, (t, tc) in enumerate(zip(teams_list, team_colors)):
        if idx == 0:
            ax_cent = fig.add_subplot(gs[0, 1])
            panel_letter = '(b)'
        else:
            ax_cent = fig.add_subplot(gs[1, 0])
            panel_letter = '(c)'

        nets_t = build_temporal_networks(df, t)
        cents_t = {tb: compute_centralities(G) for tb, G in nets_t.items()}
        top = get_top_players(cents_t, 'norm_degree', 3)
        bins = sorted(cents_t.keys())
        mins = [b * TIME_WINDOW for b in bins]

        # Print verification
        print(f"  {t} top-3 centrality: {[short_name(p) for p in top]}")

        cent_colors = ['#1E88E5', '#E53935', '#43A047']
        for j, p in enumerate(top):
            vals = [cents_t[b].get(p, {}).get('norm_degree', 0) for b in bins]
            ax_cent.plot(mins, vals, 'o-', color=cent_colors[j],
                         label=short_name(p), lw=1.2, ms=3)

        ax_cent.axvline(45, color='gray', ls=':', alpha=0.5)
        if rc_min:
            ax_cent.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7)
        ax_cent.set_title(f'{panel_letter} Centrality: {t}', fontsize=9, fontweight='bold')
        ax_cent.set_xlabel('Match Time (min)')
        ax_cent.set_ylabel('Norm. Degree Centrality')
        ax_cent.legend(fontsize=7, ncol=1)

    # (d) Network metrics over time
    ax_met = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(teams_list):
        nets_t = build_temporal_networks(df, t)
        met_t = {tb: network_metrics(G) for tb, G in nets_t.items()}
        bins = sorted(met_t.keys())
        mins = [b * TIME_WINDOW for b in bins]
        ax_met.plot(mins, [met_t[b]['density'] for b in bins], 'o-',
                    color=team_colors[i], label=f'{t} density', lw=1.2, ms=3)
        ax_met.plot(mins, [met_t[b]['clustering'] for b in bins], 's--',
                    color=team_colors[i], label=f'{t} clustering', lw=1, ms=3, alpha=0.7)
    ax_met.axvline(45, color='gray', ls=':', alpha=0.5)
    if rc_min:
        ax_met.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7)
    ax_met.set_title('(d) Network Density & Clustering', fontsize=9, fontweight='bold')
    ax_met.set_xlabel('Match Time (min)')
    ax_met.set_ylabel('Metric Value')
    ax_met.legend(fontsize=6, ncol=2)

    # (e) Modularity over time
    ax_comm = fig.add_subplot(gs[2, 0])
    for i, t in enumerate(teams_list):
        nets_t = build_temporal_networks(df, t)
        comm_data = {}
        for tb, G in nets_t.items():
            c, m = detect_communities(G)
            comm_data[tb] = {'n': len(c), 'mod': m}
        bins = sorted(comm_data.keys())
        mins = [b * TIME_WINDOW for b in bins]
        ax_comm.plot(mins, [comm_data[b]['mod'] for b in bins], 'o-',
                     color=team_colors[i], label=f'{t}', lw=1.2, ms=3)
    ax_comm.axvline(45, color='gray', ls=':', alpha=0.5)
    if rc_min:
        ax_comm.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7)
    ax_comm.set_title('(e) Modularity Over Time', fontsize=9, fontweight='bold')
    ax_comm.set_xlabel('Match Time (min)')
    ax_comm.set_ylabel('Modularity')
    ax_comm.legend(fontsize=7)

    # (f) leave blank or remove pitch network from main figure
    ax_blank = fig.add_subplot(gs[2, 1])
    # Number of communities over time instead of tiny pitch network
    for i, t in enumerate(teams_list):
        nets_t = build_temporal_networks(df, t)
        comm_data = {}
        for tb, G in nets_t.items():
            c, m = detect_communities(G)
            comm_data[tb] = len(c)
        bins = sorted(comm_data.keys())
        mins = [b * TIME_WINDOW for b in bins]
        ax_blank.plot(mins, [comm_data[b] for b in bins], 'o-',
                      color=team_colors[i], label=f'{t}', lw=1.2, ms=3)
    ax_blank.axvline(45, color='gray', ls=':', alpha=0.5)
    if rc_min:
        ax_blank.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7)
    ax_blank.set_title('(f) Number of Communities', fontsize=9, fontweight='bold')
    ax_blank.set_xlabel('Match Time (min)')
    ax_blank.set_ylabel('Communities')
    ax_blank.legend(fontsize=7)

    fig.suptitle(f'{match_label}: {home} {hs} - {aws} {away}',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(FIG_DIR, f'{filename_prefix}.pdf'))
    fig.savefig(os.path.join(FIG_DIR, f'{filename_prefix}.png'))
    plt.close(fig)
    print(f"  Saved {filename_prefix}")


# ═══════════════════════════════════════════════════════════════════════
#  REGENERATE SEASON BOXPLOTS (align with text: density, clustering, transitivity, avg_degree)
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_ci(data, n_boot=10000, ci=95):
    data = np.array(data)
    data = data[~np.isnan(data)]
    medians = np.array([np.median(np.random.choice(data, size=len(data), replace=True))
                        for _ in range(n_boot)])
    alpha = (100 - ci) / 2
    return np.percentile(medians, alpha), np.percentile(medians, 100 - alpha)


def fig_season_boxplots_fixed(df):
    """Figure: Notched boxplots matching what text discusses."""
    metrics_to_plot = [
        ('avg_density', 'Network Density', 18.5, 0.0001),
        ('avg_clustering', 'Clustering Coeff.', 10.5, 0.005),
        ('avg_transitivity', 'Transitivity', 11.2, 0.004),
        ('avg_degree', 'Average Degree', 10.4, 0.005),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
    for ax, (col, label, h_val, p_val) in zip(axes, metrics_to_plot):
        data = [df[df['outcome'] == o][col].dropna() for o in ['Win', 'Draw', 'Loss']]
        bp = ax.boxplot(data, labels=['W', 'D', 'L'], patch_artist=True,
                        notch=True, widths=0.55, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for patch, outcome in zip(bp['boxes'], ['Win', 'Draw', 'Loss']):
            patch.set_facecolor(OUTCOME_COLORS[outcome])
            patch.set_alpha(0.65)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.set_title(f'{label}\n(H={h_val:.1f}, {sig})', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle('Network Metrics by Match Outcome (La Liga 2015/16, n=760)',
                 fontsize=10, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_season_boxplots.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig1_season_boxplots.png'))
    plt.close(fig)
    print("  Saved fig1_season_boxplots (fixed: density, clustering, transitivity, avg_degree)")


# ═══════════════════════════════════════════════════════════════════════
#  REGENERATE COMMUNITY FIGURES (with correct names)
# ═══════════════════════════════════════════════════════════════════════

def fig_community_visualization(match_id, team_name, time_bin_target, filename):
    with open(os.path.join(EVENTS_DIR, f'{match_id}.json')) as f:
        ev = json.load(f)
    df = extract_passes(ev)
    df_t = df[(df['team'] == team_name) & (df['time_bin'] == time_bin_target)]
    if len(df_t) < 5:
        print(f"  Skipping community viz: not enough passes in bin {time_bin_target}")
        return

    G = build_network(df_t)
    comms, mod = detect_communities(G)
    pos_data = df_t.groupby('from').agg(x=('loc_x', 'mean'), y=('loc_y', 'mean'))

    comm_colors = ['#E53935', '#1E88E5', '#43A047', '#FF9800', '#9C27B0', '#795548']
    node_color_map = {}
    for i, comm in enumerate(comms):
        for player in comm:
            node_color_map[player] = comm_colors[i % len(comm_colors)]

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_pitch(ax)

    max_w = max((d['weight'] for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        if u in pos_data.index and v in pos_data.index:
            lw = 0.5 + (d['weight'] / max_w) * 3
            alpha = 0.2 + (d['weight'] / max_w) * 0.5
            ax.annotate('', xy=(pos_data.loc[v, 'x'], pos_data.loc[v, 'y']),
                        xytext=(pos_data.loc[u, 'x'], pos_data.loc[u, 'y']),
                        arrowprops=dict(arrowstyle='->', color='white', lw=lw,
                                        alpha=alpha, connectionstyle='arc3,rad=0.08'))

    for player in G.nodes():
        if player in pos_data.index:
            c = node_color_map.get(player, '#888888')
            ax.scatter(pos_data.loc[player, 'x'], pos_data.loc[player, 'y'],
                       s=250, c=c, edgecolors='white', linewidths=2, zorder=5)
            name = short_name(player)
            ax.annotate(name, (pos_data.loc[player, 'x'], pos_data.loc[player, 'y']),
                        xytext=(0, -18), textcoords='offset points', ha='center',
                        fontsize=8, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.8))

    for i, comm in enumerate(comms):
        members = [short_name(p) for p in comm]
        ax.plot([], [], 'o', color=comm_colors[i % len(comm_colors)], ms=8,
                label=f'C{i+1}: {", ".join(members)}')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9,
              facecolor='black', labelcolor='white')

    window_start = time_bin_target * TIME_WINDOW
    window_end = window_start + TIME_WINDOW
    ax.set_title(f'{team_name} Community Structure ({window_start}-{window_end} min, '
                 f'Q={mod:.2f})', fontsize=10, fontweight='bold', color='white')

    fig.patch.set_facecolor('#1a472a')
    fig.savefig(os.path.join(FIG_DIR, filename), facecolor='#1a472a')
    plt.close(fig)
    print(f"  Saved {filename}")


def fig_appendix_pitch_networks(match_id, match_label, filename):
    with open(os.path.join(EVENTS_DIR, f'{match_id}.json')) as f:
        ev = json.load(f)
    with open(os.path.join(MATCHES_DIR, '11', '27.json')) as f:
        season = json.load(f)
    info = next(m for m in season if m['match_id'] == match_id)
    home = info['home_team']['home_team_name']
    away = info['away_team']['away_team_name']
    hs, aws = info['home_score'], info['away_score']

    df = extract_passes(ev)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#A50044', '#0067B1'] if 'Barcelona' in home else ['#0067B1', '#A50044']
    for ax, t, c in zip(axes, [home, away], colors):
        plot_passing_network(df, t, ax=ax, color=c, min_passes=3)
    fig.patch.set_facecolor('#1a472a')
    fig.suptitle(f'{match_label}: {home} {hs} - {aws} {away}',
                 fontsize=11, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), facecolor='#1a472a')
    plt.close(fig)
    print(f"  Saved {filename}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("FIX & REGENERATE — Correcting player names and figures")
    print("=" * 60)

    # Load cached season data
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'season_results.csv'))
    print(f"Loaded {len(df)} observations")

    # 1. Regenerate season boxplots (now matching text: density, clustering, transitivity, avg_degree)
    print("\n[1] Regenerating season boxplots...")
    fig_season_boxplots_fixed(df)

    # 2. Regenerate El Clasico figures with correct player names
    print("\n[2] Regenerating El Clásico 1...")
    fig_clasico_temporal(266424, "El Clásico 1 (Bernabéu)", 'fig3_clasico1')

    print("\n[3] Regenerating El Clásico 2...")
    fig_clasico_temporal(267533, "El Clásico 2 (Camp Nou)", 'fig4_clasico2')

    # 3. Regenerate community figures with correct names
    print("\n[4] Regenerating community figures...")
    fig_community_visualization(266424, 'Barcelona', 4, 'figA4_barca_communities.png')
    fig_community_visualization(266424, 'Real Madrid', 4, 'figA5_madrid_communities.png')

    # 4. Regenerate pitch networks with correct names
    print("\n[5] Regenerating pitch networks...")
    fig_appendix_pitch_networks(266424, "El Clásico 1", 'figA2_ec1_pitch.png')
    fig_appendix_pitch_networks(267533, "El Clásico 2", 'figA3_ec2_pitch.png')

    print("\n" + "=" * 60)
    print("ALL FIGURES REGENERATED WITH CORRECT PLAYER NAMES")
    print("=" * 60)
