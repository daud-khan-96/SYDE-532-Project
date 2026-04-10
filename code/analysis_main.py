import json
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'open-data', 'data')
EVENTS_DIR = os.path.join(DATA_DIR, 'events')
MATCHES_DIR = os.path.join(DATA_DIR, 'matches')
FIG_DIR = os.path.join(BASE, 'figures')
TABLE_DIR = os.path.join(BASE, 'tables')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TIME_WINDOW = 5  # minutes
np.random.seed(42)

# ── Style ──────────────────────────────────────────────────────────────
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
TEAM_COLORS_MAP = {'Barcelona': '#A50044', 'Real Madrid': '#0067B1'}

# ═══════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def extract_passes(events):
    """Extract successful passes from match events."""
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


def get_substitutions(events, team=None):
    """Extract substitution events."""
    subs = []
    for e in events:
        if e.get('type', {}).get('name') == 'Substitution':
            t = e['team']['name']
            if team is None or t == team:
                subs.append({
                    'minute': e['minute'],
                    'player_out': e['player']['name'],
                    'player_in': e['substitution']['replacement']['name'],
                    'team': t,
                })
    return subs


def get_red_cards(events):
    """Extract red card events."""
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


def build_network(df_window):
    """Build directed weighted network from a pass dataframe."""
    G = nx.DiGraph()
    for _, row in df_window.iterrows():
        u, v = row['from'], row['to']
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def build_temporal_networks(df, team):
    """Build sequence of networks per time window for one team."""
    df_t = df[df['team'] == team]
    return {tb: build_network(df_t[df_t['time_bin'] == tb])
            for tb in sorted(df_t['time_bin'].unique())}


def compute_centralities(G):
    """Compute centrality metrics for all nodes."""
    if len(G.nodes()) < 2:
        return {}
    deg = nx.degree_centrality(G)
    close = nx.closeness_centrality(G)
    betw = nx.betweenness_centrality(G, weight='weight')
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    except:
        eig = {n: 0.0 for n in G.nodes()}
    total_deg = sum(deg.values())
    norm_deg = {n: v / total_deg if total_deg > 0 else 0 for n, v in deg.items()}
    result = {}
    for n in G.nodes():
        result[n] = {
            'degree': deg[n],
            'norm_degree': norm_deg[n],
            'closeness': close[n],
            'betweenness': betw[n],
            'eigenvector': eig[n],
        }
    return result


def network_metrics(G):
    """Compute network-level metrics."""
    if len(G.nodes()) < 2:
        return {'density': 0, 'clustering': 0, 'transitivity': 0, 'avg_degree': 0}
    return {
        'density': nx.density(G),
        'clustering': nx.average_clustering(G.to_undirected()),
        'transitivity': nx.transitivity(G.to_undirected()),
        'avg_degree': np.mean([d for _, d in G.degree()]),
    }


def detect_communities(G):
    """Louvain community detection on undirected version."""
    if len(G.nodes()) < 3:
        return [], 0
    Gu = G.to_undirected()
    comms = list(nx.community.louvain_communities(Gu, seed=42))
    mod = nx.community.modularity(Gu, comms)
    return comms, mod


def get_top_players(cents_dict, metric, n=3):
    """Get top-n players by total metric value across all windows."""
    totals = defaultdict(float)
    for tb, c in cents_dict.items():
        for player, vals in c.items():
            totals[player] += vals.get(metric, 0)
    return sorted(totals, key=totals.get, reverse=True)[:n]


# ═══════════════════════════════════════════════════════════════════════
#  PITCH DRAWING
# ═══════════════════════════════════════════════════════════════════════

def draw_pitch(ax, color='white', linewidth=1.5):
    """Draw a standard football pitch on axes (120x80 StatsBomb coords)."""
    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#2d572c')
    # Pitch outline and centre line
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect_params = dict(fill=False, edgecolor=color, linewidth=linewidth)
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color=color, lw=linewidth)
    ax.plot([60, 60], [0, 80], color=color, lw=linewidth)
    # Centre circle
    circle = plt.Circle((60, 40), 9.15, fill=False, edgecolor=color, lw=linewidth)
    ax.add_patch(circle)
    ax.plot(60, 40, 'o', color=color, ms=3)
    # Penalty areas
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], color=color, lw=linewidth)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], color=color, lw=linewidth)
    # Goal areas
    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], color=color, lw=linewidth)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], color=color, lw=linewidth)
    # Penalty spots
    ax.plot(12, 40, 'o', color=color, ms=3)
    ax.plot(108, 40, 'o', color=color, ms=3)
    # Penalty arcs
    arc1 = Arc((12, 40), 18.3, 18.3, angle=0, theta1=-55, theta2=55, color=color, lw=linewidth)
    arc2 = Arc((108, 40), 18.3, 18.3, angle=0, theta1=125, theta2=235, color=color, lw=linewidth)
    ax.add_patch(arc1)
    ax.add_patch(arc2)


def plot_passing_network(df, team, ax=None, color='#1E88E5', min_passes=2, title_extra=""):
    """Draw passing network on pitch."""
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
        name = player.split()[-1]
        ax.annotate(name, (pos.loc[player, 'x'], pos.loc[player, 'y']),
                    xytext=(0, -16), textcoords='offset points', ha='center',
                    fontsize=7, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.75))
    ax.set_title(f'{team} {title_extra}', fontsize=11, fontweight='bold', color='white', pad=8)
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  SEASON-WIDE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_match_team(events, team_name):
    """Compute aggregate network metrics for one team in one match."""
    df = extract_passes(events)
    df_t = df[df['team'] == team_name]
    if len(df_t) < 10:
        return None
    nets = build_temporal_networks(df, team_name)
    if len(nets) < 3:
        return None
    met = {tb: network_metrics(G) for tb, G in nets.items()}
    avg_density = np.mean([m['density'] for m in met.values()])
    avg_clustering = np.mean([m['clustering'] for m in met.values()])
    avg_transitivity = np.mean([m['transitivity'] for m in met.values()])
    avg_degree = np.mean([m['avg_degree'] for m in met.values()])
    mods, n_comms = [], []
    for tb, G in nets.items():
        c, m = detect_communities(G)
        mods.append(m)
        n_comms.append(len(c))
    cents = {tb: compute_centralities(G) for tb, G in nets.items()}
    all_players = set()
    for c in cents.values():
        all_players.update(c.keys())
    cent_vars = []
    for p in all_players:
        vals = [cents[tb].get(p, {}).get('norm_degree', 0) for tb in cents]
        if len(vals) > 1:
            cent_vars.append(np.std(vals))
    return {
        'total_passes': len(df_t),
        'avg_density': avg_density,
        'avg_clustering': avg_clustering,
        'avg_transitivity': avg_transitivity,
        'avg_degree': avg_degree,
        'avg_modularity': np.mean(mods),
        'avg_n_communities': np.mean(n_comms),
        'centrality_variability': np.mean(cent_vars) if cent_vars else 0,
    }


def run_season_analysis():
    """Process all 380 La Liga 2015/16 matches."""
    cache_path = os.path.join(RESULTS_DIR, 'season_results.csv')
    if os.path.exists(cache_path):
        print("Loading cached season results...")
        return pd.read_csv(cache_path)

    with open(os.path.join(MATCHES_DIR, '11', '27.json')) as f:
        season_matches = json.load(f)

    match_lookup = {}
    for m in season_matches:
        match_lookup[m['match_id']] = {
            'home': m['home_team']['home_team_name'],
            'away': m['away_team']['away_team_name'],
            'home_score': m['home_score'],
            'away_score': m['away_score'],
            'date': m['match_date'],
        }

    results = []
    found_ids = sorted(match_lookup.keys())
    print(f"Processing {len(found_ids)} matches...")

    for i, mid in enumerate(found_ids):
        if i % 50 == 0:
            print(f"  Match {i}/{len(found_ids)}...")
        try:
            filepath = os.path.join(EVENTS_DIR, f'{mid}.json')
            if not os.path.exists(filepath):
                continue
            with open(filepath) as f:
                events = json.load(f)
            info = match_lookup[mid]
            home, away = info['home'], info['away']
            hs, aws = info['home_score'], info['away_score']
            for team in [home, away]:
                is_home = (team == home)
                team_score = hs if is_home else aws
                opp_score = aws if is_home else hs
                if team_score > opp_score:
                    outcome = 'Win'
                elif team_score < opp_score:
                    outcome = 'Loss'
                else:
                    outcome = 'Draw'
                s = analyze_match_team(events, team)
                if s:
                    s['team'] = team
                    s['match_id'] = mid
                    s['outcome'] = outcome
                    s['is_home'] = is_home
                    s['goals_scored'] = team_score
                    s['goals_conceded'] = opp_score
                    s['date'] = info['date']
                    results.append(s)
        except Exception as e:
            print(f"  Error on match {mid}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(cache_path, index=False)
    print(f"Done! {len(df)} team-match observations saved.")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_ci(data, n_boot=10000, ci=95):
    """Bootstrap confidence interval for the median."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    medians = np.array([np.median(np.random.choice(data, size=len(data), replace=True))
                        for _ in range(n_boot)])
    alpha = (100 - ci) / 2
    return np.percentile(medians, alpha), np.percentile(medians, 100 - alpha)


def cliff_delta(x, y):
    """Cliff's delta effect size (non-parametric)."""
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    more = sum((xi > yj) for xi in x for yj in y)
    less = sum((xi < yj) for xi in x for yj in y)
    return (more - less) / (n_x * n_y)


def interpret_cliff(d):
    """Interpret Cliff's delta magnitude."""
    d = abs(d)
    if d < 0.147:
        return 'negligible'
    elif d < 0.33:
        return 'small'
    elif d < 0.474:
        return 'medium'
    else:
        return 'large'


def run_statistical_comparisons(df):
    """Run Kruskal-Wallis + post-hoc Dunn tests with effect sizes."""
    metrics = [
        ('avg_density', 'Network Density'),
        ('avg_clustering', 'Clustering Coefficient'),
        ('avg_transitivity', 'Transitivity'),
        ('avg_modularity', 'Modularity'),
        ('avg_degree', 'Average Degree'),
        ('avg_n_communities', 'Number of Communities'),
        ('centrality_variability', 'Centrality Variability'),
        ('total_passes', 'Total Passes'),
    ]

    stat_results = []
    for col, label in metrics:
        groups = {o: df[df['outcome'] == o][col].dropna().values for o in ['Win', 'Draw', 'Loss']}
        h_stat, p_val = stats.kruskal(*groups.values())

        # Pairwise Mann-Whitney U with Bonferroni correction
        pairs = [('Win', 'Draw'), ('Win', 'Loss'), ('Draw', 'Loss')]
        pairwise = {}
        for a, b in pairs:
            u_stat, p_mw = stats.mannwhitneyu(groups[a], groups[b], alternative='two-sided')
            p_adj = min(p_mw * 3, 1.0)  # Bonferroni
            d = cliff_delta(groups[a], groups[b])
            pairwise[f'{a} vs {b}'] = {
                'U': u_stat, 'p_raw': p_mw, 'p_adj': p_adj,
                'cliff_d': d, 'effect': interpret_cliff(d)
            }

        # Bootstrap CIs for each group
        cis = {}
        for o in ['Win', 'Draw', 'Loss']:
            lo, hi = bootstrap_ci(groups[o])
            cis[o] = (lo, hi)

        stat_results.append({
            'metric': label,
            'col': col,
            'kruskal_H': h_stat,
            'kruskal_p': p_val,
            'medians': {o: np.median(groups[o]) for o in ['Win', 'Draw', 'Loss']},
            'bootstrap_ci': cis,
            'pairwise': pairwise,
        })

    return stat_results


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════

def fig_season_boxplots_notched(df, stat_results):
    """Figure 1: Notched boxplots for season-wide comparison by outcome."""
    metrics_to_plot = [
        ('avg_density', 'Network Density'),
        ('avg_clustering', 'Clustering Coeff.'),
        ('avg_transitivity', 'Transitivity'),
        ('total_passes', 'Total Passes'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
    for ax, (col, label) in zip(axes, metrics_to_plot):
        data = [df[df['outcome'] == o][col].dropna() for o in ['Win', 'Draw', 'Loss']]
        bp = ax.boxplot(data, labels=['W', 'D', 'L'], patch_artist=True,
                        notch=True, widths=0.55, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for patch, outcome in zip(bp['boxes'], ['Win', 'Draw', 'Loss']):
            patch.set_facecolor(OUTCOME_COLORS[outcome])
            patch.set_alpha(0.65)

        # Find stat result for this metric
        sr = next((s for s in stat_results if s['col'] == col), None)
        if sr:
            sig = '***' if sr['kruskal_p'] < 0.001 else '**' if sr['kruskal_p'] < 0.01 else '*' if sr['kruskal_p'] < 0.05 else 'n.s.'
            ax.set_title(f'{label}\n(H={sr["kruskal_H"]:.1f}, {sig})', fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle('Network Metrics by Match Outcome (La Liga 2015/16, n=760)',
                 fontsize=10, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_season_boxplots.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig1_season_boxplots.png'))
    plt.close(fig)
    print("  Saved fig1_season_boxplots")


def fig_barca_madrid_comparison(df, stat_results):
    """Figure 2: Barcelona vs Real Madrid vs League with bootstrap CIs."""
    barca = df[df['team'] == 'Barcelona']
    madrid = df[df['team'] == 'Real Madrid']

    metrics = [
        ('avg_density', 'Density'),
        ('avg_clustering', 'Clustering'),
        ('avg_transitivity', 'Transitivity'),
        ('avg_degree', 'Avg Degree'),
        ('total_passes', 'Total Passes'),
        ('centrality_variability', 'Centrality Var.'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, (col, label) in zip(axes.flat, metrics):
        groups = {
            'Barcelona': barca[col].dropna().values,
            'Real Madrid': madrid[col].dropna().values,
            'League': df[col].dropna().values,
        }
        colors = ['#A50044', '#0067B1', '#888888']
        positions = [1, 2, 3]
        bp = ax.boxplot([groups['Barcelona'], groups['Real Madrid'], groups['League']],
                        positions=positions, labels=['Barça', 'Madrid', 'League'],
                        patch_artist=True, notch=True, widths=0.5, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        # Add bootstrap CI markers
        for pos, (name, vals) in zip(positions, groups.items()):
            lo, hi = bootstrap_ci(vals)
            med = np.median(vals)
            ax.plot([pos - 0.15, pos + 0.15], [med, med], 'k-', lw=2)

        ax.set_title(label, fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle('Barcelona vs Real Madrid vs League (La Liga 2015/16)',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_barca_madrid_league.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig2_barca_madrid_league.png'))
    plt.close(fig)
    print("  Saved fig2_barca_madrid_league")


def fig_clasico_temporal(match_id, match_label, filename_prefix):
    """Figure 3/4: El Clasico temporal deep dive (combined panel)."""
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
    rc_player = red_cards[0]['player'].split()[-1] if red_cards else None

    team_colors = ['#E53935', '#1E88E5']
    if 'Barcelona' in home:
        team_colors = ['#A50044', '#0067B1']
    elif 'Barcelona' in away:
        team_colors = ['#0067B1', '#A50044']

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.3)

    # Row 1: Pass volume + Centrality (top 3 per team)
    ax_vol = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(teams_list):
        df_t = df[df['team'] == t]
        ppb = df_t.groupby('time_bin').size()
        ax_vol.plot(ppb.index * TIME_WINDOW, ppb.values, 'o-',
                    color=team_colors[i], label=t.split()[-1] if len(t.split()) > 1 else t,
                    lw=1.5, ms=4)
    ax_vol.axvline(45, color='gray', ls=':', alpha=0.5, label='Half')
    if rc_min:
        ax_vol.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7,
                       label=f'Red ({rc_player})')
    ax_vol.set_xlabel('Match Time (min)')
    ax_vol.set_ylabel('Passes per Window')
    ax_vol.set_title('(a) Pass Volume', fontsize=9, fontweight='bold')
    ax_vol.legend(fontsize=7, loc='upper right')

    # Centrality panel for each team
    for idx, (t, tc) in enumerate(zip(teams_list, team_colors)):
        ax_cent = fig.add_subplot(gs[0, 1]) if idx == 0 else fig.add_subplot(gs[1, 0])
        nets_t = build_temporal_networks(df, t)
        cents_t = {tb: compute_centralities(G) for tb, G in nets_t.items()}
        top = get_top_players(cents_t, 'norm_degree', 3)
        bins = sorted(cents_t.keys())
        mins = [b * TIME_WINDOW for b in bins]
        cent_colors = ['#1E88E5', '#E53935', '#43A047']
        for j, p in enumerate(top):
            vals = [cents_t[b].get(p, {}).get('norm_degree', 0) for b in bins]
            short_name = p.split()[-1]
            ax_cent.plot(mins, vals, 'o-', color=cent_colors[j], label=short_name, lw=1.2, ms=3)
        ax_cent.axvline(45, color='gray', ls=':', alpha=0.5)
        if rc_min:
            ax_cent.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7)
        panel_letter = '(b)' if idx == 0 else '(c)'
        ax_cent.set_title(f'{panel_letter} Centrality: {t}', fontsize=9, fontweight='bold')
        ax_cent.set_xlabel('Match Time (min)')
        ax_cent.set_ylabel('Norm. Degree Centrality')
        ax_cent.legend(fontsize=7, ncol=1)

    # Row 2 right: Network metrics over time
    ax_met = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(teams_list):
        nets_t = build_temporal_networks(df, t)
        met_t = {tb: network_metrics(G) for tb, G in nets_t.items()}
        bins = sorted(met_t.keys())
        mins = [b * TIME_WINDOW for b in bins]
        short = t.split()[-1] if len(t.split()) > 1 else t
        ax_met.plot(mins, [met_t[b]['density'] for b in bins], 'o-',
                    color=team_colors[i], label=f'{short} density', lw=1.2, ms=3)
        ax_met.plot(mins, [met_t[b]['clustering'] for b in bins], 's--',
                    color=team_colors[i], label=f'{short} clustering', lw=1, ms=3, alpha=0.7)
    ax_met.axvline(45, color='gray', ls=':', alpha=0.5)
    if rc_min:
        ax_met.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7)
    ax_met.set_title('(d) Network Density & Clustering', fontsize=9, fontweight='bold')
    ax_met.set_xlabel('Match Time (min)')
    ax_met.set_ylabel('Metric Value')
    ax_met.legend(fontsize=6, ncol=2)

    # Row 3: Community detection + Pitch network
    ax_comm = fig.add_subplot(gs[2, 0])
    for i, t in enumerate(teams_list):
        nets_t = build_temporal_networks(df, t)
        comm_data = {}
        for tb, G in nets_t.items():
            c, m = detect_communities(G)
            comm_data[tb] = {'n': len(c), 'mod': m}
        bins = sorted(comm_data.keys())
        mins = [b * TIME_WINDOW for b in bins]
        short = t.split()[-1] if len(t.split()) > 1 else t
        ax_comm.plot(mins, [comm_data[b]['mod'] for b in bins], 'o-',
                     color=team_colors[i], label=f'{short}', lw=1.2, ms=3)
    ax_comm.axvline(45, color='gray', ls=':', alpha=0.5)
    if rc_min:
        ax_comm.axvline(rc_min, color='red', ls='--', lw=1.5, alpha=0.7)
    ax_comm.set_title('(e) Modularity Over Time', fontsize=9, fontweight='bold')
    ax_comm.set_xlabel('Match Time (min)')
    ax_comm.set_ylabel('Modularity')
    ax_comm.legend(fontsize=7)

    # Pitch network for the winning team (full match)
    ax_pitch = fig.add_subplot(gs[2, 1])
    winner = home if hs > aws else away
    w_color = team_colors[0] if winner == home else team_colors[1]
    plot_passing_network(df, winner, ax=ax_pitch, color=w_color, min_passes=3)
    ax_pitch.set_title(f'(f) {winner} Passing Network', fontsize=9, fontweight='bold', color='white')

    fig.suptitle(f'{match_label}: {home} {hs} - {aws} {away}',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(FIG_DIR, f'{filename_prefix}.pdf'))
    fig.savefig(os.path.join(FIG_DIR, f'{filename_prefix}.png'))
    plt.close(fig)
    print(f"  Saved {filename_prefix}")


def fig_community_visualization(match_id, team_name, time_bin_target, filename):
    """Appendix figure: Community structure on pitch for a specific time window."""
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
            short = player.split()[-1]
            ax.annotate(short, (pos_data.loc[player, 'x'], pos_data.loc[player, 'y']),
                        xytext=(0, -18), textcoords='offset points', ha='center',
                        fontsize=8, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.8))

    # Legend for communities
    for i, comm in enumerate(comms):
        members = [p.split()[-1] for p in comm]
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


def generate_stat_table(stat_results):
    """Generate LaTeX table for statistical results."""
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\footnotesize')
    lines.append(r'\caption{Statistical comparison of network metrics by match outcome. '
                 r'Kruskal-Wallis $H$ statistic and $p$-value test for differences across Win, Draw, '
                 r'and Loss groups. Cliff\'s $\delta$ measures effect size for pairwise comparisons.}')
    lines.append(r'\label{tab:stats}')
    lines.append(r'\begin{tabular}{lccccccc}')
    lines.append(r'\hline')
    lines.append(r'Metric & Med(W) & Med(D) & Med(L) & $H$ & $p$ & $\delta_{WL}$ & Effect \\')
    lines.append(r'\hline')

    for sr in stat_results:
        mw = sr['medians']['Win']
        md = sr['medians']['Draw']
        ml = sr['medians']['Loss']
        h = sr['kruskal_H']
        p = sr['kruskal_p']
        wl = sr['pairwise']['Win vs Loss']
        d = wl['cliff_d']
        eff = wl['effect']
        p_str = f'{p:.3f}' if p >= 0.001 else '$<$0.001'
        lines.append(f'{sr["metric"]} & {mw:.3f} & {md:.3f} & {ml:.3f} & '
                     f'{h:.1f} & {p_str} & {d:.3f} & {eff} \\\\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(os.path.join(TABLE_DIR, 'stat_comparison.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print("  Saved stat_comparison.tex")


def generate_barca_madrid_table(df):
    """Generate comparison table for Barcelona vs Real Madrid."""
    barca = df[df['team'] == 'Barcelona']
    madrid = df[df['team'] == 'Real Madrid']
    league = df

    metrics = [
        ('avg_density', 'Network Density'),
        ('avg_clustering', 'Clustering Coeff.'),
        ('avg_transitivity', 'Transitivity'),
        ('avg_degree', 'Average Degree'),
        ('avg_modularity', 'Modularity'),
        ('centrality_variability', 'Centrality Var.'),
        ('total_passes', 'Total Passes'),
    ]

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\footnotesize')
    lines.append(r'\caption{Median network metrics: Barcelona, Real Madrid, and league average '
                 r'across the 2015/16 La Liga season. 95\% bootstrap confidence intervals in brackets.}')
    lines.append(r'\label{tab:barca_madrid}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\hline')
    lines.append(r'Metric & Barcelona & Real Madrid & League \\')
    lines.append(r'\hline')

    for col, label in metrics:
        for name, subset in [('b', barca), ('r', madrid), ('l', league)]:
            vals = subset[col].dropna().values
            med = np.median(vals)
            lo, hi = bootstrap_ci(vals)
            if name == 'b':
                line = f'{label} & {med:.3f} [{lo:.3f}, {hi:.3f}]'
            elif name == 'r':
                line += f' & {med:.3f} [{lo:.3f}, {hi:.3f}]'
            else:
                line += f' & {med:.3f} [{lo:.3f}, {hi:.3f}] \\\\'
        lines.append(line)

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(os.path.join(TABLE_DIR, 'barca_madrid_comparison.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print("  Saved barca_madrid_comparison.tex")


def fig_appendix_extra_boxplots(df, stat_results):
    """Appendix figure: All 8 metrics with notched boxplots."""
    metrics_to_plot = [
        ('avg_density', 'Network Density'),
        ('avg_clustering', 'Clustering Coeff.'),
        ('avg_transitivity', 'Transitivity'),
        ('avg_modularity', 'Modularity'),
        ('avg_degree', 'Average Degree'),
        ('avg_n_communities', 'Num. Communities'),
        ('centrality_variability', 'Centrality Var.'),
        ('total_passes', 'Total Passes'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    for ax, (col, label) in zip(axes.flat, metrics_to_plot):
        data = [df[df['outcome'] == o][col].dropna() for o in ['Win', 'Draw', 'Loss']]
        bp = ax.boxplot(data, labels=['W', 'D', 'L'], patch_artist=True,
                        notch=True, widths=0.55, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for patch, outcome in zip(bp['boxes'], ['Win', 'Draw', 'Loss']):
            patch.set_facecolor(OUTCOME_COLORS[outcome])
            patch.set_alpha(0.65)
        sr = next((s for s in stat_results if s['col'] == col), None)
        if sr:
            sig = '***' if sr['kruskal_p'] < 0.001 else '**' if sr['kruskal_p'] < 0.01 else '*' if sr['kruskal_p'] < 0.05 else 'n.s.'
            ax.set_title(f'{label}\n(H={sr["kruskal_H"]:.1f}, {sig})', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

    fig.suptitle('All Network Metrics by Match Outcome (La Liga 2015/16, n=760)',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'figA1_all_boxplots.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'figA1_all_boxplots.png'))
    plt.close(fig)
    print("  Saved figA1_all_boxplots")


def fig_appendix_pitch_networks(match_id, match_label, filename):
    """Appendix: Both teams' pitch networks side by side."""
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
#  METHODS FIGURE
# ═══════════════════════════════════════════════════════════════════════

def fig_methods_pipeline():
    """Figure for Methods section: analysis pipeline schematic."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')

    steps = [
        'StatsBomb\nEvent Data',
        'Extract\nPasses',
        '5-min\nWindows',
        'Directed\nNetworks',
        'Network\nMetrics',
        'Statistical\nComparison',
    ]
    n = len(steps)
    box_w, box_h = 1.2, 0.7
    gap = 0.4
    total_w = n * box_w + (n - 1) * gap
    start_x = (10 - total_w) / 2

    for i, step in enumerate(steps):
        x = start_x + i * (box_w + gap)
        y = 0.5
        color = '#1565C0' if i < 4 else '#2E7D32'
        rect = plt.Rectangle((x, y), box_w, box_h, facecolor=color,
                              edgecolor='white', linewidth=1.5, alpha=0.85,
                              transform=ax.transData, zorder=2)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h / 2, step, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', transform=ax.transData, zorder=3)
        if i < n - 1:
            ax.annotate('', xy=(x + box_w + gap * 0.1, y + box_h / 2),
                        xytext=(x + box_w + gap * 0.9, y + box_h / 2),
                        arrowprops=dict(arrowstyle='<-', color='#333', lw=1.5),
                        transform=ax.transData)

    ax.set_xlim(start_x - 0.3, start_x + total_w + 0.3)
    ax.set_ylim(0, 1.8)
    ax.set_title('Analysis Pipeline', fontsize=10, fontweight='bold')
    fig.savefig(os.path.join(FIG_DIR, 'fig_methods_pipeline.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig_methods_pipeline.png'))
    plt.close(fig)
    print("  Saved fig_methods_pipeline")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("TEMPORAL NETWORK ANALYSIS — La Liga 2015/16")
    print("=" * 60)

    # Phase 1: Season-wide analysis
    print("\n[1] Running season-wide analysis...")
    df = run_season_analysis()
    print(f"  Observations: {len(df)}")
    print(f"  Outcomes: {df['outcome'].value_counts().to_dict()}")
    print(f"  Teams: {df['team'].nunique()}")

    # Phase 2: Statistical tests
    print("\n[2] Running statistical comparisons...")
    stat_results = run_statistical_comparisons(df)
    for sr in stat_results:
        sig = 'YES' if sr['kruskal_p'] < 0.05 else 'no'
        print(f"  {sr['metric']:30s} H={sr['kruskal_H']:7.1f}  p={sr['kruskal_p']:.4f}  sig={sig}")

    # Save stat results
    import pickle
    with open(os.path.join(RESULTS_DIR, 'stat_results.pkl'), 'wb') as f:
        pickle.dump(stat_results, f)

    # Phase 3: Generate figures
    print("\n[3] Generating figures...")
    fig_season_boxplots_notched(df, stat_results)
    fig_barca_madrid_comparison(df, stat_results)
    fig_clasico_temporal(266424, "El Cl\u00e1sico 1 (Bernab\u00e9u)", 'fig3_clasico1')
    fig_clasico_temporal(267533, "El Cl\u00e1sico 2 (Camp Nou)", 'fig4_clasico2')
    fig_methods_pipeline()

    # Phase 4: Generate tables
    print("\n[4] Generating tables...")
    generate_stat_table(stat_results)
    generate_barca_madrid_table(df)

    # Phase 5: Appendix figures
    print("\n[5] Generating appendix figures...")
    fig_appendix_extra_boxplots(df, stat_results)
    fig_appendix_pitch_networks(266424, "El Cl\u00e1sico 1", 'figA2_ec1_pitch.png')
    fig_appendix_pitch_networks(267533, "El Cl\u00e1sico 2", 'figA3_ec2_pitch.png')

    # Community visualizations (prof feedback)
    fig_community_visualization(266424, 'Barcelona', 4, 'figA4_barca_communities.png')
    fig_community_visualization(266424, 'Real Madrid', 4, 'figA5_madrid_communities.png')

    # Phase 6: Summary statistics
    print("\n[6] Summary stats...")
    print(f"\nBarcelona season record:")
    b = df[df['team'] == 'Barcelona']
    print(f"  {len(b[b.outcome=='Win'])}W {len(b[b.outcome=='Draw'])}D {len(b[b.outcome=='Loss'])}L")
    r = df[df['team'] == 'Real Madrid']
    print(f"Real Madrid season record:")
    print(f"  {len(r[r.outcome=='Win'])}W {len(r[r.outcome=='Draw'])}D {len(r[r.outcome=='Loss'])}L")

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables: {TABLE_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 60)
