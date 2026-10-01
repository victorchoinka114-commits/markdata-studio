"""Tennis Data Lab : données de démonstration, court à l'échelle et modèles.

Repère du court (en mètres) : origine au centre du filet,
x le long du court (-11.885 → +11.885), y dans la largeur (-5.485 → +5.485).
Le joueur analysé frappe depuis x < 0 ; ses balles rebondissent côté x > 0.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Dimensions officielles ITF (m)
HALF_LENGTH = 23.77 / 2          # 11.885
HALF_WIDTH_DOUBLES = 10.97 / 2   # 5.485
HALF_WIDTH_SINGLES = 8.23 / 2    # 4.115
SERVICE_LINE = 6.40              # distance filet -> ligne de service
NET_HEIGHT_CENTER = 0.914

PLAYERS = ["A. Moreau", "L. Fontaine", "J. Diallo", "M. Garnier", "S. Nguyen", "T. Rousseau",
           "K. Benali", "P. Chevalier", "E. Lambert", "R. Faure", "N. Perrin", "C. Dubois"]
SURFACES = ["Dur", "Terre battue", "Gazon"]


# --------------------------------------------------------------------------- #
# Court 2D à l'échelle
# --------------------------------------------------------------------------- #
def court_figure(surface: str = "Dur", height: int = 460, show_doubles: bool = True) -> go.Figure:
    """Court de tennis 2D à l'échelle (Plotly), prêt pour superposer des données."""
    colors = {"Dur": ("#3C638E", "#4A7DB5"), "Terre battue": ("#B5562B", "#C8693A"),
              "Gazon": ("#3F7D3A", "#4E9A47")}
    outer, inner = colors.get(surface, colors["Dur"])
    fig = go.Figure()
    margin_x, margin_y = 6.4, 3.66  # dégagements arrière et latéraux
    line = dict(color="white", width=2)
    shapes = [
        dict(type="rect", x0=-HALF_LENGTH - margin_x, x1=HALF_LENGTH + margin_x,
             y0=-HALF_WIDTH_DOUBLES - margin_y, y1=HALF_WIDTH_DOUBLES + margin_y,
             fillcolor=outer, line=dict(width=0), layer="below"),
        dict(type="rect", x0=-HALF_LENGTH, x1=HALF_LENGTH, y0=-HALF_WIDTH_DOUBLES, y1=HALF_WIDTH_DOUBLES,
             fillcolor=inner, line=line if show_doubles else dict(width=0), layer="below"),
        # Couloirs simple
        dict(type="line", x0=-HALF_LENGTH, x1=HALF_LENGTH, y0=HALF_WIDTH_SINGLES, y1=HALF_WIDTH_SINGLES, line=line),
        dict(type="line", x0=-HALF_LENGTH, x1=HALF_LENGTH, y0=-HALF_WIDTH_SINGLES, y1=-HALF_WIDTH_SINGLES, line=line),
        # Lignes de service
        dict(type="line", x0=SERVICE_LINE, x1=SERVICE_LINE, y0=-HALF_WIDTH_SINGLES, y1=HALF_WIDTH_SINGLES, line=line),
        dict(type="line", x0=-SERVICE_LINE, x1=-SERVICE_LINE, y0=-HALF_WIDTH_SINGLES, y1=HALF_WIDTH_SINGLES, line=line),
        # Ligne médiane de service
        dict(type="line", x0=-SERVICE_LINE, x1=SERVICE_LINE, y0=0, y1=0, line=line),
        # Marques centrales
        dict(type="line", x0=HALF_LENGTH - 0.1, x1=HALF_LENGTH, y0=0, y1=0, line=line),
        dict(type="line", x0=-HALF_LENGTH, x1=-HALF_LENGTH + 0.1, y0=0, y1=0, line=line),
        # Filet (prolongé jusqu'aux poteaux à 0.914 m hors du double)
        dict(type="line", x0=0, x1=0, y0=-HALF_WIDTH_DOUBLES - 0.914, y1=HALF_WIDTH_DOUBLES + 0.914,
             line=dict(color="#F2F2F2", width=4, dash="dot")),
    ]
    if not show_doubles:
        shapes.append(dict(type="rect", x0=-HALF_LENGTH, x1=HALF_LENGTH, y0=-HALF_WIDTH_SINGLES,
                           y1=HALF_WIDTH_SINGLES, line=line, layer="below"))
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(range=[-HALF_LENGTH - margin_x, HALF_LENGTH + margin_x], showgrid=False, zeroline=False,
                   visible=False, constrain="domain"),
        yaxis=dict(range=[-HALF_WIDTH_DOUBLES - margin_y, HALF_WIDTH_DOUBLES + margin_y], showgrid=False,
                   zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor=outer, paper_bgcolor="rgba(0,0,0,0)", height=height,
        margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.02),
    )
    return fig


def add_bounces(fig: go.Figure, df: pd.DataFrame, color_col: str | None = None, name: str = "Rebonds",
                size: int = 7) -> go.Figure:
    """Superpose des points d'impact sur le court."""
    if color_col and color_col in df:
        palette = ["#FFD23F", "#EE4266", "#3BCEAC", "#F7F7F7", "#540D6E", "#0EAD69"]
        for i, (val, grp) in enumerate(df.groupby(color_col)):
            fig.add_trace(go.Scatter(x=grp["x"], y=grp["y"], mode="markers", name=str(val),
                                     marker=dict(size=size, color=palette[i % len(palette)], opacity=0.8,
                                                 line=dict(width=0.5, color="#222"))))
    else:
        fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="markers", name=name,
                                 marker=dict(size=size, color="#FFD23F", opacity=0.8,
                                             line=dict(width=0.5, color="#222"))))
    return fig


def add_heatmap(fig: go.Figure, df: pd.DataFrame, nbins: int = 25) -> go.Figure:
    """Carte de densité des rebonds (contours 2D) au-dessus du court."""
    fig.add_trace(go.Histogram2dContour(
        x=df["x"], y=df["y"], colorscale="YlOrRd", showscale=True, opacity=0.65,
        ncontours=nbins, contours=dict(coloring="fill", showlines=False),
        colorbar=dict(title="Densité"), name="Densité",
        hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>densité=%{z}<extra></extra>",
    ))
    return fig


# --------------------------------------------------------------------------- #
# Jeux de données de démonstration
# --------------------------------------------------------------------------- #
def demo_match_stats(n_matches: int = 300, seed: int = 11) -> pd.DataFrame:
    """Statistiques de matchs (une ligne par joueur et par match)."""
    rng = np.random.default_rng(seed)
    skill = {p: rng.normal(0, 1) for p in PLAYERS}
    rows = []
    for m in range(n_matches):
        p1, p2 = rng.choice(PLAYERS, 2, replace=False)
        surface = rng.choice(SURFACES, p=[0.5, 0.35, 0.15])
        diff = skill[p1] - skill[p2] + rng.normal(0, 0.8)
        for player, opp, d in ((p1, p2, diff), (p2, p1, -diff)):
            serve_bonus = 0.04 if surface == "Gazon" else (-0.03 if surface == "Terre battue" else 0)
            first_in = np.clip(rng.normal(0.62, 0.05), 0.45, 0.8)
            first_won = np.clip(rng.normal(0.72 + 0.04 * d + serve_bonus, 0.05), 0.5, 0.9)
            second_won = np.clip(rng.normal(0.51 + 0.03 * d, 0.05), 0.3, 0.7)
            serve_pts = int(rng.integers(55, 110))
            bp_faced = int(rng.poisson(max(1, 6 - 2 * d)))
            bp_opp = int(rng.poisson(max(1, 6 + 2 * d)))
            rows.append({
                "match_id": f"M{m + 1:04d}", "joueur": player, "adversaire": opp, "surface": surface,
                "aces": int(rng.poisson(5 + 3 * (surface == "Gazon") + 1.5 * max(d, 0))),
                "doubles_fautes": int(rng.poisson(3)),
                "pct_1ere_balle": round(first_in * 100, 1),
                "pct_pts_gagnes_1ere": round(first_won * 100, 1),
                "pct_pts_gagnes_2nde": round(second_won * 100, 1),
                "pct_pts_gagnes_retour": round(np.clip(rng.normal(38 + 4 * d, 4), 20, 60), 1),
                "points_au_service": serve_pts,
                "coups_gagnants": int(rng.poisson(max(5, 25 + 5 * d))),
                "fautes_directes": int(rng.poisson(max(5, 28 - 4 * d))),
                "balles_break_sauvees": int(min(bp_faced, rng.binomial(bp_faced, np.clip(0.6 + 0.08 * d, 0.1, 0.95)))),
                "balles_break_concedees": bp_faced,
                "balles_break_converties": int(rng.binomial(bp_opp, np.clip(0.4 + 0.08 * d, 0.05, 0.9))),
                "balles_break_obtenues": bp_opp,
                "vitesse_moy_service_kmh": round(rng.normal(185 + 6 * skill[player], 8), 1),
                "longueur_moy_rallye": round(np.clip(rng.normal(4.8 + (1.5 if surface == "Terre battue" else 0), 0.8), 2, 9), 1),
                "victoire": int(d > 0),
            })
    return pd.DataFrame(rows)


SHOT_TYPES = ["Service", "Coup droit", "Revers", "Volée"]


def demo_tracking(n_shots: int = 1500, seed: int = 21, surface: str = "Dur") -> pd.DataFrame:
    """Données de tracking fictives (façon Hawk-Eye) : rebonds des frappes du joueur analysé.

    Le joueur frappe depuis x < 0 ; les rebonds se situent côté adversaire (x > 0).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_shots):
        shot = rng.choice(SHOT_TYPES, p=[0.22, 0.4, 0.3, 0.08])
        side = rng.choice(["Égalité", "Avantage"])
        if shot == "Service":
            target = rng.choice(["Extérieur", "Corps", "T"], p=[0.4, 0.2, 0.4])
            y_center = {"Extérieur": 3.4, "Corps": 2.0, "T": 0.4}[target]
            y = (y_center if side == "Égalité" else -y_center) + rng.normal(0, 0.55)
            x = SERVICE_LINE - abs(rng.normal(1.1, 0.8))
            speed = rng.normal(188, 12)
            spin = rng.normal(1800, 500)
            rally = int(rng.choice([1, 1, 2, 3, 4, 5, 6]))
        else:
            depth = rng.beta(4, 2) * (HALF_LENGTH + 0.6)
            x = np.clip(depth, 0.5, HALF_LENGTH + 1.2)
            bias = 1.2 if shot == "Coup droit" else -0.8
            y = rng.normal(bias, 2.4) if shot != "Volée" else rng.normal(0, 3)
            if shot == "Volée":
                x = np.clip(rng.normal(5.5, 2), 0.5, HALF_LENGTH + 0.5)
            speed = rng.normal({"Coup droit": 120, "Revers": 108, "Volée": 85}[shot], 12)
            spin = rng.normal({"Coup droit": 2600, "Revers": 1900, "Volée": 600}[shot], 500)
            rally = int(rng.integers(2, 18))
            target = None
        is_in = (0 < x <= (SERVICE_LINE if shot == "Service" else HALF_LENGTH)) and abs(y) <= HALF_WIDTH_SINGLES
        if shot == "Service":
            is_in = is_in and ((y >= 0) if side == "Égalité" else (y <= 0))
        # Probabilité de gagner le point : plus la balle est profonde / rapide / près des lignes, mieux c'est
        depth_factor = x / HALF_LENGTH
        width_factor = abs(y) / HALF_WIDTH_SINGLES
        logit = -0.6 + 1.6 * depth_factor + 0.9 * width_factor + 0.015 * (speed - 120)
        if shot == "Service":
            logit = 0.3 + 0.02 * (speed - 185) + 0.5 * (target != "Corps")
        won = bool(is_in and rng.random() < 1 / (1 + np.exp(-logit)))
        rows.append({
            "shot_id": i + 1, "type_coup": shot, "cote": side, "cible_service": target,
            "x": round(float(x), 2), "y": round(float(y), 2),
            "vitesse_kmh": round(float(speed), 1), "effet_rpm": int(max(spin, 0)),
            "dans_le_court": is_in, "point_gagne": won, "longueur_rallye": rally,
        })
    return pd.DataFrame(rows)


def depth_zone(x: float) -> str:
    """Zone de profondeur côté adversaire (distance au filet)."""
    if x <= SERVICE_LINE:
        return "Courte (carré de service)"
    if x <= HALF_LENGTH - 2.5:
        return "Moyenne"
    return "Profonde (< 2,5 m de la ligne)"


def lateral_zone(y: float) -> str:
    if y > HALF_WIDTH_SINGLES / 3:
        return "Gauche"
    if y < -HALF_WIDTH_SINGLES / 3:
        return "Droite"
    return "Centre"


def zone_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Taux de réussite par zone (profondeur x latéralité) des frappes en jeu."""
    rally = df[(df["type_coup"] != "Service") & df["dans_le_court"]].copy()
    rally["profondeur"] = rally["x"].map(depth_zone)
    rally["couloir"] = rally["y"].map(lateral_zone)
    out = rally.groupby(["profondeur", "couloir"]).agg(
        frappes=("shot_id", "count"), pct_points_gagnes=("point_gagne", "mean"),
        vitesse_moy=("vitesse_kmh", "mean")).reset_index()
    out["pct_points_gagnes"] = (out["pct_points_gagnes"] * 100).round(1)
    out["vitesse_moy"] = out["vitesse_moy"].round(1)
    return out


# --------------------------------------------------------------------------- #
# Cas équipementier : styles de jeu
# --------------------------------------------------------------------------- #
STYLES = {
    "Attaquant de fond de court": dict(net=0.08, rally=5.0, serve=192, ratio=1.1, spin=2900, ue=30),
    "Contreur / Défenseur": dict(net=0.05, rally=7.5, serve=172, ratio=0.7, spin=2400, ue=18),
    "Serveur-volleyeur": dict(net=0.38, rally=3.0, serve=205, ratio=1.4, spin=1800, ue=24),
    "Joueur complet (all-court)": dict(net=0.18, rally=5.2, serve=188, ratio=1.0, spin=2300, ue=22),
}

RACKET_RECO = {
    "Attaquant de fond de court": {
        "raquette": "Raquette « puissance-contrôle » 300-305 g, tamis 98-100 in², équilibre ~32 cm, plan de cordage ouvert 16x19",
        "cordage": "Monofilament polyester profilé (effet), 1,25 mm, tension 22-24 kg",
        "argument": "Maximiser l'effet lifté et la vitesse de balle tout en gardant du contrôle en fond de court.",
    },
    "Contreur / Défenseur": {
        "raquette": "Raquette tolérante 285-300 g, tamis 100-102 in², légèrement en tête, plan 16x19",
        "cordage": "Hybride polyester / multifilament, tension 21-23 kg",
        "argument": "Aide à la longueur de balle en défense et confort sur les longs échanges.",
    },
    "Serveur-volleyeur": {
        "raquette": "Raquette lourde et maniable 310-320 g, tamis 95-98 in², équilibre en manche, plan fermé 18x20",
        "cordage": "Multifilament ou boyau naturel, tension 23-25 kg",
        "argument": "Stabilité à la volée, précision et toucher au filet, puissance au service.",
    },
    "Joueur complet (all-court)": {
        "raquette": "Raquette polyvalente 300 g, tamis 98 in², équilibre neutre, plan 16x19",
        "cordage": "Hybride polyester (montants) / multifilament (travers), tension 22-24 kg",
        "argument": "Compromis puissance / toucher pour varier le jeu sur toutes les zones du court.",
    },
}
STYLE_FEATURES = ["pct_montees_filet", "longueur_moy_rallye", "vitesse_service_kmh",
                  "ratio_gagnants_fautes", "effet_moyen_rpm", "fautes_directes_par_set"]


def demo_player_profiles(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """Profils de joueurs étiquetés par style (pour la classification)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        style = rng.choice(list(STYLES))
        c = STYLES[style]
        rows.append({
            "pct_montees_filet": round(float(np.clip(rng.normal(c["net"], 0.06), 0, 0.8)), 3),
            "longueur_moy_rallye": round(float(np.clip(rng.normal(c["rally"], 0.9), 1.5, 12)), 2),
            "vitesse_service_kmh": round(float(rng.normal(c["serve"], 9)), 1),
            "ratio_gagnants_fautes": round(float(np.clip(rng.normal(c["ratio"], 0.2), 0.2, 3)), 2),
            "effet_moyen_rpm": int(rng.normal(c["spin"], 300)),
            "fautes_directes_par_set": round(float(np.clip(rng.normal(c["ue"], 4), 3, 50)), 1),
            "style": style,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cas Fan Engagement : probabilité de victoire (modèle de Markov)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def p_game(p: float, a: int = 0, b: int = 0) -> float:
    """Probabilité que le serveur gagne le jeu, à partir du score (a, b) en points.

    p = probabilité de gagner un point sur son service (supposée i.i.d.).
    """
    q = 1 - p
    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0
    if a >= 3 and b >= 3:
        deuce = p * p / (p * p + q * q)
        if a == b:
            return deuce
        return p + q * deuce if a > b else p * deuce
    return p * p_game(p, a + 1, b) + q * p_game(p, a, b + 1)


def _tb_server_is_first(n_points_played: int) -> bool:
    """Au tie-break : 1er point par le 1er serveur, puis alternance tous les 2 points."""
    return n_points_played == 0 or ((n_points_played - 1) // 2) % 2 == 1


@lru_cache(maxsize=None)
def p_tiebreak(pa: float, pb: float, a: int = 0, b: int = 0, a_serves_first: bool = True) -> float:
    """Probabilité que A gagne le tie-break (7 points, 2 d'écart)."""
    if a >= 7 and a - b >= 2:
        return 1.0
    if b >= 7 and b - a >= 2:
        return 0.0
    if a >= 6 and a == b:
        # Chaque paire de points comporte un service de chacun : formule fermée
        x = pa * (1 - pb)
        y = (1 - pa) * pb
        return x / (x + y) if x + y > 0 else 0.5
    first = _tb_server_is_first(a + b)
    a_serving = first if a_serves_first else not first
    p_point = pa if a_serving else 1 - pb
    return p_point * p_tiebreak(pa, pb, a + 1, b, a_serves_first) + \
        (1 - p_point) * p_tiebreak(pa, pb, a, b + 1, a_serves_first)


@lru_cache(maxsize=None)
def p_set(pa: float, pb: float, ga: int = 0, gb: int = 0, a_serving: bool = True) -> float:
    """Probabilité que A gagne le set depuis (ga, gb) jeux, A au service si `a_serving`."""
    if (ga >= 6 and ga - gb >= 2) or ga == 7:
        return 1.0
    if (gb >= 6 and gb - ga >= 2) or gb == 7:
        return 0.0
    if ga == 6 and gb == 6:
        return p_tiebreak(pa, pb, 0, 0, a_serving)
    pg = p_game(pa) if a_serving else 1 - p_game(pb)
    return pg * p_set(pa, pb, ga + 1, gb, not a_serving) + (1 - pg) * p_set(pa, pb, ga, gb + 1, not a_serving)


def _p_set_neutral(pa: float, pb: float) -> float:
    """Set futur : moyenne des deux ordres de service possibles."""
    return 0.5 * (p_set(pa, pb, 0, 0, True) + p_set(pa, pb, 0, 0, False))


def p_match_from_sets(pa: float, pb: float, sa: int, sb: int, best_of: int = 3) -> float:
    need = best_of // 2 + 1
    if sa >= need:
        return 1.0
    if sb >= need:
        return 0.0
    ps = _p_set_neutral(pa, pb)

    @lru_cache(maxsize=None)
    def rec(x: int, y: int) -> float:
        if x >= need:
            return 1.0
        if y >= need:
            return 0.0
        return ps * rec(x + 1, y) + (1 - ps) * rec(x, y + 1)

    return rec(sa, sb)


def live_win_probability(pa: float, pb: float, sets=(0, 0), games=(0, 0), points=(0, 0),
                         a_serving: bool = True, best_of: int = 3, tiebreak_first_server_a: bool | None = None) -> float:
    """Probabilité que A gagne le match depuis n'importe quel score.

    points : nombre de points gagnés dans le jeu en cours (0,1,2,3,4… et non 15/30/40)
    ou dans le tie-break si games == (6, 6).
    """
    pa, pb = round(float(pa), 4), round(float(pb), 4)
    sa, sb = sets
    ga, gb = games
    pta, ptb = points
    need = best_of // 2 + 1
    if sa >= need:
        return 1.0
    if sb >= need:
        return 0.0
    if ga == 6 and gb == 6:
        first_a = a_serving if tiebreak_first_server_a is None else tiebreak_first_server_a
        p_cur_set = p_tiebreak(pa, pb, pta, ptb, first_a)
    else:
        pg = p_game(pa, pta, ptb) if a_serving else 1 - p_game(pb, ptb, pta)
        p_cur_set = pg * p_set(pa, pb, ga + 1, gb, not a_serving) + \
            (1 - pg) * p_set(pa, pb, ga, gb + 1, not a_serving)
    return p_cur_set * p_match_from_sets(pa, pb, sa + 1, sb, best_of) + \
        (1 - p_cur_set) * p_match_from_sets(pa, pb, sa, sb + 1, best_of)


POINT_LABELS = {0: "0", 1: "15", 2: "30", 3: "40"}


def format_score(sets_hist: list[tuple[int, int]], games, points, tiebreak: bool) -> str:
    done = " ".join(f"{a}-{b}" for a, b in sets_hist)
    if tiebreak:
        pts = f"TB {points[0]}-{points[1]}"
    else:
        a, b = points
        if a >= 3 and b >= 3:
            pts = "40-40" if a == b else ("AV-40" if a > b else "40-AV")
        else:
            pts = f"{POINT_LABELS.get(a, a)}-{POINT_LABELS.get(b, b)}"
    return f"{done} | {games[0]}-{games[1]} | {pts}".strip(" |")


def simulate_match(pa: float, pb: float, best_of: int = 3, seed: int | None = None,
                   a_serves_first: bool = True) -> pd.DataFrame:
    """Simule un match point par point et calcule la probabilité de victoire de A après chaque point."""
    rng = np.random.default_rng(seed)
    need = best_of // 2 + 1
    sa = sb = ga = gb = pta = ptb = 0
    a_serving = a_serves_first
    tb_first_a = a_serving
    sets_hist: list[tuple[int, int]] = []
    rows = [{"point": 0, "proba_victoire_A": live_win_probability(pa, pb, (0, 0), (0, 0), (0, 0), a_serving, best_of),
             "score": "0-0", "serveur": "A" if a_serving else "B", "gagnant_point": None, "evenement": "Début"}]
    n = 0
    while sa < need and sb < need:
        tiebreak = ga == 6 and gb == 6
        if tiebreak:
            first = _tb_server_is_first(pta + ptb)
            server_a = first if tb_first_a else not first
        else:
            server_a = a_serving
        p_point = pa if server_a else 1 - pb
        a_wins = rng.random() < p_point
        pta, ptb = (pta + 1, ptb) if a_wins else (pta, ptb + 1)
        n += 1
        event = ""
        game_over = (max(pta, ptb) >= (7 if tiebreak else 4)) and abs(pta - ptb) >= 2
        if game_over:
            a_game = pta > ptb
            if not tiebreak and a_game != server_a:
                event = "Break " + ("A" if a_game else "B")
            ga, gb = (ga + 1, gb) if a_game else (ga, gb + 1)
            pta = ptb = 0
            if tiebreak:
                a_serving = not tb_first_a  # l'autre joueur sert le 1er jeu du set suivant
            else:
                a_serving = not a_serving
            set_over = (max(ga, gb) >= 6 and abs(ga - gb) >= 2) or max(ga, gb) == 7
            if set_over:
                sets_hist.append((ga, gb))
                if ga > gb:
                    sa += 1
                else:
                    sb += 1
                event = f"Set {'A' if ga > gb else 'B'} ({ga}-{gb})"
                ga = gb = 0
            if ga == 6 and gb == 6:
                tb_first_a = a_serving
        prob = live_win_probability(pa, pb, (sa, sb), (ga, gb), (pta, ptb), a_serving, best_of, tb_first_a)
        rows.append({"point": n, "proba_victoire_A": prob,
                     "score": format_score(sets_hist, (ga, gb), (pta, ptb), ga == 6 and gb == 6),
                     "serveur": "A" if server_a else "B", "gagnant_point": "A" if a_wins else "B",
                     "evenement": event})
    return pd.DataFrame(rows)
