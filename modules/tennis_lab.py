"""MODULE 5 — Tennis Data Lab & Coach Analytics."""
from __future__ import annotations

import contextlib
import io
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from modules import llm, ui
from modules.core import tennis as tn


def _data() -> dict[str, pd.DataFrame] | None:
    return st.session_state.get("tennis_data")


def _load_demo(surface: str = "Dur") -> None:
    st.session_state["tennis_data"] = {
        "matchs": tn.demo_match_stats(),
        "tracking": tn.demo_tracking(surface=surface),
        "profils": tn.demo_player_profiles(),
    }
    st.session_state["tennis_surface"] = surface


# --------------------------------------------------------------------------- #
# Données & visualisations
# --------------------------------------------------------------------------- #
def _datasets_tab() -> None:
    c1, c2 = st.columns([1, 3])
    surface = c1.selectbox("Surface du court", tn.SURFACES)
    if c1.button("🎾 Charger les données de démonstration", type="primary", width="stretch"):
        _load_demo(surface)
    data = _data()
    if not data:
        c2.info("Chargez les données de démonstration : statistiques de 300 matchs, 1 500 frappes trackées "
                "et 600 profils de joueurs (données fictives).")
        return
    c2.success("Données chargées : " + " · ".join(f"**{k}** ({len(v)} lignes)" for k, v in data.items()))
    for tab, (name, df) in zip(st.tabs([f"📄 {k}" for k in data]), data.items()):
        with tab:
            st.dataframe(df.head(300), hide_index=True, width="stretch")
            st.download_button(f"⬇️ {name}.csv", df.to_csv(index=False).encode("utf-8"), f"tennis_{name}.csv",
                               key=f"dl_t_{name}")
            if st.button(f"📤 Envoyer « {name} » vers Analytics & ML", key=f"send_{name}"):
                ui.save_dataset(f"Tennis — {name}", df)
                st.success("Disponible dans le module Analytics & ML.")

    st.divider()
    st.markdown("#### 🗺️ Court 2D à l'échelle & points d'impact")
    trk = data["tracking"]
    c1, c2, c3 = st.columns(3)
    shots = c1.multiselect("Types de coups", tn.SHOT_TYPES, default=["Service"])
    mode = c2.radio("Affichage", ["Points d'impact", "Carte de chaleur", "Les deux"], horizontal=True)
    color = c3.selectbox("Couleur des points", [None, "type_coup", "point_gagne", "cible_service", "cote", "dans_le_court"])
    sub = trk[trk["type_coup"].isin(shots)] if shots else trk
    fig = tn.court_figure(st.session_state.get("tennis_surface", "Dur"), height=520)
    if mode in ("Carte de chaleur", "Les deux"):
        tn.add_heatmap(fig, sub)
    if mode in ("Points d'impact", "Les deux"):
        tn.add_bounces(fig, sub, color)
    fig.update_layout(title=f"{len(sub)} rebonds — le joueur frappe depuis la gauche (x < 0)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### ⚡ Efficacité au service & rallyes")
    serves = trk[trk["type_coup"] == "Service"].copy()
    targets = {"Extérieur": 3.4, "Corps": 2.0, "T": 0.4}
    serves["erreur_cible_m"] = np.hypot(
        serves["x"] - (tn.SERVICE_LINE - 1.1),
        serves["y"].abs() - serves["cible_service"].map(targets)).round(2)
    c1, c2 = st.columns(2)
    fig = px.scatter(serves, x="vitesse_kmh", y="erreur_cible_m", color="dans_le_court", trendline=None,
                     color_discrete_map={True: ui.PALETTE[1], False: ui.PALETTE[3]}, opacity=0.7,
                     title="Vitesse de service vs précision (distance à la cible)",
                     labels={"erreur_cible_m": "écart à la cible (m)", "vitesse_kmh": "vitesse (km/h)"})
    c1.plotly_chart(fig, width="stretch")
    speed_bins = pd.cut(serves["vitesse_kmh"], bins=[0, 170, 180, 190, 200, 260],
                        labels=["< 170", "170-180", "180-190", "190-200", "> 200"])
    eff = serves.groupby(speed_bins, observed=True).agg(
        pct_in=("dans_le_court", "mean"), pct_gagnes=("point_gagne", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=eff["vitesse_kmh"].astype(str), y=eff["pct_in"] * 100, name="% services in", marker_color=ui.PALETTE[0])
    fig.add_bar(x=eff["vitesse_kmh"].astype(str), y=eff["pct_gagnes"] * 100, name="% points gagnés", marker_color=ui.PALETTE[1])
    fig.update_layout(title="Le compromis vitesse / régularité", barmode="group", yaxis_title="%", xaxis_title="vitesse (km/h)")
    c2.plotly_chart(fig, width="stretch")
    rally = trk[trk["type_coup"] != "Service"]
    fig = px.histogram(rally, x="longueur_rallye", color="point_gagne", barmode="group", nbins=17,
                       color_discrete_map={True: ui.PALETTE[1], False: ui.PALETTE[3]},
                       title="Longueur des rallyes selon l'issue du point")
    st.plotly_chart(fig, width="stretch")

    m = data["matchs"]
    st.markdown("#### 📊 Statistiques de matchs : qu'est-ce qui fait gagner ?")
    num = m.select_dtypes("number").drop(columns=["victoire"])
    corr = num.corrwith(m["victoire"]).sort_values()
    fig = px.bar(corr, orientation="h", title="Corrélation de chaque statistique avec la victoire",
                 color=corr.values, color_continuous_scale="RdBu", range_color=[-0.6, 0.6])
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=480, yaxis_title="", xaxis_title="corrélation")
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Cas 1 : équipementier
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Entraînement du classifieur de styles…")
def _train_style_model(profiles: pd.DataFrame):
    """Random Forest de classification des styles (mis en cache)."""
    X_tr, X_te, y_tr, y_te = train_test_split(profiles[tn.STYLE_FEATURES], profiles["style"], test_size=0.25,
                                              random_state=42, stratify=profiles["style"])
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    return model, X_te, y_te


def _case_equipment() -> None:
    data = _data()
    st.markdown("""**🎯 Brief métier** — Un équipementier veut recommander la bonne raquette et le bon cordage selon
le **style de jeu** détecté à partir de statistiques de match. Objectif : personnaliser le conseil en magasin
et les campagnes CRM (segmentation par style).""")
    profiles = data["profils"]
    feats = tn.STYLE_FEATURES
    c1, c2 = st.columns(2)
    fig = px.scatter(profiles, x="pct_montees_filet", y="longueur_moy_rallye", color="style", opacity=0.7,
                     color_discrete_sequence=ui.PALETTE, title="Les styles se séparent-ils naturellement ?")
    c1.plotly_chart(fig, width="stretch")
    fig = px.box(profiles, x="style", y="vitesse_service_kmh", color="style", color_discrete_sequence=ui.PALETTE,
                 title="Vitesse de service par style")
    fig.update_layout(showlegend=False, xaxis_title="")
    c2.plotly_chart(fig, width="stretch")

    model, X_te, y_te = _train_style_model(profiles)
    pred = model.predict(X_te)
    labels = sorted(profiles["style"].unique())
    c1, c2 = st.columns(2)
    c1.metric("Accuracy (test)", f"{accuracy_score(y_te, pred):.1%}")
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    c1.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              labels=dict(x="Prédit", y="Réel"), title="Matrice de confusion"), width="stretch")
    imp = pd.Series(model.feature_importances_, index=feats).sort_values()
    c2.plotly_chart(px.bar(imp, orientation="h", title="Variables discriminantes"), width="stretch")
    with st.expander("Rapport de classification"):
        st.code(classification_report(y_te, pred, zero_division=0))

    st.markdown("#### 🧑‍🔬 Profilez un joueur et obtenez une recommandation")
    c = st.columns(3)
    vals = {
        "pct_montees_filet": c[0].slider("% de montées au filet", 0.0, 0.6, 0.12, 0.01),
        "longueur_moy_rallye": c[1].slider("Longueur moyenne des rallyes", 1.5, 10.0, 5.0, 0.1),
        "vitesse_service_kmh": c[2].slider("Vitesse moyenne de service (km/h)", 140, 230, 188),
        "ratio_gagnants_fautes": c[0].slider("Ratio coups gagnants / fautes directes", 0.2, 2.5, 1.0, 0.05),
        "effet_moyen_rpm": c[1].slider("Effet moyen (tours/min)", 1000, 3800, 2400, 50),
        "fautes_directes_par_set": c[2].slider("Fautes directes par set", 5, 45, 22),
    }
    proba = pd.Series(model.predict_proba(pd.DataFrame([vals]))[0], index=model.classes_).sort_values(ascending=False)
    style = proba.index[0]
    reco = tn.RACKET_RECO[style]
    c1, c2 = st.columns([2, 3])
    c1.plotly_chart(px.bar(proba[::-1], orientation="h", title="Probabilité par style", range_x=[0, 1])
                    .update_layout(showlegend=False, yaxis_title="", xaxis_title=""), width="stretch")
    with c2:
        st.success(f"Style détecté : **{style}** ({proba.iloc[0]:.0%})")
        st.markdown(f"- 🎾 **Raquette** : {reco['raquette']}\n- 🧵 **Cordage** : {reco['cordage']}\n"
                    f"- 💡 **Argumentaire** : {reco['argument']}")
        st.caption("Recommandations génériques à visée pédagogique, à valider avec un expert matériel.")


# --------------------------------------------------------------------------- #
# Cas 2 : tracking & performance
# --------------------------------------------------------------------------- #
def _case_tracking() -> None:
    data = _data()
    trk = data["tracking"]
    st.markdown("""**🎯 Brief métier** — À partir des rebonds trackés (façon Hawk-Eye), analyser la **profondeur de balle**
et identifier les **zones de vulnérabilité** : où le joueur gagne-t-il / perd-il le plus de points ?""")
    rally = trk[(trk["type_coup"] != "Service") & trk["dans_le_court"]].copy()
    rally["profondeur"] = rally["x"].map(tn.depth_zone)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frappes en jeu (hors service)", len(rally))
    c2.metric("Profondeur moyenne", f"{rally['x'].mean():.1f} m du filet")
    c3.metric("% balles profondes", f"{(rally['profondeur'].str.startswith('Profonde')).mean():.0%}")
    c4.metric("% fautes (hors service)", f"{1 - trk.loc[trk['type_coup'] != 'Service', 'dans_le_court'].mean():.0%}")

    zones = tn.zone_analysis(trk)
    fig = tn.court_figure(st.session_state.get("tennis_surface", "Dur"), height=500)
    x_edges = {"Courte (carré de service)": (0, tn.SERVICE_LINE), "Moyenne": (tn.SERVICE_LINE, tn.HALF_LENGTH - 2.5),
               "Profonde (< 2,5 m de la ligne)": (tn.HALF_LENGTH - 2.5, tn.HALF_LENGTH)}
    third = tn.HALF_WIDTH_SINGLES / 3
    y_edges = {"Gauche": (third, tn.HALF_WIDTH_SINGLES), "Centre": (-third, third), "Droite": (-tn.HALF_WIDTH_SINGLES, -third)}
    vmin, vmax = zones["pct_points_gagnes"].min(), zones["pct_points_gagnes"].max()
    for _, z in zones.iterrows():
        x0, x1 = x_edges[z["profondeur"]]
        y0, y1 = y_edges[z["couloir"]]
        t = (z["pct_points_gagnes"] - vmin) / (vmax - vmin + 1e-9)
        color = f"rgba({int(220 - 180 * t)}, {int(60 + 160 * t)}, 80, 0.75)"
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line=dict(color="white", width=1))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, showarrow=False, font=dict(color="white", size=10),
                           text=f"<b>{z['pct_points_gagnes']:.0f}%</b><br>n={z['frappes']}")
    fig.update_layout(title="% de points gagnés selon la zone de rebond (vert = zone payante, rouge = zone de vulnérabilité)")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    by_depth = rally.groupby(["profondeur", "type_coup"])["point_gagne"].mean().mul(100).round(1).reset_index()
    c1.plotly_chart(px.bar(by_depth, x="profondeur", y="point_gagne", color="type_coup", barmode="group",
                           title="% points gagnés par profondeur et type de coup", color_discrete_sequence=ui.PALETTE)
                    .update_layout(yaxis_title="% points gagnés", xaxis_title=""), width="stretch")
    c2.plotly_chart(px.histogram(rally, x="x", color="type_coup", nbins=30, barmode="overlay", opacity=0.7,
                                 title="Distribution de la profondeur (distance au filet, m)",
                                 color_discrete_sequence=ui.PALETTE)
                    .add_vline(x=tn.SERVICE_LINE, line_dash="dash", annotation_text="ligne de service")
                    .add_vline(x=tn.HALF_LENGTH, line_dash="dash", annotation_text="ligne de fond"), width="stretch")
    worst = zones.sort_values("pct_points_gagnes").iloc[0]
    best = zones.sort_values("pct_points_gagnes").iloc[-1]
    st.info(f"🔎 **Lecture coach** : zone la plus payante = *{best['profondeur']} / {best['couloir']}* "
            f"({best['pct_points_gagnes']:.0f} %) ; zone de vulnérabilité = *{worst['profondeur']} / {worst['couloir']}* "
            f"({worst['pct_points_gagnes']:.0f} %). Piste d'entraînement : travailler la longueur de balle "
            "(cibles à 2,5 m de la ligne de fond) et éviter les balles courtes centrales qui offrent l'attaque à l'adversaire.")


# --------------------------------------------------------------------------- #
# Cas 3 : win probability
# --------------------------------------------------------------------------- #
def _case_winprob() -> None:
    st.markdown("""**🎯 Brief métier** — Pour une application fan / un diffuseur, afficher en direct la
**probabilité de victoire**. Modèle : chaîne de Markov hiérarchique (point → jeu → set → match) en supposant
que chaque joueur gagne ses points de service avec une probabilité constante (modèle de Barnett & Clarke).""")
    c1, c2, c3 = st.columns(3)
    pa = c1.slider("Joueur A : % points gagnés au service", 0.40, 0.85, 0.65, 0.01)
    pb = c2.slider("Joueur B : % points gagnés au service", 0.40, 0.85, 0.62, 0.01)
    best_of = c3.radio("Format", [3, 5], horizontal=True, format_func=lambda b: f"Best of {b} ({b // 2 + 1} sets gagnants)")
    pre = tn.live_win_probability(pa, pb, best_of=best_of)
    st.metric("Probabilité pré-match que A gagne", f"{pre:.1%}",
              help="Calculée exactement (récursion mémoïsée), sans simulation.")

    st.markdown("##### 📺 Score en direct")
    c = st.columns(6)
    sa = c[0].number_input("Sets A", 0, best_of // 2, 0)
    sb = c[1].number_input("Sets B", 0, best_of // 2, 0)
    ga = c[2].number_input("Jeux A", 0, 6, 3)
    gb = c[3].number_input("Jeux B", 0, 6, 4)
    labels = ["0", "15", "30", "40", "AV"]
    pta = labels.index(c[4].selectbox("Points A", labels, index=2)) if not (ga == 6 and gb == 6) else c[4].number_input("Points TB A", 0, 20, 0)
    ptb = labels.index(c[5].selectbox("Points B", labels, index=3)) if not (ga == 6 and gb == 6) else c[5].number_input("Points TB B", 0, 20, 0)
    if not (ga == 6 and gb == 6) and pta == 4 and ptb == 4:
        st.warning("Les deux joueurs ne peuvent pas avoir l'avantage.")
        return
    if not (ga == 6 and gb == 6):  # « AV » = avantage : 4 points contre 3
        if pta == 4:
            ptb = 3
        if ptb == 4:
            pta = 3
    serve = st.radio("Au service", ["A", "B"], horizontal=True)
    live = tn.live_win_probability(pa, pb, (sa, sb), (ga, gb), (pta, ptb), serve == "A", best_of)
    fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=live * 100, number={"suffix": " %"},
                                 delta={"reference": pre * 100, "suffix": " pts"},
                                 title={"text": "Probabilité de victoire de A"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": ui.PALETTE[0]},
                                        "steps": [{"range": [0, 50], "color": "#FDE2E4"}, {"range": [50, 100], "color": "#E2F0CB"}]}))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### 🎬 Simulation d'un match point par point")
    c1, c2 = st.columns([1, 3])
    seed = c1.number_input("Graine", 0, 10_000, 7)
    if c1.button("▶️ Simuler", type="primary") or "tennis_sim" not in st.session_state:
        st.session_state["tennis_sim"] = tn.simulate_match(pa, pb, best_of, seed=int(seed))
    sim = st.session_state["tennis_sim"]
    fig = px.line(sim, x="point", y="proba_victoire_A", hover_data=["score", "serveur"],
                  title="Courbe de probabilité de victoire (momentum)")
    fig.update_traces(line_color=ui.PALETTE[0])
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey")
    events = sim[sim["evenement"].str.len() > 0].iloc[1:]
    fig.add_scatter(x=events["point"], y=events["proba_victoire_A"], mode="markers", name="Breaks / sets",
                    marker=dict(size=9, color=ui.PALETTE[3]), text=events["evenement"], hovertemplate="%{text}<extra></extra>")
    fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%", title="P(victoire A)"), height=420)
    c2.plotly_chart(fig, width="stretch")
    winner = "A" if sim["proba_victoire_A"].iloc[-1] > 0.5 else "B"
    c1.metric("Vainqueur simulé", f"Joueur {winner}")
    c1.metric("Points joués", int(sim["point"].iloc[-1]))
    swings = sim["proba_victoire_A"].diff().abs()
    top = sim.loc[swings.nlargest(5).index, ["point", "score", "gagnant_point", "proba_victoire_A"]]
    with st.expander("🔥 Les 5 points les plus décisifs (plus forte variation de probabilité)"):
        st.dataframe(top.assign(proba_victoire_A=top["proba_victoire_A"].map("{:.1%}".format)), hide_index=True)
    with st.expander("🎓 Limites du modèle (à discuter en soutenance)"):
        st.markdown("""
- Hypothèse **i.i.d.** : la probabilité de gagner un point ne dépend ni du score, ni de la fatigue, ni du momentum.
- Estimation de `p` : à calibrer sur l'historique (surface, adversaire, classement Elo) et à **mettre à jour en direct** (approche bayésienne).
- Le premier serveur des sets futurs est moyenné (approximation).
- Pistes : modèles Elo par surface, gradient boosting sur données point par point, intervalles de confiance par simulation Monte-Carlo.""")


# --------------------------------------------------------------------------- #
# Coach virtuel
# --------------------------------------------------------------------------- #
CHALLENGES = {
    "1. Pandas — % de 1ère balle par surface": {
        "niveau": "Débutant",
        "enonce": "À partir de `matchs`, calculez pour chaque surface le pourcentage moyen de premières balles et "
                  "le nombre moyen d'aces, triés par aces décroissants. Stockez le résultat dans `result`.",
        "starter": "result = (\n    matchs\n    # à compléter\n)\nprint(result)",
    },
    "2. Pandas — Efficacité sur balles de break": {
        "niveau": "Intermédiaire",
        "enonce": "Créez une colonne `pct_bb_converties` (balles de break converties / obtenues, en gérant la division "
                  "par zéro), puis classez les joueurs par taux moyen de conversion. Bonus : comparez gagnants vs perdants.",
        "starter": "df = matchs.copy()\n# à compléter\nresult = df\n",
    },
    "3. Viz — Heatmap des services": {
        "niveau": "Intermédiaire",
        "enonce": "Filtrez les services dans `tracking`, puis créez une figure Plotly `fig` montrant les rebonds "
                  "(x, y) colorés selon `point_gagne`. Calculez le % de points gagnés par `cible_service`.",
        "starter": "import plotly.express as px\nserves = tracking[tracking['type_coup'] == 'Service']\n# à compléter\nfig = None\n",
    },
    "4. ML — Prédire la victoire (attention aux fuites !)": {
        "niveau": "Avancé",
        "enonce": "Entraînez un modèle prédisant `victoire` à partir des statistiques de `matchs`. Évaluez-le "
                  "correctement (split, métriques). Question piège : certaines variables ne sont connues qu'à la fin "
                  "du match — lesquelles posent un problème si l'on veut prédire AVANT le match ?",
        "starter": ("from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n"
                    "from sklearn.linear_model import LogisticRegression\n\nX = matchs.drop(columns=['victoire', 'match_id', 'joueur', 'adversaire', 'surface'])\n"
                    "y = matchs['victoire']\nX_scaled = StandardScaler().fit_transform(X)  # ⚠️ est-ce correct ?\n"
                    "X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2)\n"
                    "model = LogisticRegression().fit(X_tr, y_tr)\nprint('score', model.score(X_te, y_te))\n"),
    },
    "5. ML — Clustering de styles de jeu": {
        "niveau": "Avancé",
        "enonce": "Sans utiliser la colonne `style`, segmentez `profils` avec K-Means (choix de k justifié par la "
                  "silhouette), puis comparez les clusters obtenus aux vrais styles (tableau croisé).",
        "starter": "from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n# à compléter\n",
    },
}


def _run_user_code(code: str, data: dict) -> tuple[str, dict]:
    """Exécute le code de l'utilisateur localement (sa propre machine) et capture la sortie."""
    namespace = {"pd": pd, "np": np, "px": px, "go": go, **{k: v.copy() for k, v in data.items()}}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            exec(compile(code, "<votre_code>", "exec"), namespace)  # noqa: S102 - exécution locale volontaire
        except Exception:  # noqa: BLE001
            buffer.write("\n" + traceback.format_exc(limit=3))
    return buffer.getvalue(), namespace


def _coach() -> None:
    data = _data()
    st.markdown("Choisissez un défi, écrivez votre code, exécutez-le localement puis faites-le **évaluer par Claude** "
                "(correction, détection de fuites de données / sur-apprentissage, conseils).")
    name = st.selectbox("Défi", list(CHALLENGES))
    ch = CHALLENGES[name]
    st.markdown(f"<span class='md-badge'>{ch['niveau']}</span> {ch['enonce']}", unsafe_allow_html=True)
    st.caption("Variables disponibles : `matchs`, `tracking`, `profils` (DataFrames), `pd`, `np`, `px`, `go`.")
    code = st.text_area("Votre code Python", ch["starter"], height=280, key=f"code_{name}")
    c1, c2, c3 = st.columns(3)
    if c1.button("▶️ Exécuter localement", width="stretch"):
        out, ns = _run_user_code(code, data)
        st.session_state["coach_run"] = (out, ns.get("result"), ns.get("fig"))
    if c2.button("🧑‍🏫 Évaluer avec Claude", type="primary", width="stretch"):
        out = st.session_state.get("coach_run", ("(code non exécuté)", None, None))[0]
        schema = "\n".join(f"- {k}: {', '.join(f'{c} ({t})' for c, t in v.dtypes.astype(str).items())}" for k, v in data.items())
        prompt = (f"DÉFI : {name}\nÉNONCÉ : {ch['enonce']}\n\nSCHÉMA DES DONNÉES :\n{schema}\n\n"
                  f"CODE DE L'ÉTUDIANT :\n```python\n{code}\n```\n\nSORTIE D'EXÉCUTION :\n```\n{out[-3000:]}\n```\n\n"
                  "Évalue ce code selon ta grille.")
        llm.run_stream(llm.TENNIS_COACH_CONTEXT, prompt, "coach_review", effort="high")
    elif st.session_state.get("coach_review"):
        with st.container(border=True):
            st.markdown(st.session_state["coach_review"])
    if c3.button("💡 Indice", width="stretch"):
        llm.run_stream(llm.TENNIS_COACH_CONTEXT,
                       f"Donne un indice progressif (sans la solution complète) pour ce défi : {ch['enonce']}",
                       "coach_hint", max_tokens=4000, effort="low")
    run = st.session_state.get("coach_run")
    if run:
        out, result, fig = run
        with st.expander("🖥️ Résultat de l'exécution", expanded=True):
            st.code(out or "(aucune sortie)")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result, width="stretch")
            if isinstance(fig, go.Figure):
                st.plotly_chart(fig, width="stretch")
    st.caption("⚠️ Le code s'exécute sur votre machine avec les droits de l'application : n'exécutez que du code que vous comprenez.")

    st.divider()
    st.markdown("#### 💬 Question libre au coach")
    q = st.text_input("Ex : comment construire un modèle Elo par surface ? Quelles métriques pour un modèle de win probability ?")
    if st.button("Demander") and q:
        llm.run_stream(llm.TENNIS_COACH_CONTEXT, q, "coach_qa")
    elif st.session_state.get("coach_qa"):
        with st.container(border=True):
            st.markdown(st.session_state["coach_qa"])


def render() -> None:
    ui.hero("🎾 Tennis Data Lab & Coach Analytics",
            "Performance sportive, tracking de balle, équipementiers et modélisation prédictive — apprentissage guidé.")
    tabs = st.tabs(["🗂️ Données & Visualisations", "🏷️ Cas 1 · Équipementier", "📍 Cas 2 · Tracking & Performance",
                    "📈 Cas 3 · Win Probability", "🧑‍🏫 Coach virtuel"])
    with tabs[0]:
        _datasets_tab()
    if not _data():
        for t in tabs[1:]:
            with t:
                st.info("Chargez d'abord les données de démonstration dans l'onglet « Données & Visualisations ».")
        return
    with tabs[1]:
        _case_equipment()
    with tabs[2]:
        _case_tracking()
    with tabs[3]:
        _case_winprob()
    with tabs[4]:
        _coach()
