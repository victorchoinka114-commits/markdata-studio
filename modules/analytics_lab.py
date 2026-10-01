"""MODULE 2 — Analytics, KPIs & Data Science Sandbox."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import ui
from modules.core import analytics as an
from modules.core.cleaning import to_excel_bytes
from modules.core.samples import demo_crm_b2b, demo_transactions

DEMOS = {
    "CRM B2B (propre)": lambda: demo_crm_b2b(dirty=False),
    "Historique de commandes": demo_transactions,
}


# --------------------------------------------------------------------------- #
# Explorateur
# --------------------------------------------------------------------------- #
def _explorer(df: pd.DataFrame) -> None:
    num, cat = ui.numeric_columns(df), ui.categorical_columns(df)
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{len(df):,}".replace(",", " "))
    c2.metric("Variables numériques", len(num))
    c3.metric("Variables catégorielles", len(cat))
    c4.metric("Complétude", f"{(1 - df.isna().mean().mean()) * 100:.1f} %")

    chart = st.radio("Type de graphique", ["Histogramme", "Nuage de points", "Boîte à moustaches",
                                           "Série temporelle", "Barres (agrégat)", "Matrice de corrélation"],
                     horizontal=True)
    color = None
    if chart == "Histogramme" and num:
        a, b, c = st.columns(3)
        x = a.selectbox("Variable", num)
        color = b.selectbox("Couleur", [None] + cat)
        bins = c.slider("Nombre de classes", 5, 100, 30)
        fig = px.histogram(df, x=x, color=color, nbins=bins, marginal="box", barmode="overlay",
                           color_discrete_sequence=ui.PALETTE, opacity=0.8)
    elif chart == "Nuage de points" and len(num) >= 2:
        a, b, c, d = st.columns(4)
        x = a.selectbox("Axe X", num)
        y = b.selectbox("Axe Y", num, index=1)
        color = c.selectbox("Couleur", [None] + cat + num)
        size = d.selectbox("Taille", [None] + num)
        trend = "ols" if st.checkbox("Droite de tendance (OLS)") and color is None else None
        plot_df = df.dropna(subset=[size]) if size else df
        fig = px.scatter(plot_df, x=x, y=y, color=color, size=size, trendline=trend if _has_statsmodels() else None,
                         color_discrete_sequence=ui.PALETTE, opacity=0.75, hover_data=df.columns[:4])
    elif chart == "Boîte à moustaches" and num:
        a, b = st.columns(2)
        y = a.selectbox("Variable", num)
        x = b.selectbox("Grouper par", [None] + cat)
        fig = px.box(df, x=x, y=y, color=x, points="outliers", color_discrete_sequence=ui.PALETTE)
    elif chart == "Série temporelle":
        candidates = date_cols or [c for c in df.columns if "date" in str(c).lower()]
        if not candidates or not num:
            st.info("Aucune colonne date détectée.")
            return
        a, b, c, d = st.columns(4)
        dcol = a.selectbox("Date", candidates)
        y = b.selectbox("Mesure", num)
        agg = c.selectbox("Agrégation", ["sum", "mean", "count"])
        freq = d.selectbox("Période", ["D", "W", "MS", "QS", "YS"], index=2,
                           format_func={"D": "Jour", "W": "Semaine", "MS": "Mois", "QS": "Trimestre", "YS": "Année"}.get)
        ts = df.assign(_d=pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)).dropna(subset=["_d"])
        ts = ts.set_index("_d")[y].resample(freq).agg(agg).reset_index()
        fig = px.line(ts, x="_d", y=y, markers=True, color_discrete_sequence=ui.PALETTE,
                      labels={"_d": dcol})
        if len(ts) > 3:
            ts["moyenne mobile (3)"] = ts[y].rolling(3).mean()
            fig.add_scatter(x=ts["_d"], y=ts["moyenne mobile (3)"], name="Moyenne mobile (3)",
                            line=dict(dash="dash", color=ui.PALETTE[2]))
    elif chart == "Barres (agrégat)" and cat:
        a, b, c = st.columns(3)
        x = a.selectbox("Dimension", cat)
        y = b.selectbox("Mesure", ["(nombre de lignes)"] + num)
        agg = c.selectbox("Agrégation", ["sum", "mean", "median"])
        data = (df.groupby(x).size().rename("nb").reset_index() if y == "(nombre de lignes)"
                else df.groupby(x)[y].agg(agg).reset_index())
        ycol = "nb" if y == "(nombre de lignes)" else y
        fig = px.bar(data.sort_values(ycol, ascending=False), x=x, y=ycol, text_auto=".3s",
                     color_discrete_sequence=ui.PALETTE)
    elif chart == "Matrice de corrélation" and len(num) >= 2:
        cols = st.multiselect("Variables", num, default=num[:12])
        method = st.radio("Méthode", ["pearson", "spearman"], horizontal=True)
        corr = df[cols].corr(method=method)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=max(420, 38 * len(cols)))
        st.plotly_chart(fig, width="stretch")
        pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
                 .rename("corrélation").reset_index().assign(abs=lambda d: d["corrélation"].abs())
                 .sort_values("abs", ascending=False).drop(columns="abs").head(8))
        st.caption("Top corrélations (attention : corrélation ≠ causalité)")
        st.dataframe(pairs.round(3), hide_index=True)
        return
    else:
        st.info("Ce graphique nécessite davantage de colonnes adaptées dans le jeu de données.")
        return
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=480)
    st.plotly_chart(fig, width="stretch")


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# RFM
# --------------------------------------------------------------------------- #
def _rfm(df: pd.DataFrame) -> None:
    st.markdown("Le scoring **RFM** note chaque client de 1 à 5 sur la **R**écence, la **F**réquence et le "
                "**M**ontant de ses achats (quintiles), puis l'affecte à un segment actionnable.")
    cols = df.columns.tolist()

    def guess(*keys):
        return next((i for i, c in enumerate(cols) if any(k in str(c).lower() for k in keys)), 0)

    a, b, c = st.columns(3)
    cust = a.selectbox("Identifiant client", cols, index=guess("client", "id", "siren"))
    date = b.selectbox("Date de commande", cols, index=guess("date"))
    amount = c.selectbox("Montant", cols, index=guess("montant", "amount", "ca", "prix"))
    if not st.button("Calculer la segmentation RFM", type="primary"):
        if "rfm_result" not in st.session_state:
            return
    else:
        try:
            st.session_state["rfm_result"] = an.compute_rfm(df, cust, date, amount)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Calcul impossible : {exc}")
            return
    rfm = st.session_state["rfm_result"]
    summary = an.rfm_summary(rfm)
    kcols = st.columns(len(summary))
    for col, (_, row) in zip(kcols, summary.iterrows()):
        col.metric(row["segment"], f"{int(row['clients'])} clients", f"{row['% CA']} % du CA", delta_color="off")

    c1, c2 = st.columns(2)
    fig = px.treemap(summary, path=["segment"], values="clients", color="segment",
                     color_discrete_map=an.RFM_COLORS, title="Répartition des clients")
    c1.plotly_chart(fig, width="stretch")
    fig = px.scatter(rfm, x="recence_jours", y="frequence", size="montant", color="segment",
                     color_discrete_map=an.RFM_COLORS, log_y=True, title="Récence vs Fréquence (taille = montant)",
                     hover_data=[rfm.columns[0], "score_RFM"])
    c2.plotly_chart(fig, width="stretch")
    heat = rfm.pivot_table(index="F", columns="R", values="montant", aggfunc="mean").sort_index(ascending=False)
    fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                    title="Montant moyen par score R (colonnes) et F (lignes)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(summary, hide_index=True, width="stretch")
    with st.expander("Détail par client"):
        st.dataframe(rfm, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Export RFM (.xlsx)", to_excel_bytes({"rfm": rfm, "synthese": summary}),
                       "segmentation_rfm.xlsx", width="stretch")
    if c2.button("📤 Enregistrer comme jeu de données", width="stretch"):
        ui.save_dataset("Segmentation RFM", rfm, make_active=False)
        st.success("Segmentation RFM disponible dans les autres onglets.")


# --------------------------------------------------------------------------- #
# K-Means
# --------------------------------------------------------------------------- #
def _kmeans(df: pd.DataFrame) -> None:
    num = ui.numeric_columns(df)
    if len(num) < 2:
        st.info("Il faut au moins deux variables numériques.")
        return
    feats = st.multiselect("Variables de segmentation", num, default=num[: min(4, len(num))])
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Nombre de clusters (k)", 2, 10, 4)
    dims = c2.radio("Projection", [2, 3], horizontal=True, format_func=lambda d: f"{d}D (ACP)")
    scale = c3.toggle("Standardisation (StandardScaler)", value=True,
                      help="Indispensable si les variables ont des unités différentes (€ vs %).")
    if len(feats) < 2:
        return
    with st.expander("📐 Aide au choix de k : méthode du coude & silhouette"):
        if st.button("Lancer l'analyse k = 2…10"):
            el = an.elbow_analysis(df, feats, scale=scale)
            fig = go.Figure()
            fig.add_scatter(x=el["k"], y=el["inertie"], name="Inertie", mode="lines+markers")
            fig.add_scatter(x=el["k"], y=el["silhouette"], name="Silhouette", mode="lines+markers", yaxis="y2")
            fig.update_layout(yaxis=dict(title="Inertie"), yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
                              height=360, margin=dict(t=20))
            st.plotly_chart(fig, width="stretch")
            best = int(el.loc[el["silhouette"].idxmax(), "k"])
            st.info(f"Silhouette maximale pour **k = {best}** (à confronter au sens métier des segments).")
    try:
        res = an.run_kmeans(df, feats, k, scale=scale, n_components=dims)
    except ValueError as exc:
        st.error(str(exc))
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Lignes segmentées", len(res.labels))
    m2.metric("Score silhouette", f"{res.silhouette:.3f}" if res.silhouette is not None else "—",
              help="De -1 à 1 : > 0,5 bonne séparation, < 0,25 segments peu distincts.")
    m3.metric("Variance expliquée (ACP)", f"{sum(res.explained_variance) * 100:.1f} %" if res.explained_variance else "—")
    proj = res.projection
    if dims == 3 and "Axe 3" in proj:
        fig = px.scatter_3d(proj, x="Axe 1", y="Axe 2", z="Axe 3", color="cluster", opacity=0.75,
                            color_discrete_sequence=ui.PALETTE)
        fig.update_traces(marker_size=4)
    else:
        fig = px.scatter(proj, x="Axe 1", y="Axe 2", color="cluster", opacity=0.75, color_discrete_sequence=ui.PALETTE)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Profil moyen des clusters**")
    st.dataframe(res.profile, width="stretch")
    norm = res.profile[feats]
    norm = (norm - norm.min()) / (norm.max() - norm.min()).replace(0, 1)
    fig = go.Figure()
    for i, (cluster, row) in enumerate(norm.iterrows()):
        fig.add_trace(go.Scatterpolar(r=list(row.values) + [row.values[0]], theta=feats + [feats[0]], fill="toself",
                                      name=cluster, line_color=ui.PALETTE[i % len(ui.PALETTE)], opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=450, title="Radar des profils (normalisé)")
    st.plotly_chart(fig, width="stretch")
    labelled = df.loc[res.index].assign(cluster=res.projection["cluster"].values)
    st.download_button("⬇️ Export des clusters (.xlsx)", to_excel_bytes({"clusters": labelled, "profils": res.profile.reset_index()}),
                       "clusters_kmeans.xlsx")


# --------------------------------------------------------------------------- #
# ML sandbox
# --------------------------------------------------------------------------- #
def _ml(df: pd.DataFrame) -> None:
    cols = df.columns.tolist()
    default_target = cols.index("churn") if "churn" in cols else len(cols) - 1
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Variable cible", cols, index=default_target)
    y = df[target].dropna()
    task_auto = an.detect_task(y) if len(y) else "classification"
    task = c2.radio("Type de problème", ["classification", "regression"],
                    index=0 if task_auto == "classification" else 1, horizontal=True,
                    format_func={"classification": "Classification", "regression": "Régression"}.get)
    algo = c3.selectbox("Algorithme", an.ALGOS[task])
    candidates = [c for c in cols if c != target
                  and (pd.api.types.is_float_dtype(df[c]) or df[c].nunique() < 0.95 * len(df))]
    excluded = {"id_client", "siren", "email", "contact", "telephone", "organisation", "date_dernier_contact"}
    default_feats = [c for c in candidates if c not in excluded][:10]
    feats = st.multiselect("Variables explicatives", [c for c in cols if c != target], default=default_feats)
    c1, c2 = st.columns(2)
    test_size = c1.slider("Part du jeu de test", 0.1, 0.5, 0.25, 0.05)
    seed = c2.number_input("Graine aléatoire", 0, 9999, 42)
    if not feats:
        return
    if st.button("🚀 Entraîner le modèle", type="primary"):
        try:
            with st.spinner("Entraînement et validation croisée…"):
                st.session_state["ml_result"] = an.train_model(df, target, feats, algo, task, test_size, int(seed))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Entraînement impossible : {exc}")
            return
    res: an.ModelResult | None = st.session_state.get("ml_result")
    if res is None:
        return
    for w in res.warnings:
        st.warning("⚠️ " + w)
    mcols = st.columns(len(res.metrics))
    for col, (name, val) in zip(mcols, res.metrics.items()):
        col.metric(name, f"{val:,.3f}".replace(",", " "))
    if res.cv_scores is not None:
        st.caption(f"Validation croisée 5 plis (train) : {res.cv_scores.mean():.3f} ± {res.cv_scores.std():.3f}")

    c1, c2 = st.columns(2)
    if res.task == "classification" and res.confusion is not None:
        cm = res.confusion
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Prédit", y="Réel", color="Effectif"), title="Matrice de confusion")
        c1.plotly_chart(fig, width="stretch")
    else:
        comp = pd.DataFrame({"Réel": res.y_test.values, "Prédit": res.y_pred})
        fig = px.scatter(comp, x="Réel", y="Prédit", opacity=0.6, title="Réel vs prédit",
                         color_discrete_sequence=ui.PALETTE)
        lo, hi = comp.min().min(), comp.max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
        c1.plotly_chart(fig, width="stretch")
    imp = res.importance.head(15)
    fig = px.bar(imp, x="importance", y="variable", orientation="h", title="Importance des variables",
                 color="importance", color_continuous_scale="Blues")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    c2.plotly_chart(fig, width="stretch")
    with st.expander("🎓 Comment lire ces résultats ?"):
        if res.task == "classification":
            st.markdown("""
- **Accuracy** : part de prédictions correctes — trompeuse si les classes sont déséquilibrées (comparez à la *baseline*).
- **Précision** : parmi les clients prédits « à risque », combien le sont réellement (coût des fausses alertes).
- **Rappel** : parmi les vrais churners, combien sont détectés (coût des clients ratés).
- **F1-Score** : moyenne harmonique précision/rappel. **ROC AUC** : capacité à classer (0,5 = hasard).
- **Importance** : contribution relative des variables (RF : réduction d'impureté ; logistique : |coefficients| standardisés).""")
        else:
            st.markdown("""
- **R²** : part de la variance expliquée (1 = parfait, 0 = pas mieux que la moyenne).
- **RMSE** : erreur type, dans l'unité de la cible, pénalise les grosses erreurs. **MAE** : erreur absolue moyenne.
- Le prétraitement (imputation, standardisation, one-hot) est intégré au pipeline et appris **uniquement sur le train** : pas de fuite de données.""")


# --------------------------------------------------------------------------- #
# Power BI
# --------------------------------------------------------------------------- #
def _powerbi(df: pd.DataFrame) -> None:
    st.markdown("""
Un **schéma en étoile** sépare une **table de faits** (événements mesurables : commandes, interactions, tickets)
de **tables de dimensions** (axes d'analyse : client, offre, géographie, date) reliées par des **clés de substitution**
(entiers). Bénéfices dans Power BI : modèle plus léger, DAX plus simple, filtres fiables (relations 1-*).""")
    prof = an.profile_columns(df)
    st.markdown("**1. Profil des colonnes et rôle suggéré**")
    st.dataframe(prof, hide_index=True, width="stretch")

    measures_default = prof.loc[prof["rôle suggéré"].str.startswith("Mesure"), "colonne"].tolist()
    attrs_default = prof.loc[prof["rôle suggéré"] == "Attribut de dimension", "colonne"].tolist()
    dates = prof.loc[prof["rôle suggéré"].str.startswith("Date"), "colonne"].tolist()

    st.markdown("**2. Définissez le modèle**")
    c1, c2, c3 = st.columns(3)
    measures = c1.multiselect("Mesures (table de faits)", df.columns.tolist(), default=measures_default)
    attrs = c2.multiselect("Attributs de dimensions", df.columns.tolist(), default=attrs_default)
    date_col = c3.selectbox("Colonne date (→ DimDate)", [None] + df.columns.tolist(),
                            index=(df.columns.tolist().index(dates[0]) + 1) if dates else 0)
    if not attrs:
        return
    dims = an.suggest_dimensions(df, attrs)
    st.markdown("**3. Dimensions proposées** (regroupement par dépendances fonctionnelles, ex. ville → code postal)")
    edited = {}
    cols = st.columns(min(3, len(dims)) or 1)
    for i, (name, members) in enumerate(dims.items()):
        with cols[i % len(cols)]:
            new_name = st.text_input("Nom de la dimension", name, key=f"dimname_{name}")
            edited[new_name] = st.multiselect("Attributs", attrs, default=members, key=f"dim_{name}")
    tables = an.build_star_schema(df, edited, measures, date_col)

    st.markdown("**4. Schéma résultant**")
    _star_diagram(tables)
    tab_names = list(tables)
    for tab, name in zip(st.tabs(tab_names), tab_names):
        with tab:
            t = tables[name]
            st.caption(f"{len(t):,} lignes × {t.shape[1]} colonnes".replace(",", " "))
            st.dataframe(t.head(100), hide_index=True, width="stretch")
    st.markdown("**5. Mesures DAX de démarrage**")
    st.code(an.dax_measures(measures), language="sql")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Modèle complet (.xlsx, 1 onglet par table)", to_excel_bytes(tables),
                       "modele_etoile_powerbi.xlsx", width="stretch")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, t in tables.items():
            z.writestr(f"{name}.csv", t.to_csv(index=False, sep=";").encode("utf-8-sig"))
    c2.download_button("⬇️ Tables en CSV (.zip)", buf.getvalue(), "modele_etoile_csv.zip", width="stretch")
    with st.expander("✅ Bonnes pratiques Power BI"):
        st.markdown("""
- Relations **1 → plusieurs** de chaque dimension vers la table de faits, filtrage **unidirectionnel**.
- Marquer **DimDate** comme *table de dates* ; ne jamais filtrer sur la date de la table de faits.
- Masquer les clés techniques (`*_key`) en vue rapport ; formater les mesures (€, %).
- Préférer des **mesures DAX** explicites aux colonnes calculées ; éviter les relations bidirectionnelles.
- Les attributs à forte cardinalité (emails, noms) restent dans la dimension client, jamais dans les faits.""")


def _star_diagram(tables: dict[str, pd.DataFrame]) -> None:
    """Diagramme simple du schéma en étoile."""
    dims = [n for n in tables if n != "FaitPrincipal"]
    n = max(len(dims), 1)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    fig = go.Figure()
    for name, ang in zip(dims, angles):
        x, y = np.cos(ang), np.sin(ang)
        fig.add_scatter(x=[0, x], y=[0, y], mode="lines", line=dict(color="#A0AEC0", width=2),
                        hoverinfo="skip", showlegend=False)
        fig.add_annotation(x=x * 0.5, y=y * 0.5, text="1 → *", showarrow=False, font=dict(size=10, color="#718096"))
        fig.add_scatter(x=[x], y=[y], mode="markers+text", text=[f"<b>{name}</b><br>{tables[name].shape[1]} col."],
                        textposition="middle center", marker=dict(size=95, color="#EEF2FF", line=dict(color="#2E5BFF", width=2)),
                        hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=[0], y=[0], mode="markers+text", text=["<b>FaitPrincipal</b>"], textposition="middle center",
                    marker=dict(size=120, color="#2E5BFF"), textfont=dict(color="white"), hoverinfo="skip", showlegend=False)
    fig.update_layout(height=420, xaxis=dict(visible=False, range=[-1.5, 1.5]),
                      yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x"),
                      margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


def render() -> None:
    ui.hero("📊 Analytics, KPIs & Data Science", "Explorez, segmentez (RFM, K-Means), modélisez et préparez vos données pour Power BI.")
    name, df = ui.require_dataset("analytics_ds", DEMOS)
    if df is None:
        return
    tabs = st.tabs(["🔎 Explorateur", "🏆 Segmentation RFM", "🧩 Clustering K-Means", "🤖 ML Sandbox", "⭐ Préparation Power BI"])
    with tabs[0]:
        _explorer(df)
    with tabs[1]:
        _rfm(df)
    with tabs[2]:
        _kmeans(df)
    with tabs[3]:
        _ml(df)
    with tabs[4]:
        _powerbi(df)
