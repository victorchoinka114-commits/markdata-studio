"""MarkData Studio — application Streamlit principale.

Lancement :  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from modules import analytics_lab, campaign_lab, data_refinery, llm, project_hub, tennis_lab, ui

st.set_page_config(page_title="MarkData Studio", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")


def home() -> None:
    ui.hero("📊 MarkData Studio",
            "Votre studio data marketing : outil métier B2B, compagnon de Master 2 Data Marketing & IA "
            "et laboratoire de sports analytics.")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("🧪 Data Refinery", "Import Excel/CSV, audit qualité, nettoyage en un clic (manquants, doublons, "
                "téléphones, codes postaux) et export.")
    with c2:
        ui.card("📊 Analytics & Data Science", "Explorateur Plotly, segmentation RFM, K-Means, bac à sable ML et "
                "préparation de schémas en étoile pour Power BI.")
    with c3:
        ui.card("📣 Campaign Lab", "Séquences emailing A/B (texte + HTML), contenus SEO, posts LinkedIn, personas "
                "et diagnostic digital propulsés par Claude.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("🎓 M2 Project Hub", "Cadrage SMART & cahier des charges, CAC / LTV / ROI, scoring prestataires "
                "et atelier agile (user stories, DoR / DoD).")
    with c2:
        ui.card("🎾 Tennis Data Lab", "Court à l'échelle, heatmaps de rebonds, cas équipementier, tracking "
                "façon Hawk-Eye, win probability et coach de code.")
    with c3:
        ui.card("🤖 IA Claude", "Configurez votre clé API dans la barre latérale. Les modules hors IA "
                "fonctionnent entièrement en local.")
    st.divider()
    st.subheader("🚀 Parcours de démarrage conseillé")
    st.markdown("""
1. **Data Refinery** → chargez le *CRM B2B secteur public (données sales)*, observez le score qualité, puis
   normalisez emails, téléphones et codes postaux et dédoublonnez sur l'email.
2. **Envoyez** le jeu nettoyé vers **Analytics** → entraînez un modèle de **churn** et lisez l'importance des variables.
3. Chargez l'*historique de commandes* → calculez la **segmentation RFM** et exportez-la pour une campagne.
4. **Campaign Lab** → générez la séquence de réactivation du segment « En risque ».
5. **Project Hub** → cadrez le projet de dashboard et préparez le modèle en étoile pour Power BI.
6. **Tennis Data Lab** → relevez les défis de code et faites-vous corriger par le coach virtuel.
""")
    store = ui.datasets()
    if store:
        st.caption("Jeux de données en mémoire : " + ", ".join(f"**{k}** ({len(v)} lignes)" for k, v in store.items()))


def sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="md-brand">📊 MarkData Studio</div>', unsafe_allow_html=True)
        st.caption("Data Marketing · IA · Sports Analytics")
    llm.render_sidebar_settings()
    with st.sidebar:
        store = ui.datasets()
        if store:
            st.markdown("**🗂️ Jeux de données en mémoire**")
            for name, df in list(store.items()):
                c1, c2 = st.columns([5, 1])
                c1.caption(f"{name} — {len(df)} × {df.shape[1]}")
                if c2.button("✕", key=f"del_{name}", help="Retirer ce jeu de données"):
                    del store[name]
                    st.rerun()
        st.caption("Données traitées localement. Seuls les textes envoyés aux fonctions IA transitent par l'API Claude.")


def main() -> None:
    ui.inject_css()
    pages = {
        "Accueil": [st.Page(home, title="Accueil", icon="🏠", url_path="accueil", default=True)],
        "Data & Analytics": [
            st.Page(data_refinery.render, title="Data Refinery", icon="🧪", url_path="data-refinery"),
            st.Page(analytics_lab.render, title="Analytics & ML", icon="📊", url_path="analytics"),
        ],
        "Marketing & Projet": [
            st.Page(campaign_lab.render, title="Campaign Lab", icon="📣", url_path="campaign-lab"),
            st.Page(project_hub.render, title="M2 Project Hub", icon="🎓", url_path="project-hub"),
        ],
        "Sports Analytics": [
            st.Page(tennis_lab.render, title="Tennis Data Lab", icon="🎾", url_path="tennis-lab"),
        ],
    }
    nav = st.navigation(pages, position="sidebar")
    sidebar()
    nav.run()


if __name__ == "__main__":
    main()
