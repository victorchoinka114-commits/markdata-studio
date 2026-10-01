"""MODULE 4 — M2 Project Hub : cadrage, budget, scoring prestataires, agilité."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import llm, ui
from modules.core import marketing as mk
from modules.core.cleaning import to_excel_bytes


# --------------------------------------------------------------------------- #
# Cadrage & cahier des charges
# --------------------------------------------------------------------------- #
def build_cdc_markdown(f: dict) -> str:
    """Cahier des charges structuré (Markdown) à partir du formulaire de cadrage."""
    def bullets(text: str) -> str:
        items = [line.strip(" -•") for line in (text or "").splitlines() if line.strip()]
        return "\n".join(f"- {i}" for i in items) if items else "- [À COMPLÉTER]"

    risks = f.get("risques")
    risk_table = "| Risque | Probabilité | Impact | Mitigation |\n|---|---|---|---|\n"
    if isinstance(risks, pd.DataFrame) and not risks.empty:
        for _, r in risks.dropna(subset=["risque"]).iterrows():
            risk_table += f"| {r['risque']} | {r['probabilité']} | {r['impact']} | {r['mitigation'] or ''} |\n"
    else:
        risk_table += "| [À COMPLÉTER] | | | |\n"
    return f"""# Cahier des charges — {f['titre']}

*Version 1.0 — {date.today().strftime('%d/%m/%Y')} — Rédigé par : {f['auteur'] or '[À COMPLÉTER]'} — Commanditaire : {f['sponsor'] or '[À COMPLÉTER]'}*

## 1. Contexte et enjeux
{f['contexte'] or '[À COMPLÉTER]'}

## 2. Objectifs SMART
| Critère | Formulation |
|---|---|
| **S**pécifique | {f['smart_s']} |
| **M**esurable | {f['smart_m']} |
| **A**tteignable | {f['smart_a']} |
| **R**éaliste / pertinent | {f['smart_r']} |
| **T**emporellement défini | {f['smart_t']} |

## 3. Cibles et parties prenantes
{bullets(f['parties_prenantes'])}

## 4. Périmètre fonctionnel
### 4.1 Dans le périmètre
{bullets(f['perimetre_in'])}

### 4.2 Hors périmètre
{bullets(f['perimetre_out'])}

### 4.3 Exigences fonctionnelles (MoSCoW)
{bullets(f['exigences'])}

## 5. Architecture technique et données
- **Sources de données** : {f['sources'] or '[À COMPLÉTER]'}
- **Outils / stack** : {f['stack'] or '[À COMPLÉTER]'}
- **Hébergement & sécurité** : {f['securite'] or '[À COMPLÉTER]'}
- **Conformité RGPD** : base légale, registre des traitements, durée de conservation, analyse d'impact (AIPD) si nécessaire.

## 6. Contraintes
- **Budget** : {ui.fmt_eur(f['budget'])}
- **Échéance** : {f['deadline'].strftime('%d/%m/%Y')}
- **Autres** : {f['contraintes'] or '—'}

## 7. Analyse des risques
{risk_table}
## 8. Livrables et jalons
{bullets(f['livrables'])}

## 9. Gouvernance et méthodologie
- Méthode : {f['methode']}
- Instances : comité de pilotage mensuel, points d'équipe hebdomadaires, revue de sprint (si agile).
- Indicateurs de pilotage : avancement, consommation budgétaire, risques ouverts, satisfaction des parties prenantes.

## 10. Critères de recette
{bullets(f['recette'])}
"""


def _cadrage() -> None:
    st.markdown("Remplissez le formulaire de cadrage : un **cahier des charges** structuré est généré instantanément, "
                "puis peut être **enrichi par Claude**.")
    with st.form("cadrage"):
        c1, c2, c3 = st.columns(3)
        titre = c1.text_input("Titre du projet", "Mise en place d'un dashboard de pilotage marketing")
        auteur = c2.text_input("Auteur")
        sponsor = c3.text_input("Commanditaire / sponsor", "Direction marketing & communication")
        contexte = st.text_area("Contexte et problématique",
                                "Les données marketing (CRM, emailing, site web, salons) sont dispersées, "
                                "ce qui empêche de mesurer le ROI des actions et la contribution au pipe commercial.")
        st.markdown("**Objectif SMART**")
        s1, s2 = st.columns(2)
        smart_s = s1.text_input("Spécifique", "Centraliser les KPIs marketing dans un tableau de bord Power BI unique")
        smart_m = s2.text_input("Mesurable", "100 % des campagnes suivies ; 10 KPIs automatisés")
        smart_a = s1.text_input("Atteignable", "Sources accessibles via export / API ; compétences internes Power BI")
        smart_r = s2.text_input("Réaliste / pertinent", "Répond au besoin de pilotage du ROI exprimé par la direction")
        smart_t = s1.text_input("Temporel", "Mise en production d'ici la fin du trimestre")
        c1, c2 = st.columns(2)
        parties = c1.text_area("Parties prenantes (une par ligne)", "Direction marketing\nÉquipe commerciale\nDSI\nDPO")
        exigences = c2.text_area("Exigences fonctionnelles (MoSCoW)",
                                 "MUST : suivi des leads par source\nMUST : coût par lead et par campagne\n"
                                 "SHOULD : attribution multi-touch\nCOULD : prévision du pipe\nWON'T : scoring IA en v1")
        perimetre_in = c1.text_area("Dans le périmètre", "Campagnes emailing\nSalons et webinars\nSite web & SEO")
        perimetre_out = c2.text_area("Hors périmètre", "Données financières détaillées\nRefonte du CRM")
        c1, c2, c3 = st.columns(3)
        sources = c1.text_input("Sources de données", "CRM, outil d'emailing, Google Analytics 4, fichiers Excel salons")
        stack = c2.text_input("Stack / outils", "Power BI, Power Query, SharePoint")
        securite = c3.text_input("Hébergement & sécurité", "Tenant Microsoft 365, accès par rôles")
        c1, c2, c3 = st.columns(3)
        budget = c1.number_input("Budget (€)", 0, 10_000_000, 15000, 500)
        deadline = c2.date_input("Échéance", date(date.today().year, 12, 15) if date.today().month < 12 else date.today())
        methode = c3.selectbox("Méthodologie", ["Agile Scrum", "Kanban", "Cycle en V", "Hybride (cadrage V + réalisation agile)"])
        contraintes = st.text_input("Autres contraintes", "Disponibilité limitée de la DSI ; validation DPO requise")
        st.markdown("**Risques**")
        risques = st.data_editor(pd.DataFrame({
            "risque": ["Qualité des données CRM insuffisante", "Adoption faible par les commerciaux"],
            "probabilité": ["Élevée", "Moyenne"], "impact": ["Élevé", "Élevé"],
            "mitigation": ["Audit qualité + règles de saisie", "Ateliers de co-construction, formation"],
        }), num_rows="dynamic", hide_index=True, width="stretch", column_config={
            "probabilité": st.column_config.SelectboxColumn(options=["Faible", "Moyenne", "Élevée"]),
            "impact": st.column_config.SelectboxColumn(options=["Faible", "Moyen", "Élevé"]),
        })
        c1, c2 = st.columns(2)
        livrables = c1.text_area("Livrables & jalons", "Note de cadrage (S2)\nModèle de données (S4)\n"
                                                         "Dashboard v1 (S8)\nFormation & documentation (S10)")
        recette = c2.text_area("Critères de recette", "KPIs cohérents avec les sources à ± 1 %\n"
                                                        "Actualisation quotidienne automatique\nTemps de chargement < 5 s")
        submitted = st.form_submit_button("📄 Générer le cahier des charges", type="primary")
    if submitted:
        st.session_state["cdc_md"] = build_cdc_markdown(dict(
            titre=titre, auteur=auteur, sponsor=sponsor, contexte=contexte, smart_s=smart_s, smart_m=smart_m,
            smart_a=smart_a, smart_r=smart_r, smart_t=smart_t, parties_prenantes=parties, perimetre_in=perimetre_in,
            perimetre_out=perimetre_out, exigences=exigences, sources=sources, stack=stack, securite=securite,
            budget=budget, deadline=deadline, methode=methode, contraintes=contraintes, risques=risques,
            livrables=livrables, recette=recette))
        st.session_state.pop("cdc_ai", None)
    cdc = st.session_state.get("cdc_md")
    if not cdc:
        return
    t1, t2 = st.tabs(["📄 Cahier des charges", "🤖 Version enrichie par Claude"])
    with t1:
        with st.container(border=True):
            st.markdown(cdc)
        ui.download_markdown("⬇️ Télécharger le CdC (.md)", cdc, "cahier_des_charges.md", key="dl_cdc")
    with t2:
        if st.button("✨ Enrichir et challenger le CdC"):
            prompt = ("Voici un cahier des charges rédigé à partir d'un formulaire de cadrage. Produis une version "
                      "complète et professionnelle (Markdown) : reformule, complète chaque section, ajoute les "
                      "exigences non fonctionnelles, un RACI, un planning prévisionnel (tableau), une estimation "
                      "budgétaire ventilée et une section « Points de vigilance ». Ensuite, liste en fin de document "
                      "les incohérences ou manques détectés (objectifs non mesurables, risques oubliés…).\n\n" + cdc)
            llm.run_stream(llm.M2_CONTEXT, prompt, "cdc_ai")
        elif st.session_state.get("cdc_ai"):
            with st.container(border=True):
                st.markdown(st.session_state["cdc_ai"])
        if st.session_state.get("cdc_ai"):
            ui.download_markdown("⬇️ CdC enrichi (.md)", st.session_state["cdc_ai"], "cahier_des_charges_enrichi.md", key="dl_cdc_ai")


# --------------------------------------------------------------------------- #
# Calculateurs
# --------------------------------------------------------------------------- #
def _calculators() -> None:
    st.markdown("#### 💶 Unit economics : CAC, LTV, ratio LTV/CAC")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Acquisition**")
        mkt = st.number_input("Dépenses marketing (€ / période)", 0.0, value=60000.0, step=1000.0)
        sales = st.number_input("Dépenses commerciales (€ / période)", 0.0, value=90000.0, step=1000.0)
        new = st.number_input("Nouveaux clients sur la période", 0, value=25)
    with c2:
        st.markdown("**Valeur client**")
        arpa = st.number_input("Revenu mensuel moyen par client (ARPA, €)", 0.0, value=1200.0, step=50.0)
        margin = st.slider("Marge brute (%)", 0, 100, 75)
        churn = st.slider("Churn mensuel (%)", 0.1, 10.0, 1.0, 0.1)
    cac = mk.cac(mkt, sales, new)
    ltv = mk.ltv(arpa, margin, churn)
    ratio = mk.safe_div(ltv, cac) if (ltv and cac) else None
    payback = mk.payback_months(cac, arpa, margin)
    with c3:
        st.markdown("**Résultats**")
        st.metric("CAC", ui.fmt_eur(cac))
        st.metric("LTV", ui.fmt_eur(ltv))
        st.metric("Ratio LTV / CAC", f"{ratio:.1f} : 1" if ratio else "—")
        st.metric("Payback du CAC", ui.fmt_num(payback, 1, " mois"))
    level, msg = mk.ltv_cac_diagnostic(ratio)
    getattr(st, level)(msg)
    with st.expander("📐 Formules"):
        st.latex(r"CAC = \frac{\text{Dépenses marketing} + \text{Dépenses commerciales}}{\text{Nouveaux clients}}")
        st.latex(r"LTV = \frac{ARPA \times \text{Marge brute}}{\text{Churn mensuel}} \qquad "
                 r"Payback = \frac{CAC}{ARPA \times \text{Marge brute}}")
        st.caption(f"Durée de vie moyenne implicite : {ui.fmt_num(mk.lifetime_months(churn), 0, ' mois')}.")

    st.divider()
    st.markdown("#### 📈 ROI de campagne & simulateur d'entonnoir")
    c1, c2 = st.columns([1, 2])
    with c1:
        budget = st.number_input("Budget campagne (€)", 0.0, value=20000.0, step=500.0)
        cpm = st.number_input("CPM (€ / 1000 impressions)", 0.1, value=35.0)
        ctr = st.slider("CTR (%)", 0.1, 5.0, 0.8, 0.1)
        cvr = st.slider("Conversion clic → lead (%)", 0.5, 30.0, 8.0, 0.5)
        l2o = st.slider("Lead → opportunité (%)", 1, 80, 25)
        win = st.slider("Taux de signature (%)", 1, 80, 20)
        deal = st.number_input("Valeur moyenne d'un contrat (€, 1ère année)", 0.0, value=18000.0, step=500.0)
    funnel = mk.campaign_funnel(budget, cpm, ctr, cvr, l2o, win, deal)
    revenue = float(funnel["revenu_estime"].iloc[-1])
    roi = mk.roi(revenue, budget)
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("CA généré estimé", ui.fmt_eur(revenue))
        m2.metric("ROI", f"{roi * 100:,.0f} %".replace(",", " ") if roi is not None else "—")
        m3.metric("Coût par lead", ui.fmt_eur(funnel["coût unitaire (€)"].iloc[2], 0))
        fig = go.Figure(go.Funnel(y=funnel["étape"], x=funnel["volume"], textinfo="value+percent initial",
                                  marker_color=ui.PALETTE[:5]))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    st.caption("Hypothèses indicatives à remplacer par vos historiques de campagne. "
               "Le ROI sur la LTV (et non la 1ère année) est souvent plus pertinent en SaaS.")


# --------------------------------------------------------------------------- #
# Scoring prestataires
# --------------------------------------------------------------------------- #
def _vendor_scoring() -> None:
    st.markdown("Comparez des prestataires ou logiciels avec une **matrice multicritère pondérée** "
                "(notes de 0 à 5, 5 = meilleur ; pour le coût, 5 = le moins cher).")
    st.markdown("**1. Pondérations des critères (%)**")
    crit_df = st.data_editor(pd.DataFrame({"critère": list(mk.DEFAULT_CRITERIA), "poids": list(mk.DEFAULT_CRITERIA.values())}),
                             num_rows="dynamic", hide_index=True, width="stretch", key="crit_editor")
    crit_df = crit_df.dropna(subset=["critère"])
    total = crit_df["poids"].fillna(0).sum()
    (st.success if abs(total - 100) < 1e-9 else st.warning)(
        f"Somme des pondérations : {total:.0f} % " + ("" if abs(total - 100) < 1e-9 else "(normalisée automatiquement à 100 %)"))
    criteria = crit_df["critère"].tolist()
    st.markdown("**2. Notation des prestataires**")
    default = pd.DataFrame(
        [[3, 4, 4, 3, 3], [4, 3, 3, 4, 4], [2, 5, 5, 4, 2]], index=["Prestataire A", "Prestataire B", "Prestataire C"],
        columns=list(mk.DEFAULT_CRITERIA))
    base = default.reindex(columns=criteria).fillna(3).reset_index(names="prestataire")
    scores = st.data_editor(base, num_rows="dynamic", hide_index=True, width="stretch", key=f"scores_{'|'.join(criteria)}",
                            column_config={c: st.column_config.NumberColumn(min_value=0, max_value=5, step=0.5) for c in criteria})
    scores = scores.dropna(subset=["prestataire"]).set_index("prestataire").fillna(0)
    if scores.empty or not criteria:
        return
    try:
        result = mk.score_vendors(scores[criteria], dict(zip(crit_df["critère"], crit_df["poids"].fillna(0))))
    except ValueError as exc:
        st.error(str(exc))
        return
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**3. Classement**")
        st.dataframe(result[["Score global /100", "Rang"]], width="stretch")
        best = result.index[0]
        st.success(f"🏆 Recommandation : **{best}** ({result.loc[best, 'Score global /100']:.1f}/100)")
        fig = px.bar(result.reset_index(), x="prestataire", y=criteria, title="Contribution des critères au score")
        fig.update_layout(height=320, legend=dict(orientation="h", y=-0.3), margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        theta = criteria + [criteria[0]]
        for i, (name, row) in enumerate(scores[criteria].iterrows()):
            vals = row.tolist() + [row.tolist()[0]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill="toself", name=str(name), opacity=0.55,
                                          line_color=ui.PALETTE[i % len(ui.PALETTE)]))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])), height=500, title="Radar comparatif")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Analyse de sensibilité** : le classement reste-t-il stable si un critère pèse +10 points ?")
    sens = []
    weights = dict(zip(crit_df["critère"], crit_df["poids"].fillna(0)))
    for c in criteria:
        w = dict(weights)
        w[c] += 10
        r = mk.score_vendors(scores[criteria], w)
        sens.append({"critère renforcé": c, "1er": r.index[0], "score": r.iloc[0]["Score global /100"]})
    st.dataframe(pd.DataFrame(sens), hide_index=True, width="stretch")
    st.download_button("⬇️ Grille de scoring (.xlsx)",
                       to_excel_bytes({"notes": scores.reset_index(), "resultats": result.reset_index(), "ponderations": crit_df}),
                       "grille_scoring_prestataires.xlsx")


# --------------------------------------------------------------------------- #
# Agile
# --------------------------------------------------------------------------- #
DOR = ["La story suit le format « En tant que… je veux… afin de… »", "La valeur métier est claire et priorisée",
       "Les critères d'acceptation sont rédigés et testables", "Les dépendances sont identifiées",
       "La story est estimée par l'équipe", "Elle tient dans un sprint (INVEST)"]
DOD = ["Développement / paramétrage terminé", "Critères d'acceptation validés", "Tests réalisés (fonctionnels, données)",
       "Revue par un pair", "Documentation mise à jour", "Validée par le Product Owner en revue de sprint",
       "Conforme RGPD et accessibilité"]


def _agile() -> None:
    st.markdown("#### 🧱 Atelier User Stories")
    backlog: list[dict] = st.session_state.setdefault("backlog", [])
    with st.form("story_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        role = c1.text_input("En tant que…", "responsable marketing")
        want = c2.text_input("je veux…", "visualiser le coût par lead de chaque campagne")
        why = c3.text_input("afin de…", "réallouer le budget vers les canaux les plus rentables")
        c1, c2, c3, c4 = st.columns(4)
        epic = c1.text_input("Epic", "Dashboard marketing")
        prio = c2.selectbox("Priorité (MoSCoW)", ["Must", "Should", "Could", "Won't"])
        pts = c3.selectbox("Story points", [1, 2, 3, 5, 8, 13, 21], index=2)
        sprint = c4.number_input("Sprint", 1, 30, 1)
        criteria = st.text_area("Critères d'acceptation (Gherkin : Étant donné / Quand / Alors — un par ligne)",
                                "Étant donné une campagne terminée, quand j'ouvre le dashboard, alors le coût par lead s'affiche")
        if st.form_submit_button("➕ Ajouter au backlog", type="primary"):
            backlog.append({"id": f"US-{len(backlog) + 1:03d}", "epic": epic,
                            "user story": f"En tant que {role}, je veux {want} afin de {why}.",
                            "critères d'acceptation": criteria, "priorité": prio, "points": pts, "sprint": int(sprint),
                            "statut": "À faire"})
    if backlog:
        df = st.data_editor(pd.DataFrame(backlog), hide_index=True, width="stretch", key="backlog_editor",
                            column_config={"statut": st.column_config.SelectboxColumn(options=["À faire", "En cours", "Fait"]),
                                           "priorité": st.column_config.SelectboxColumn(options=["Must", "Should", "Could", "Won't"])})
        st.session_state["backlog"] = df.to_dict("records")
        c1, c2, c3 = st.columns(3)
        c1.metric("Stories", len(df))
        c2.metric("Points totaux", int(df["points"].sum()))
        c3.metric("Points « Fait »", int(df.loc[df["statut"] == "Fait", "points"].sum()))
        vel = df.groupby(["sprint", "statut"])["points"].sum().reset_index()
        fig = px.bar(vel, x="sprint", y="points", color="statut", title="Charge par sprint",
                     color_discrete_map={"À faire": "#A0AEC0", "En cours": ui.PALETTE[2], "Fait": ui.PALETTE[1]})
        fig.update_layout(height=300, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
        b1, b2 = st.columns(2)
        b1.download_button("⬇️ Backlog (.xlsx)", to_excel_bytes(df), "backlog.xlsx", width="stretch")
        if b2.button("🗑️ Vider le backlog", width="stretch"):
            st.session_state["backlog"] = []
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Definition of Ready (DoR)**")
        for item in DOR:
            st.checkbox(item, key=f"dor_{item}")
    with c2:
        st.markdown("**Definition of Done (DoD)**")
        for item in DOD:
            st.checkbox(item, key=f"dod_{item}")

    st.divider()
    st.markdown("#### 🤖 Assistant : du besoin au backlog")
    need = st.text_area("Décrivez un besoin métier en langage naturel",
                        "Les commerciaux veulent être alertés quand un prospect chaud consulte plusieurs fois la page tarifs.")
    n = st.slider("Nombre de user stories", 2, 10, 4)
    if st.button("Découper en user stories"):
        prompt = (f"Découpe ce besoin en {n} user stories respectant INVEST, au format « En tant que… je veux… afin de… ». "
                  "Pour chacune : critères d'acceptation en Gherkin (Étant donné / Quand / Alors), priorité MoSCoW, "
                  "estimation en points de Fibonacci justifiée, dépendances. Termine par une proposition de DoR et DoD "
                  "adaptées et les risques RGPD éventuels.\n\nBesoin : " + need)
        llm.run_stream(llm.M2_CONTEXT, prompt, "stories_ai")
    elif st.session_state.get("stories_ai"):
        with st.container(border=True):
            st.markdown(st.session_state["stories_ai"])
    if st.session_state.get("stories_ai"):
        ui.download_markdown("⬇️ User stories (.md)", st.session_state["stories_ai"], "user_stories.md", key="dl_us")


def render() -> None:
    ui.hero("🎓 M2 Project Hub — Pilotage stratégique",
            "Cadrage, cahier des charges, unit economics, scoring prestataires et méthodes agiles.")
    tabs = st.tabs(["📋 Cadrage & CdC", "💶 Budget & calculateurs", "⚖️ Scoring prestataires", "🔄 Agile"])
    with tabs[0]:
        _cadrage()
    with tabs[1]:
        _calculators()
    with tabs[2]:
        _vendor_scoring()
    with tabs[3]:
        _agile()
