"""MODULE 3 — Campaign Lab : marketing opérationnel B2B propulsé par Claude."""
from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from modules import llm, ui

CIBLES = ["DGS", "DAF / Directeur financier", "DRH", "DSI", "Élu(e) / Maire", "Directeur d'hôpital / d'EHPAD",
          "Responsable paie", "Secrétaire de mairie", "Dirigeant(e) d'entreprise"]
SECTEURS = ["Commune / intercommunalité", "Département / Région", "Établissement de santé / médico-social",
            "Entreprise privée"]
OFFRES = ["Gestion financière", "RH / Paie", "Relation usager", "Suite complète"]
OBJECTIFS = ["Invitation à un webinar", "Prise de rendez-vous / démo", "Nurturing de leads froids",
             "Relance après salon", "Téléchargement d'un livre blanc", "Montée en gamme (upsell clients)",
             "Réactivation de clients inactifs"]
FRAMEWORKS = {"AIDA": "Attention, Intérêt, Désir, Action",
              "PAS": "Problème, Agitation, Solution",
              "BAB": "Before, After, Bridge",
              "4U": "Utile, Urgent, Unique, Ultra-spécifique"}

EMAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "strategie": {"type": "string"},
        "emails": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "etape": {"type": "integer"},
                    "timing": {"type": "string"},
                    "objectif": {"type": "string"},
                    "objet_a": {"type": "string"},
                    "objet_b": {"type": "string"},
                    "pre_header": {"type": "string"},
                    "accroche": {"type": "string"},
                    "corps": {"type": "array", "items": {"type": "string"}},
                    "cta_libelle": {"type": "string"},
                    "ps": {"type": "string"},
                },
                "required": ["etape", "timing", "objectif", "objet_a", "objet_b", "pre_header", "accroche",
                             "corps", "cta_libelle", "ps"],
                "additionalProperties": False,
            },
        },
        "conseils_ab_test": {"type": "array", "items": {"type": "string"}},
        "kpis_cibles": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["strategie", "emails", "conseils_ab_test", "kpis_cibles"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Export email
# --------------------------------------------------------------------------- #
def email_to_text(mail: dict, signature: str) -> str:
    body = "\n\n".join(mail["corps"])
    return (f"OBJET (A) : {mail['objet_a']}\nOBJET (B) : {mail['objet_b']}\nPRÉ-EN-TÊTE : {mail['pre_header']}\n"
            f"{'-' * 60}\n{mail['accroche']}\n\n{body}\n\n➡ {mail['cta_libelle']} : [LIEN]\n\n{signature}\n\n"
            f"P.S. {mail['ps']}\n\n---\nVous recevez cet email car … Se désinscrire : [LIEN_DESINSCRIPTION]")


def email_to_html(mail: dict, signature: str, brand_color: str = "#2E5BFF") -> str:
    """Gabarit HTML responsive à base de tables, styles inline (compatible CRM / Outlook)."""
    e = html.escape
    paragraphs = "".join(
        f'<p style="margin:0 0 16px 0;font-size:15px;line-height:1.6;color:#2D3748;">{e(p)}</p>' for p in mail["corps"])
    sig = "<br>".join(e(line) for line in signature.splitlines())
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(mail['objet_a'])}</title>
</head>
<body style="margin:0;padding:0;background:#F4F6FA;font-family:Arial,Helvetica,sans-serif;">
<span style="display:none;max-height:0;overflow:hidden;opacity:0;">{e(mail['pre_header'])}</span>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#F4F6FA;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="600" cellspacing="0" cellpadding="0" style="max-width:600px;width:100%;background:#FFFFFF;border-radius:8px;">
<tr><td style="background:{brand_color};padding:20px 32px;border-radius:8px 8px 0 0;">
<span style="color:#FFFFFF;font-size:20px;font-weight:bold;">{{{{logo_ou_nom_marque}}}}</span></td></tr>
<tr><td style="padding:32px;">
<p style="margin:0 0 16px 0;font-size:15px;color:#2D3748;">{{{{civilite}}}} {{{{nom}}}},</p>
<p style="margin:0 0 16px 0;font-size:16px;line-height:1.6;color:#1A202C;font-weight:bold;">{e(mail['accroche'])}</p>
{paragraphs}
<table role="presentation" cellspacing="0" cellpadding="0" style="margin:24px 0;"><tr>
<td style="background:{brand_color};border-radius:6px;">
<a href="{{{{lien_cta}}}}" style="display:inline-block;padding:14px 28px;color:#FFFFFF;font-size:15px;font-weight:bold;text-decoration:none;">{e(mail['cta_libelle'])}</a>
</td></tr></table>
<p style="margin:0 0 16px 0;font-size:15px;line-height:1.6;color:#2D3748;">{sig}</p>
<p style="margin:0;font-size:14px;line-height:1.6;color:#4A5568;"><em>P.S. {e(mail['ps'])}</em></p>
</td></tr>
<tr><td style="padding:16px 32px;border-top:1px solid #E2E8F0;font-size:12px;color:#718096;">
Vous recevez cet email dans le cadre de nos échanges professionnels.
<a href="{{{{lien_desinscription}}}}" style="color:#718096;">Se désinscrire</a></td></tr>
</table></td></tr></table>
</body></html>"""


def _emailing() -> None:
    st.markdown("Générez une séquence d'emails B2B structurée, avec objets A/B, pré-en-têtes et gabarit HTML prêt pour le CRM.")
    c1, c2, c3 = st.columns(3)
    cible = c1.selectbox("Persona cible", CIBLES)
    secteur = c2.selectbox("Secteur", SECTEURS)
    offre = c3.selectbox("Offre mise en avant", OFFRES)
    c1, c2, c3 = st.columns(3)
    objectif = c1.selectbox("Objectif de campagne", OBJECTIFS)
    framework = c2.selectbox("Framework rédactionnel", list(FRAMEWORKS), format_func=lambda k: f"{k} — {FRAMEWORKS[k]}")
    n_mails = c3.slider("Emails dans la séquence", 1, 5, 3)
    c1, c2 = st.columns(2)
    ton = c1.select_slider("Ton", ["Institutionnel", "Expert", "Pédagogique", "Direct", "Chaleureux"], value="Expert")
    longueur = c2.select_slider("Longueur", ["Très court (< 80 mots)", "Court (80-150 mots)", "Moyen (150-250 mots)"],
                                value="Court (80-150 mots)")
    contexte = st.text_area("Contexte, actualité ou preuve à exploiter (optionnel)",
                            placeholder="Ex : webinar du 12/11 sur la réforme de la protection sociale complémentaire ; "
                                        "nouvelle fonctionnalité de dématérialisation des factures…")
    signature = st.text_area("Signature", "Prénom Nom\nFonction — eksaé\nTél. : [À COMPLÉTER]", height=90)

    if st.button("✨ Générer la séquence", type="primary"):
        prompt = f"""Crée une séquence de {n_mails} email(s) B2B.
- Persona : {cible} — secteur : {secteur}
- Offre : {offre}
- Objectif : {objectif}
- Framework : {framework} ({FRAMEWORKS[framework]}) — applique-le explicitement dans chaque email
- Ton : {ton} — longueur du corps : {longueur}
- Contexte / preuves : {contexte or "aucun, reste factuel et utilise [À COMPLÉTER] si besoin"}

Contraintes : objets < 60 caractères, sans majuscules abusives ni mots spam ; objet_a et objet_b testent \
deux leviers différents (ex : bénéfice vs curiosité) ; pré-en-tête de 60 à 100 caractères complémentaire de \
l'objet ; corps = 2 à 4 paragraphes courts (sans formule d'appel ni signature) ; un seul CTA clair ; \
timing = délai depuis l'email précédent (ex : J+0, J+4). Dans `strategie`, explique en 3-4 phrases la \
logique de la séquence. Donne 3 conseils de test A/B et les KPIs cibles (taux d'ouverture, de clic…) \
avec des fourchettes indicatives B2B secteur public, présentées comme des ordres de grandeur."""
        with st.spinner("Claude rédige la séquence…"):
            data = llm.generate_json(llm.EKSAE_CONTEXT, prompt, EMAIL_SCHEMA)
        if data:
            st.session_state["email_seq"] = data
            st.session_state["email_sig"] = signature

    data = st.session_state.get("email_seq")
    if not data:
        return
    sig = st.session_state.get("email_sig", signature)
    st.success(data["strategie"])
    for mail in data["emails"]:
        with st.expander(f"📧 Email {mail['etape']} — {mail['timing']} · {mail['objectif']}", expanded=mail["etape"] == 1):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Objet A** : {mail['objet_a']}  \n<small>{len(mail['objet_a'])} car.</small>", unsafe_allow_html=True)
            c2.markdown(f"**Objet B** : {mail['objet_b']}  \n<small>{len(mail['objet_b'])} car.</small>", unsafe_allow_html=True)
            st.caption(f"Pré-en-tête : {mail['pre_header']}")
            t1, t2, t3 = st.tabs(["Aperçu HTML", "Texte brut", "Code HTML"])
            html_code = email_to_html(mail, sig)
            text = email_to_text(mail, sig)
            with t1:
                components.html(html_code, height=620, scrolling=True)
            with t2:
                st.text(text)
            with t3:
                st.code(html_code, language="html")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ .txt", text, f"email_{mail['etape']}.txt", key=f"txt{mail['etape']}", width="stretch")
            d2.download_button("⬇️ .html", html_code, f"email_{mail['etape']}.html", mime="text/html",
                               key=f"html{mail['etape']}", width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🧪 Conseils A/B test**")
        st.markdown("\n".join(f"- {c}" for c in data["conseils_ab_test"]))
    with c2:
        st.markdown("**🎯 KPIs cibles**")
        st.markdown("\n".join(f"- {k}" for k in data["kpis_cibles"]))
    st.download_button("⬇️ Séquence complète (.json)", json.dumps(data, ensure_ascii=False, indent=2),
                       "sequence_emailing.json", mime="application/json")


# --------------------------------------------------------------------------- #
# Rédaction web & LinkedIn
# --------------------------------------------------------------------------- #
CONTENT_TYPES = {
    "Brief SEO": "un brief SEO complet : intention de recherche, persona, mot-clé principal et champ lexical, "
                 "questions à traiter (PAA), structure Hn recommandée, longueur cible, maillage interne suggéré, "
                 "CTA, critères E-E-A-T",
    "Plan d'article expert": "un plan détaillé d'article de thought leadership (H1, H2, H3) avec, pour chaque partie, "
                             "les idées clés, les preuves/sources à mobiliser et les encadrés",
    "Article complet": "un article complet optimisé SEO, structuré en Markdown (H1, H2, H3), avec introduction "
                       "accrocheuse, paragraphes courts, listes, un encadré « À retenir » et une conclusion avec CTA",
    "Métadonnées SEO": "3 propositions de balise Title (< 60 caractères) et 3 meta-descriptions (140-155 caractères), "
                       "un slug d'URL, des textes alternatifs d'images et un extrait optimisé pour la position 0",
}


def _content_studio() -> None:
    c1, c2 = st.columns([2, 1])
    sujet = c1.text_input("Sujet", "Dématérialisation de la chaîne comptable dans les collectivités : par où commencer ?")
    ctype = c2.selectbox("Livrable", list(CONTENT_TYPES))
    c1, c2, c3 = st.columns(3)
    kw = c1.text_input("Mot-clé principal", "dématérialisation comptable collectivités")
    kw2 = c2.text_input("Mots-clés secondaires", "chaîne comptable, Chorus Pro, M57, DGFiP")
    cible = c3.selectbox("Lecteur cible", CIBLES)
    c1, c2 = st.columns(2)
    angle = c1.selectbox("Angle éditorial", ["Pédagogique / guide pratique", "Retour d'expérience (à compléter)",
                                             "Analyse réglementaire", "Tendances & prospective", "Checklist opérationnelle"])
    longueur = c2.slider("Longueur cible (mots, article)", 600, 2500, 1200, 100)

    if st.button("✍️ Générer", type="primary"):
        prompt = (f"Rédige {CONTENT_TYPES[ctype]}.\nSujet : {sujet}\nMot-clé principal : {kw}\n"
                  f"Mots-clés secondaires : {kw2}\nLecteur cible : {cible}\nAngle : {angle}\n"
                  f"Longueur visée pour l'article : {longueur} mots.\n"
                  "Les références réglementaires doivent être exactes ; en cas de doute, signale-le explicitement.")
        llm.run_stream(llm.EKSAE_CONTEXT, prompt, "content_out")
    elif st.session_state.get("content_out"):
        with st.container(border=True):
            st.markdown(st.session_state["content_out"])
    if st.session_state.get("content_out"):
        ui.download_markdown("⬇️ Télécharger (.md)", st.session_state["content_out"], "contenu_web.md")

    st.divider()
    st.markdown("#### 🔗 Déclinaison LinkedIn")
    c1, c2, c3 = st.columns(3)
    n_posts = c1.slider("Nombre de posts", 1, 5, 3)
    voice = c2.selectbox("Voix", ["Page entreprise", "Profil d'un expert métier", "Profil d'un(e) commercial(e)"])
    fmt = c3.selectbox("Format", ["Post texte", "Carrousel (slides)", "Sondage + texte", "Mix des formats"])
    source = st.text_area("Contenu source (par défaut : le dernier contenu généré ci-dessus)",
                          value="", placeholder="Collez un article, un communiqué ou laissez vide…", height=100)
    if st.button("🚀 Générer les posts LinkedIn"):
        base = source or st.session_state.get("content_out") or sujet
        prompt = (f"Décline ce contenu en {n_posts} publication(s) LinkedIn B2B distinctes (format : {fmt}, voix : {voice}).\n"
                  "Pour chaque post : accroche forte sur les 2 premières lignes (avant « voir plus »), corps aéré, "
                  "1 question ou CTA final, 3 à 5 hashtags pertinents, et une suggestion de visuel. "
                  "Pour un carrousel, détaille le texte de chaque slide. Pas d'emojis excessifs.\n\n"
                  f"CONTENU SOURCE :\n{base[:12000]}")
        llm.run_stream(llm.EKSAE_CONTEXT, prompt, "linkedin_out")
    elif st.session_state.get("linkedin_out"):
        with st.container(border=True):
            st.markdown(st.session_state["linkedin_out"])
    if st.session_state.get("linkedin_out"):
        ui.download_markdown("⬇️ Posts LinkedIn (.md)", st.session_state["linkedin_out"], "posts_linkedin.md", key="dl_li")


# --------------------------------------------------------------------------- #
# Personas & diagnostic digital
# --------------------------------------------------------------------------- #
CHANNELS = ["Site web & UX", "SEO / contenus", "LinkedIn & social selling", "Emailing & nurturing",
            "Webinars & événements", "Marketing automation / CRM", "Relations presse & influence",
            "Mesure & data (analytics, dashboards)"]


def _personas() -> None:
    st.markdown("#### 👤 Générateur de buyer persona B2B")
    c1, c2, c3 = st.columns(3)
    role = c1.selectbox("Fonction", CIBLES, key="p_role")
    secteur = c2.selectbox("Type de structure", SECTEURS, key="p_sect")
    taille = c3.selectbox("Taille", ["< 3 500 habitants / < 50 agents", "3 500 – 20 000 hab.", "20 000 – 100 000 hab.",
                                     "> 100 000 hab. / grand établissement"])
    offre = st.selectbox("Offre concernée", OFFRES, key="p_off")
    insights = st.text_area("Insights terrain (verbatims commerciaux, retours support, études…)",
                            placeholder="Ex : « on manque de temps pour former les agents », peur de la migration des données…")
    if st.button("Créer la fiche persona", type="primary"):
        prompt = f"""Construis une fiche buyer persona B2B détaillée :
Fonction : {role} — structure : {secteur} — taille : {taille} — offre : {offre}
Insights terrain : {insights or "aucun (signale les hypothèses à valider)"}

Sections attendues (Markdown) : prénom fictif & citation ; profil et périmètre ; objectifs & indicateurs de \
succès ; irritants / freins (budget, marchés publics, sécurité, conduite du changement) ; motivations d'achat ; \
rôle dans le processus de décision (décideur, prescripteur, utilisateur) et autres parties prenantes ; \
canaux et sources d'information privilégiés ; objections typiques et réponses ; messages clés et contenus \
adaptés à chaque étape du parcours (découverte, considération, décision) ; hypothèses à valider."""
        llm.run_stream(llm.EKSAE_CONTEXT, prompt, "persona_out")
    elif st.session_state.get("persona_out"):
        with st.container(border=True):
            st.markdown(st.session_state["persona_out"])
    if st.session_state.get("persona_out"):
        ui.download_markdown("⬇️ Fiche persona (.md)", st.session_state["persona_out"], "persona.md", key="dl_persona")

    st.divider()
    st.markdown("#### 🩺 Canevas d'audit de présence digitale")
    st.caption("Évaluez la maturité actuelle et la cible (0 = inexistant, 5 = excellence) pour chaque levier.")
    rows = st.session_state.setdefault("audit_grid", pd.DataFrame({
        "levier": CHANNELS, "maturité actuelle": [2] * len(CHANNELS), "cible 12 mois": [4] * len(CHANNELS),
        "constat / preuve": [""] * len(CHANNELS)}))
    grid = st.data_editor(rows, hide_index=True, width="stretch", key="audit_editor",
                          column_config={
                              "levier": st.column_config.TextColumn(disabled=True),
                              "maturité actuelle": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
                              "cible 12 mois": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
                          })
    fig = go.Figure()
    theta = grid["levier"].tolist() + [grid["levier"].iloc[0]]
    for col, color in (("maturité actuelle", ui.PALETTE[3]), ("cible 12 mois", ui.PALETTE[0])):
        vals = grid[col].fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=theta, fill="toself", name=col, line_color=color, opacity=0.55))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])), height=460, margin=dict(t=30, b=30))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    gap = grid.assign(écart=grid["cible 12 mois"] - grid["maturité actuelle"]).sort_values("écart", ascending=False)
    c2.markdown("**Priorités (écart le plus fort)**")
    c2.dataframe(gap[["levier", "écart"]], hide_index=True, width="stretch")
    if st.button("🧠 Analyse & plan d'action par Claude"):
        prompt = ("Voici l'auto-évaluation de la présence digitale d'un éditeur de logiciels B2B secteur public :\n"
                  f"{grid.to_markdown(index=False) if _has_tabulate() else grid.to_csv(index=False)}\n\n"
                  "Produis : 1) un diagnostic synthétique (forces / faiblesses), 2) une matrice impact x effort des "
                  "actions, 3) une feuille de route 30/60/90 jours, 4) les KPIs de suivi par levier, "
                  "5) les quick wins réalisables par un(e) alternant(e).")
        llm.run_stream(llm.EKSAE_CONTEXT, prompt, "audit_out")
    elif st.session_state.get("audit_out"):
        with st.container(border=True):
            st.markdown(st.session_state["audit_out"])
    if st.session_state.get("audit_out"):
        ui.download_markdown("⬇️ Diagnostic (.md)", st.session_state["audit_out"], "diagnostic_digital.md", key="dl_audit")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def render() -> None:
    ui.hero("📣 Campaign Lab — Marketing B2B", "Emailing, contenus experts, LinkedIn, personas et diagnostic digital, "
            "propulsés par Claude et ancrés dans le contexte eksaé.")
    if not llm.is_configured():
        st.warning("🔑 Ce module utilise l'API Claude : renseignez votre clé dans la barre latérale "
                   "(ou la variable d'environnement ANTHROPIC_API_KEY).")
    tabs = st.tabs(["📧 Emailing & Nurturing", "✍️ Rédaction web & LinkedIn", "👤 Personas & Diagnostic digital"])
    with tabs[0]:
        _emailing()
    with tabs[1]:
        _content_studio()
    with tabs[2]:
        _personas()
