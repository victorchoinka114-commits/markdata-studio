"""Intégration Claude (SDK officiel Anthropic) : client, streaming, sorties JSON.

Clé API : saisie masquée dans la sidebar, avec repli sur la variable
d'environnement ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

import anthropic
import streamlit as st

MODELS = {
    "Claude Opus 5.5 (recommandé)": "claude-opus-5-5",
    "Claude Sonnet 5.5 (rapide)": "claude-sonnet-5-5",
    "Claude Haiku 4.5 (économique)": "claude-haiku-4-5",
}
EFFORTS = {"Rapide": "low", "Équilibré": "medium", "Approfondi": "high"}
# Modèles acceptant la réflexion adaptative, `effort` et le repli serveur en cas de refus
ADVANCED_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

EKSAE_CONTEXT = """Tu es le copilote marketing & communication B2B d'un(e) alternant(e) chez eksaé, \
éditeur de logiciels pour le secteur public (collectivités territoriales), la santé et les entreprises, \
sur trois grands domaines : gestion financière, RH / paie et relation usager.

Cibles décisionnaires : DGS, DAF, DRH, DSI, élus, directeurs d'établissements de santé, responsables paie.
Enjeux typiques de ces cibles : contraintes budgétaires et réglementaires, modernisation et dématérialisation \
du service public, sécurité et souveraineté des données (RGPD, hébergement), conduite du changement, \
continuité de service, qualité de la relation aux usagers / agents, commande publique (marchés publics, UGAP…).

Règles de rédaction :
- Français professionnel, clair, sans jargon creux ; vouvoiement ; ton expert, sobre et rassurant.
- N'invente jamais de chiffres, de noms de produits, de références clients ou de témoignages : \
utilise des balises explicites [À COMPLÉTER : …] lorsque l'information manque.
- Respecte les contraintes RGPD pour l'emailing B2B (lien de désinscription, base légale).
- Structure tes réponses en Markdown lisible (titres, listes, tableaux si utile)."""

M2_CONTEXT = """Tu es un expert senior en data marketing, gestion de projet digital et intelligence artificielle, \
qui accompagne un(e) étudiant(e) en Master 2 Data Marketing & IA, en alternance chez un éditeur de logiciels \
B2B pour le secteur public. Tu produis des livrables professionnels, rigoureux et directement réutilisables \
(cahiers des charges, user stories, analyses), en français, au format Markdown. N'invente pas de faits \
sur l'entreprise : utilise [À COMPLÉTER : …] quand une information manque."""

TENNIS_COACH_CONTEXT = """Tu es un coach data science spécialisé dans l'analyse de la performance au tennis \
(tracking de balle façon Hawk-Eye, statistiques de matchs, modèles de probabilité de victoire, marketing \
des équipementiers). Tu es aussi un relecteur de code Python exigeant et pédagogue (pandas, numpy, \
scikit-learn, plotly). Quand tu évalues du code :
1. Donne un verdict global (note /20) et ce qui est réussi.
2. Liste les erreurs (syntaxe, logique, résultats faux) avec la ligne concernée.
3. Détecte les pièges ML : fuite de données (fit du scaler/encodeur avant le split, variables connues \
après l'issue du point/match), sur-apprentissage, mauvaise métrique, déséquilibre de classes, absence de \
validation croisée, graines aléatoires.
4. Propose une version corrigée et idiomatique du code (bloc ```python).
5. Termine par 2-3 conseils pour progresser et une question de réflexion.
Réponds en français, en Markdown."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    """Clé saisie dans la sidebar en priorité, sinon variable d'environnement."""
    return st.session_state.get("api_key_input") or os.environ.get("ANTHROPIC_API_KEY") or None


def is_configured() -> bool:
    return bool(get_api_key())


def current_model() -> str:
    return st.session_state.get("llm_model", "claude-opus-5-5")


def current_effort() -> str:
    return st.session_state.get("llm_effort", "medium")


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_api_key(), max_retries=2)


def _request_kwargs(system: str, prompt: str, max_tokens: int, effort: str | None = None) -> tuple[dict, bool]:
    """Paramètres de requête selon le modèle. Renvoie (kwargs, utiliser_beta)."""
    model = current_model()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model in ADVANCED_MODELS:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort or current_effort()}
        # Repli automatique côté serveur si un classifieur de sécurité refuse la requête
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
        return kwargs, True
    return kwargs, False


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "Clé API invalide ou expirée. Vérifiez la clé saisie dans la barre latérale."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "Accès refusé pour ce modèle avec cette clé API. Essayez un autre modèle."
    if isinstance(exc, anthropic.RateLimitError):
        return "Limite de débit atteinte. Patientez quelques secondes puis relancez."
    if isinstance(exc, anthropic.BadRequestError):
        return f"Requête refusée par l'API : {exc.message}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Erreur de l'API Claude ({exc.status_code}) : {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Impossible de joindre l'API Claude. Vérifiez votre connexion internet / proxy."
    return f"Erreur inattendue : {exc}"


# --------------------------------------------------------------------------- #
# Appels
# --------------------------------------------------------------------------- #
def stream_text(system: str, prompt: str, max_tokens: int = 32000, effort: str | None = None) -> Iterator[str]:
    """Générateur de texte en streaming (à passer à `st.write_stream`)."""
    kwargs, use_beta = _request_kwargs(system, prompt, max_tokens, effort)
    client = _client()
    api = client.beta.messages if use_beta else client.messages
    with api.stream(**kwargs) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n> ⚠️ La requête a été refusée par les garde-fous du modèle. Reformulez la demande."
    elif final.stop_reason == "max_tokens":
        yield "\n\n> ⚠️ Réponse tronquée (limite de longueur atteinte). Demandez une version plus courte."


def run_stream(system: str, prompt: str, state_key: str, max_tokens: int = 32000,
               effort: str | None = None) -> str | None:
    """Affiche une réponse en streaming dans Streamlit et la mémorise dans session_state."""
    if not is_configured():
        st.warning("🔑 Renseignez votre clé API Anthropic dans la barre latérale pour utiliser l'IA.")
        return None
    try:
        with st.container(border=True):
            text = st.write_stream(stream_text(system, prompt, max_tokens, effort))
    except anthropic.APIError as exc:
        st.error(_friendly_error(exc))
        return None
    st.session_state[state_key] = text
    return text


def generate_json(system: str, prompt: str, schema: dict, max_tokens: int = 16000,
                  effort: str | None = None) -> dict | None:
    """Sortie structurée validée par un schéma JSON (structured outputs)."""
    if not is_configured():
        st.warning("🔑 Renseignez votre clé API Anthropic dans la barre latérale pour utiliser l'IA.")
        return None
    kwargs, use_beta = _request_kwargs(system, prompt, max_tokens, effort)
    kwargs.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": schema}
    client = _client()
    api = client.beta.messages if use_beta else client.messages
    try:
        with api.stream(**kwargs) as stream:
            response = stream.get_final_message()
    except anthropic.APIError as exc:
        st.error(_friendly_error(exc))
        return None
    if response.stop_reason == "refusal":
        st.error("La requête a été refusée par les garde-fous du modèle. Reformulez la demande.")
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        st.error("Réponse JSON invalide (sortie probablement tronquée). Réessayez.")
        return None


def render_sidebar_settings() -> None:
    """Bloc de configuration IA dans la sidebar."""
    with st.sidebar.expander("🤖 Configuration IA (Claude)", expanded=not is_configured()):
        st.text_input(
            "Clé API Anthropic", type="password", key="api_key_input",
            help="Stockée uniquement en mémoire pour la session. À défaut, la variable "
                 "d'environnement ANTHROPIC_API_KEY est utilisée.",
            placeholder="sk-ant-…",
        )
        if st.session_state.get("api_key_input"):
            st.caption("✅ Clé saisie (session en cours)")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            st.caption("✅ Clé détectée dans ANTHROPIC_API_KEY")
        else:
            st.caption("⚠️ Aucune clé : les fonctions IA sont désactivées")
        label = st.selectbox("Modèle", list(MODELS), index=0)
        st.session_state["llm_model"] = MODELS[label]
        if MODELS[label] in ADVANCED_MODELS:
            eff = st.select_slider("Profondeur de réflexion", list(EFFORTS), value="Équilibré")
            st.session_state["llm_effort"] = EFFORTS[eff]
