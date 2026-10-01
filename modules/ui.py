"""Composants d'interface partagés : style, en-têtes, KPI, gestion des datasets."""
from __future__ import annotations

import pandas as pd
import streamlit as st

PALETTE = ["#2E5BFF", "#1B9E77", "#E6A100", "#D1495B", "#7B61FF", "#00A3BF", "#8C6D46", "#5C677D"]

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
.md-hero {background: linear-gradient(120deg, #1A2B6D 0%, #2E5BFF 60%, #00A3BF 100%);
  color: #fff; padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem;}
.md-hero h1 {color: #fff; font-size: 1.75rem; margin: 0 0 .25rem 0;}
.md-hero p {color: #E6ECFF; margin: 0; font-size: .98rem;}
.md-card {border: 1px solid #E3E8F2; border-radius: 12px; padding: 1rem 1.1rem; background: #fff;
  height: 100%; box-shadow: 0 1px 2px rgba(16,24,40,.04);}
.md-card h4 {margin: 0 0 .4rem 0; font-size: 1.02rem;}
.md-card p {margin: 0; color: #4A5568; font-size: .9rem;}
.md-badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
  background:#EEF2FF; color:#2E5BFF; margin-right: 4px; font-weight: 600;}
div[data-testid="stMetric"] {background: #F7F9FC; border: 1px solid #E3E8F2; border-radius: 12px;
  padding: .7rem .9rem;}
section[data-testid="stSidebar"] .md-brand {font-weight: 800; font-size: 1.25rem; color: #1A2B6D;}
@media (prefers-color-scheme: dark) {
  .md-card {background: #161B26; border-color: #2A3245;}
  .md-card p {color: #B8C0D0;}
  div[data-testid="stMetric"] {background: #161B26; border-color: #2A3245;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="md-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="md-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def fmt_eur(value: float | None, decimals: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f} €".replace(",", " ").replace(".", ",")


def fmt_num(value: float | None, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}{suffix}".replace(",", " ").replace(".", ",")


# --------------------------------------------------------------------------- #
# Gestion des datasets partagés entre modules
# --------------------------------------------------------------------------- #
def datasets() -> dict[str, pd.DataFrame]:
    return st.session_state.setdefault("datasets", {})


def save_dataset(name: str, df: pd.DataFrame, make_active: bool = True) -> None:
    datasets()[name] = df
    if make_active:
        st.session_state["active_dataset"] = name


def dataset_picker(label: str = "Jeu de données", key: str = "picker") -> tuple[str | None, pd.DataFrame | None]:
    """Sélecteur des datasets chargés (Data Refinery ou démo)."""
    store = datasets()
    if not store:
        return None, None
    names = list(store)
    active = st.session_state.get("active_dataset")
    idx = names.index(active) if active in names else 0
    name = st.selectbox(label, names, index=idx, key=key)
    return name, store[name]


def require_dataset(key: str, demo_loaders: dict) -> tuple[str | None, pd.DataFrame | None]:
    """Sélecteur de dataset, avec boutons de chargement de démo si aucun n'est disponible."""
    name, df = dataset_picker(key=key)
    if df is None:
        st.info("Aucun jeu de données chargé. Importez un fichier dans **Data Refinery** "
                "ou chargez un jeu de démonstration :")
        cols = st.columns(len(demo_loaders))
        for col, (label, loader) in zip(cols, demo_loaders.items()):
            if col.button(label, key=f"{key}_{label}", width="stretch"):
                save_dataset(label, loader())
                st.rerun()
    return name, df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    return [c for c in df.columns if c not in numeric_columns(df) and df[c].nunique() <= max_unique]


def download_markdown(label: str, text: str, filename: str, key: str | None = None) -> None:
    st.download_button(label, text.encode("utf-8"), file_name=filename, mime="text/markdown", key=key)
