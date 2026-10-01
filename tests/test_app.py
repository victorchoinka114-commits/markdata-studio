"""Tests de fumée de l'interface (streamlit.testing)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

PAGES = {
    "data_refinery": "modules.data_refinery",
    "analytics_lab": "modules.analytics_lab",
    "campaign_lab": "modules.campaign_lab",
    "project_hub": "modules.project_hub",
    "tennis_lab": "modules.tennis_lab",
}


def _page_script(module: str, preload: str = "") -> str:
    return f"""
import streamlit as st
from modules import ui
import {module} as page
ui.inject_css()
{preload}
page.render()
"""


def _run(script: str) -> AppTest:
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_home():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception


def test_pages_render_empty():
    for module in PAGES.values():
        _run(_page_script(module))


def test_refinery_with_demo():
    at = _run(_page_script("modules.data_refinery", """
from modules.core.samples import demo_crm_b2b
from modules.core.cleaning import infer_types
if "refinery_df" not in st.session_state:
    st.session_state["refinery_df"] = infer_types(demo_crm_b2b())[0]
"""))
    assert any("Score qualité" in m.label for m in at.metric)
    before = len(at.session_state["refinery_df"])
    next(b for b in at.button if b.label == "Dédoublonner").click().run()
    assert not at.exception
    assert len(at.session_state["refinery_df"]) < before


def test_analytics_with_demo():
    preload = """
from modules.core.samples import demo_crm_b2b, demo_transactions
ui.save_dataset("crm", demo_crm_b2b(n=250, dirty=False))
"""
    at = _run(_page_script("modules.analytics_lab", preload))
    next(b for b in at.button if b.label == "🚀 Entraîner le modèle").click().run()
    assert not at.exception
    assert any(m.label == "F1-Score" for m in at.metric)


def test_rfm_with_demo():
    preload = """
from modules.core.samples import demo_transactions
ui.save_dataset("tx", demo_transactions(n_clients=100))
"""
    at = _run(_page_script("modules.analytics_lab", preload))
    next(b for b in at.button if b.label == "Calculer la segmentation RFM").click().run()
    assert not at.exception
    assert "rfm_result" in at.session_state


def test_tennis_with_demo():
    preload = """
from modules import tennis_lab
if "tennis_data" not in st.session_state:
    tennis_lab._load_demo()
"""
    _run(_page_script("modules.tennis_lab", preload))


def test_project_hub_cdc():
    at = _run(_page_script("modules.project_hub"))
    next(b for b in at.button if "Générer le cahier des charges" in b.label).click().run()
    assert not at.exception
    assert at.session_state["cdc_md"].startswith("# Cahier des charges")
