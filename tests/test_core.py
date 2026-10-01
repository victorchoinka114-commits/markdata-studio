"""Tests de la logique métier (pytest)."""
import numpy as np
import pandas as pd
import pytest

from modules.core import analytics as an
from modules.core import cleaning as cl
from modules.core import marketing as mk
from modules.core import tennis as tn
from modules.core.samples import demo_crm_b2b, demo_transactions


# ----------------------------- Nettoyage ---------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("06 12 34 56 78", "06 12 34 56 78"),
    ("06.12.34.56.78", "06 12 34 56 78"),
    ("+33 6 12 34 56 78", "06 12 34 56 78"),
    ("0033612345678", "06 12 34 56 78"),
    (612345678.0, "06 12 34 56 78"),
    ("12345", None),
    (None, None),
])
def test_normalize_phone(raw, expected):
    assert cl.normalize_phone_fr(raw) == expected


def test_normalize_phone_e164():
    assert cl.normalize_phone_fr("06 12 34 56 78", fmt="e164") == "+33612345678"


@pytest.mark.parametrize("raw, expected", [(1000, "01000"), ("69003", "69003"), (1000.0, "01000"), ("abc", None)])
def test_postal_code(raw, expected):
    assert cl.normalize_postal_code_fr(raw) == expected


def test_infer_types_keeps_identifiers():
    df = pd.DataFrame({"montant": ["1 234,50", "10", "3,2"], "code_postal": ["01000", "69003", "75001"],
                       "date": ["01/02/2024", "15/03/2024", "30/12/2024"]})
    out, changes = cl.infer_types(df)
    assert changes == {"montant": "numérique", "date": "date"}
    assert out["montant"].tolist() == [1234.5, 10.0, 3.2]
    assert out["code_postal"].tolist() == ["01000", "69003", "75001"]
    assert out["date"].iloc[0] == pd.Timestamp("2024-02-01")


def test_audit_and_cleaning_pipeline():
    df = demo_crm_b2b(n=200)
    report = cl.audit_dataframe(df)
    assert report.n_duplicates > 0
    assert report.missing["manquants"].sum() > 0
    assert 0 <= report.quality_score <= 100

    clean = cl.normalize_text(df, ["email"], case="lower")
    dedup = cl.deduplicate(clean, ["email"])
    assert dedup["email"].duplicated().sum() == 0
    assert len(dedup) < len(df)

    filled = cl.handle_missing(df, ["nps"], "median")
    assert filled["nps"].isna().sum() == 0
    filled = cl.handle_missing(df, ["fonction"], "constant", "Inconnu")
    assert (filled["fonction"] == "Inconnu").any()

    phones, invalid = cl.normalize_column(df, "telephone", "phone")
    valid = phones["telephone"].dropna().astype(str)
    assert valid.str.match(r"^0\d( \d\d){4}$").mean() > 0.95


def test_fuzzy_duplicates():
    df = pd.DataFrame({"email": ["A.B@x.fr", " a.b@x.fr ", "c@x.fr"], "v": [1, 2, 3]})
    assert len(cl.find_duplicates(df, ["email"], fuzzy=True)) == 2
    assert len(cl.deduplicate(df, ["email"], fuzzy=True)) == 2


def test_outliers_and_exports():
    df = pd.DataFrame({"x": list(range(1, 21)) + [1000]})
    assert cl.treat_outliers(df, ["x"], "drop")["x"].max() == 20
    assert cl.treat_outliers(df, ["x"], "clip")["x"].max() < 1000
    assert cl.to_excel_bytes(df)[:2] == b"PK"
    assert cl.to_csv_bytes(df).startswith(b"\xef\xbb\xbf")


# ----------------------------- Analytics ---------------------------------- #
def test_rfm_segments():
    tx = demo_transactions(n_clients=120)
    rfm = an.compute_rfm(tx, "id_client", "date_commande", "montant_ht")
    assert len(rfm) == 120
    assert set(rfm["segment"]) <= set(an.RFM_SEGMENTS)
    assert rfm[["R", "F", "M"]].min().min() >= 1 and rfm[["R", "F", "M"]].max().max() <= 5
    best = rfm.sort_values("recence_jours").iloc[0]
    assert best["R"] == 5
    summary = an.rfm_summary(rfm)
    assert summary["clients"].sum() == 120


def test_kmeans():
    df = demo_crm_b2b(n=200, dirty=False)
    res = an.run_kmeans(df, ["taux_utilisation", "nps", "nb_tickets_support", "score_engagement"], k=3, n_components=3)
    assert len(set(res.labels)) == 3
    assert {"Axe 1", "Axe 2", "Axe 3"} <= set(res.projection.columns)
    el = an.elbow_analysis(df, ["taux_utilisation", "nps"], k_range=range(2, 5))
    assert len(el) == 3


@pytest.mark.parametrize("algo", ["Régression logistique", "Random Forest"])
def test_classification(algo):
    df = demo_crm_b2b(n=400, dirty=False)
    feats = ["taux_utilisation", "nb_tickets_support", "nps", "anciennete_mois", "secteur", "offre_interet"]
    res = an.train_model(df, "churn", feats, algo)
    assert res.task == "classification"
    assert 0 <= res.metrics["F1-Score"] <= 1
    assert res.confusion.shape == (2, 2)
    assert set(res.importance["variable"]) == set(feats)


def test_regression_and_leakage():
    df = demo_crm_b2b(n=300, dirty=False)
    df["fuite"] = df["score_engagement"] * 2
    res = an.train_model(df, "score_engagement", ["taux_utilisation", "nb_interactions_marketing", "fuite"],
                         "Régression linéaire")
    assert res.task == "regression"
    assert res.metrics["R²"] > 0.99
    assert any("fuite" in w for w in res.warnings)


def test_star_schema():
    df = demo_crm_b2b(n=150, dirty=False)
    dims = an.suggest_dimensions(df, ["ville", "code_postal", "type_structure", "secteur"])
    assert any({"ville", "code_postal"} <= set(m) for m in dims.values())
    tables = an.build_star_schema(df, dims, ["mrr_eur"], "date_dernier_contact")
    fact = tables["FaitPrincipal"]
    assert len(fact) == len(df)
    assert "date_key" in fact and "DimDate" in tables
    for name, cols in dims.items():
        key = f"{name.removeprefix('Dim').lower()}_key"
        assert fact[key].notna().all()
        assert tables[name][key].is_unique


# ----------------------------- Marketing ---------------------------------- #
def test_marketing_calculators():
    assert mk.cac(60000, 40000, 50) == 2000
    assert mk.ltv(100, 80, 2) == pytest.approx(4000)
    assert mk.roi(150, 100) == pytest.approx(0.5)
    assert mk.cac(1, 1, 0) is None
    assert mk.ltv_cac_diagnostic(4)[0] == "success"
    funnel = mk.campaign_funnel(10000, 50, 1, 10, 20, 25, 10000)
    assert funnel["volume"].iloc[0] == pytest.approx(200000)


def test_vendor_scoring():
    scores = pd.DataFrame({"a": [5, 1], "b": [5, 5]}, index=["X", "Y"])
    res = mk.score_vendors(scores, {"a": 50, "b": 50})
    assert res.loc["X", "Score global /100"] == 100
    assert res.index[0] == "X"


# ------------------------------- Tennis ----------------------------------- #
def test_game_probabilities():
    assert tn.p_game(0.5) == pytest.approx(0.5)
    assert tn.p_game(0.6) == pytest.approx(0.7357, abs=1e-3)  # valeur de référence classique
    assert tn.p_set(0.6, 0.6) == pytest.approx(0.5, abs=0.02)
    assert tn.p_tiebreak(0.6, 0.6) == pytest.approx(0.5, abs=1e-9)


def test_live_win_probability():
    pre = tn.live_win_probability(0.65, 0.62)
    assert 0.5 < pre < 1
    # A mène 1 set à 0 et 5-0 : quasi gagné
    assert tn.live_win_probability(0.65, 0.62, (1, 0), (5, 0), (3, 0), True) > 0.99
    assert tn.live_win_probability(0.65, 0.62, (2, 0)) == 1.0
    symmetric = tn.live_win_probability(0.6, 0.6, best_of=5)
    assert symmetric == pytest.approx(0.5, abs=1e-6)


def test_simulate_match_ends():
    sim = tn.simulate_match(0.65, 0.6, best_of=3, seed=1)
    assert sim["proba_victoire_A"].iloc[-1] in (0.0, 1.0)
    assert sim["proba_victoire_A"].between(0, 1).all()


def test_tennis_demo_data():
    trk = tn.demo_tracking(300)
    assert {"x", "y", "type_coup", "point_gagne"} <= set(trk.columns)
    zones = tn.zone_analysis(trk)
    assert zones["pct_points_gagnes"].between(0, 100).all()
    m = tn.demo_match_stats(50)
    assert set(m["victoire"]) == {0, 1}
    assert len(tn.demo_player_profiles(80)) == 80
    fig = tn.court_figure()
    assert len(fig.layout.shapes) >= 9
