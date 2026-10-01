"""Calculateurs marketing (CAC, LTV, ROI) et scoring multicritère de prestataires."""
from __future__ import annotations

import pandas as pd


def safe_div(a: float, b: float) -> float | None:
    return None if not b else a / b


def cac(marketing_spend: float, sales_spend: float, new_customers: int) -> float | None:
    """Coût d'Acquisition Client = (dépenses marketing + commerciales) / nouveaux clients."""
    return safe_div(marketing_spend + sales_spend, new_customers)


def ltv(arpa_monthly: float, gross_margin_pct: float, monthly_churn_pct: float) -> float | None:
    """Customer Lifetime Value (modèle SaaS) = ARPA x marge brute / churn mensuel."""
    churn = monthly_churn_pct / 100
    return safe_div(arpa_monthly * gross_margin_pct / 100, churn)


def lifetime_months(monthly_churn_pct: float) -> float | None:
    return safe_div(1, monthly_churn_pct / 100)


def payback_months(cac_value: float | None, arpa_monthly: float, gross_margin_pct: float) -> float | None:
    """Nombre de mois pour rembourser le CAC avec la marge mensuelle."""
    if cac_value is None:
        return None
    return safe_div(cac_value, arpa_monthly * gross_margin_pct / 100)


def roi(revenue: float, cost: float) -> float | None:
    """ROI = (gains - coût) / coût."""
    return safe_div(revenue - cost, cost)


def campaign_funnel(budget: float, cpm: float, ctr_pct: float, conv_lead_pct: float,
                    lead_to_opp_pct: float, win_rate_pct: float, avg_deal: float) -> pd.DataFrame:
    """Simulation d'entonnoir B2B : impressions -> clics -> leads -> opportunités -> contrats."""
    impressions = budget / cpm * 1000 if cpm else 0
    clicks = impressions * ctr_pct / 100
    leads = clicks * conv_lead_pct / 100
    opps = leads * lead_to_opp_pct / 100
    deals = opps * win_rate_pct / 100
    revenue = deals * avg_deal
    return pd.DataFrame({
        "étape": ["Impressions", "Clics", "Leads (MQL)", "Opportunités (SQL)", "Contrats signés"],
        "volume": [impressions, clicks, leads, opps, deals],
        "coût unitaire (€)": [budget / impressions * 1000 if impressions else None,
                              safe_div(budget, clicks), safe_div(budget, leads),
                              safe_div(budget, opps), safe_div(budget, deals)],
    }).assign(revenu_estime=[None, None, None, None, revenue])


def ltv_cac_diagnostic(ratio: float | None) -> tuple[str, str]:
    """Interprétation standard du ratio LTV/CAC."""
    if ratio is None:
        return "info", "Ratio non calculable (vérifiez les valeurs saisies)."
    if ratio < 1:
        return "error", "Chaque client coûte plus qu'il ne rapporte : modèle non viable."
    if ratio < 3:
        return "warning", "Rentable mais sous le seuil de référence de 3:1 — optimiser l'acquisition ou la rétention."
    if ratio <= 5:
        return "success", "Zone saine (3:1 à 5:1) : acquisition efficace et croissance soutenable."
    return "info", "Ratio très élevé (> 5:1) : vous sous-investissez peut-être en acquisition."


# --------------------------------------------------------------------------- #
# Scoring prestataires
# --------------------------------------------------------------------------- #
DEFAULT_CRITERIA = {
    "Coût / TCO": 25,
    "Sécurité & conformité (RGPD, HDS, SecNumCloud)": 20,
    "Couverture fonctionnelle": 25,
    "Support & accompagnement": 15,
    "Délais de mise en œuvre": 15,
}


def score_vendors(scores: pd.DataFrame, weights: dict[str, float], scale_max: float = 5) -> pd.DataFrame:
    """Score pondéré sur 100.

    `scores` : index = prestataires, colonnes = critères, notes de 0 à `scale_max`
    (plus la note est élevée, meilleur est le prestataire sur le critère).
    """
    w = pd.Series(weights, dtype=float).reindex(scores.columns).fillna(0)
    total_w = w.sum()
    if total_w <= 0:
        raise ValueError("La somme des pondérations doit être positive.")
    w = w / total_w
    normalized = scores.astype(float) / scale_max
    result = pd.DataFrame(index=scores.index)
    for crit in scores.columns:
        result[crit] = (normalized[crit] * w[crit] * 100).round(2)
    result["Score global /100"] = result.sum(axis=1).round(1)
    result["Rang"] = result["Score global /100"].rank(ascending=False, method="min").astype(int)
    return result.sort_values("Rang")
