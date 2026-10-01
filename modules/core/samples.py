"""Jeux de données de démonstration (fictifs) pour le contexte B2B secteur public.

Toutes les organisations, personnes et valeurs sont générées aléatoirement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRUCTURES = {
    "Commune": ("Collectivité", 0.34),
    "Communauté de communes": ("Collectivité", 0.14),
    "Communauté d'agglomération": ("Collectivité", 0.08),
    "Département": ("Collectivité", 0.05),
    "Région": ("Collectivité", 0.02),
    "Centre hospitalier": ("Santé", 0.10),
    "EHPAD": ("Santé", 0.09),
    "Entreprise privée": ("Entreprise", 0.18),
}
FONCTIONS = ["DGS", "DAF", "DRH", "DSI", "Élu(e)", "Directeur(trice) d'établissement", "Responsable paie"]
OFFRES = ["Finances", "RH / Paie", "Relation usager"]
VILLES = [("Lyon", "69003"), ("Bourg-en-Bresse", "01000"), ("Nantes", "44000"), ("Lille", "59000"),
          ("Toulouse", "31000"), ("Rennes", "35000"), ("Dijon", "21000"), ("Annecy", "74000"),
          ("Ajaccio", "20000"), ("Montpellier", "34000"), ("Laon", "02000"), ("Gap", "05000")]
PRENOMS = ["Camille", "Julien", "Sophie", "Nicolas", "Claire", "Thomas", "Isabelle", "Marc",
           "Nadia", "Karim", "Élodie", "Pierre", "Inès", "Laurent", "Aurélie", "Yanis"]
NOMS = ["Martin", "Bernard", "Dubois", "Lefèvre", "Moreau", "Garnier", "Roux", "Fontaine",
        "Benali", "Nguyen", "Perrin", "Chevalier", "Lambert", "Faure", "Rousseau", "Diallo"]


def _messy_phone(rng: np.random.Generator) -> object:
    d = "0" + str(rng.choice([1, 2, 3, 4, 5, 6, 7, 9])) + "".join(str(x) for x in rng.integers(0, 10, 8))
    style = rng.integers(0, 6)
    if style == 0:
        return " ".join(d[i:i + 2] for i in range(0, 10, 2))
    if style == 1:
        return ".".join(d[i:i + 2] for i in range(0, 10, 2))
    if style == 2:
        return "+33 " + d[1:]
    if style == 3:
        return float(d)  # zéro initial perdu par Excel
    if style == 4:
        return d
    return "0033" + d[1:]


def demo_crm_b2b(n: int = 400, seed: int = 42, dirty: bool = True) -> pd.DataFrame:
    """Base CRM B2B fictive (clients & prospects) avec cible churn.

    Si `dirty=True`, on injecte volontairement des défauts de qualité :
    manquants, doublons, espaces, casse incohérente, téléphones hétérogènes,
    codes postaux tronqués et valeurs aberrantes.
    """
    rng = np.random.default_rng(seed)
    types = list(STRUCTURES)
    probs = np.array([STRUCTURES[t][1] for t in types])
    probs = probs / probs.sum()
    rows = []
    for i in range(n):
        t = rng.choice(types, p=probs)
        secteur = STRUCTURES[t][0]
        ville, cp = VILLES[rng.integers(len(VILLES))]
        prenom, nom = rng.choice(PRENOMS), rng.choice(NOMS)
        pop = int(rng.lognormal(9, 1.2)) if secteur == "Collectivité" else int(rng.lognormal(5.5, 0.8))
        statut = "Client" if rng.random() < 0.65 else "Prospect"
        anciennete = int(rng.integers(1, 120)) if statut == "Client" else 0
        nb_users = max(1, int(rng.lognormal(2.5, 0.9)))
        utilisation = float(np.clip(rng.beta(5, 2), 0, 1))
        tickets = int(rng.poisson(3 + 6 * (1 - utilisation)))
        nps = int(np.clip(rng.normal(30 + 50 * utilisation - 3 * tickets, 20), -100, 100))
        mrr = round(float(rng.lognormal(6.3, 0.7)) * (1.6 if secteur == "Collectivité" else 1), 2) if statut == "Client" else 0.0
        interactions = int(rng.poisson(4 + 8 * utilisation))
        engagement = float(np.clip(20 + 50 * utilisation + 2.5 * interactions - 1.5 * tickets + rng.normal(0, 8), 0, 100))
        # Probabilité de churn : faible utilisation, tickets, NPS bas, faible ancienneté
        logit = -1.2 - 3.5 * (utilisation - 0.6) + 0.18 * (tickets - 4) - 0.015 * nps - 0.01 * anciennete
        churn = int(rng.random() < 1 / (1 + np.exp(-logit))) if statut == "Client" else None
        last_contact = pd.Timestamp("2026-09-30") - pd.Timedelta(days=int(rng.exponential(90)))
        orga = f"{t} de {ville}" if secteur != "Entreprise" else f"{nom} {rng.choice(['SAS', 'SA', 'SARL'])}"
        rows.append({
            "id_client": f"C{i + 1:05d}",
            "organisation": orga,
            "type_structure": t,
            "secteur": secteur,
            "siren": "".join(str(x) for x in rng.integers(0, 10, 9)),
            "contact": f"{prenom} {nom}",
            "fonction": rng.choice(FONCTIONS),
            "email": f"{prenom.lower()}.{nom.lower()}@{ville.lower().replace(' ', '-')}.fr"
                     .replace("é", "e").replace("è", "e").replace("É", "e"),
            "telephone": _messy_phone(rng),
            "code_postal": cp,
            "ville": ville,
            "population_ou_effectif": pop,
            "offre_interet": rng.choice(OFFRES),
            "statut": statut,
            "anciennete_mois": anciennete,
            "nb_utilisateurs": nb_users,
            "taux_utilisation": round(utilisation, 3),
            "nb_tickets_support": tickets,
            "nps": nps,
            "mrr_eur": mrr,
            "nb_interactions_marketing": interactions,
            "score_engagement": round(engagement, 1),
            "date_dernier_contact": last_contact.date().isoformat(),
            "churn": churn,
        })
    df = pd.DataFrame(rows)
    if not dirty:
        return df

    # ---- Injection de défauts de qualité ----
    idx = df.sample(frac=0.08, random_state=seed).index
    df.loc[idx, "email"] = "  " + df.loc[idx, "email"].str.upper() + " "
    idx = df.sample(frac=0.06, random_state=seed + 1).index
    df.loc[idx, "organisation"] = df.loc[idx, "organisation"].str.lower() + "  "
    idx = df.sample(frac=0.25, random_state=seed + 2).index
    df["code_postal"] = df["code_postal"].astype(object)
    df.loc[idx, "code_postal"] = df.loc[idx, "code_postal"].astype(int)  # '01000' -> 1000
    for col, frac in [("fonction", 0.05), ("nps", 0.07), ("score_engagement", 0.04), ("telephone", 0.06)]:
        idx = df.sample(frac=frac, random_state=seed + len(col)).index
        df.loc[idx, col] = None
    idx = df[df["statut"] == "Client"].sample(n=5, random_state=seed + 3).index
    df.loc[idx, "mrr_eur"] = df.loc[idx, "mrr_eur"] * 25  # valeurs aberrantes
    dups = df.sample(n=max(5, n // 25), random_state=seed + 4).copy()
    dups["email"] = dups["email"].str.strip().str.title()  # doublons partiels
    exact = df.sample(n=max(3, n // 50), random_state=seed + 5)
    return pd.concat([df, dups, exact], ignore_index=True)


def demo_transactions(n_clients: int = 300, seed: int = 7, end: str = "2026-09-30") -> pd.DataFrame:
    """Historique de commandes / renouvellements pour la segmentation RFM."""
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end)
    rows = []
    for c in range(n_clients):
        activity = rng.choice(["fort", "moyen", "faible", "inactif"], p=[0.2, 0.35, 0.3, 0.15])
        n_orders = {"fort": rng.integers(8, 25), "moyen": rng.integers(3, 10),
                    "faible": rng.integers(1, 4), "inactif": rng.integers(1, 3)}[activity]
        last_gap = {"fort": 30, "moyen": 120, "faible": 300, "inactif": 600}[activity]
        basket = rng.lognormal(7.5 if activity == "fort" else 6.8, 0.6)
        for _ in range(int(n_orders)):
            days_ago = int(rng.exponential(last_gap)) + int(rng.integers(0, 60))
            rows.append({
                "id_client": f"C{c + 1:05d}",
                "date_commande": (end_ts - pd.Timedelta(days=min(days_ago, 1400))).date(),
                "montant_ht": round(float(basket * rng.uniform(0.5, 1.6)), 2),
                "offre": rng.choice(OFFRES),
            })
    df = pd.DataFrame(rows)
    df["date_commande"] = pd.to_datetime(df["date_commande"])
    return df.sort_values("date_commande").reset_index(drop=True)
