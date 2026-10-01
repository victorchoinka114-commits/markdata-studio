"""Analytics & Data Science : RFM, K-Means, modèles supervisés, schéma en étoile."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Segmentation RFM
# --------------------------------------------------------------------------- #
RFM_SEGMENTS = {
    "Champions": "Achats récents, fréquents et élevés : ambassadeurs à valoriser (témoignages, programme de référence).",
    "À fidéliser": "Bons clients au potentiel de montée en gamme : nurturing, cross-sell, contenus experts.",
    "En risque": "Clients autrefois actifs mais sans achat récent : plan de réactivation prioritaire.",
    "Perdus": "Peu d'achats et inactifs depuis longtemps : campagne de win-back ou désengagement.",
}
RFM_COLORS = {"Champions": "#1B9E77", "À fidéliser": "#2E5BFF", "En risque": "#E6A100", "Perdus": "#D1495B"}


def _quantile_score(series: pd.Series, reverse: bool = False, n_bins: int = 5) -> pd.Series:
    """Score 1..n_bins par quantiles (rang pour gérer les ex-æquo)."""
    bins = int(min(n_bins, series.nunique(), len(series)))
    if bins < 2:
        return pd.Series(3, index=series.index)
    labels = list(range(1, bins + 1))
    scores = pd.qcut(series.rank(method="first"), bins, labels=labels).astype(int)
    if bins != n_bins:  # ramène sur une échelle 1..5
        scores = 1 + (scores - 1) * (n_bins - 1) / (bins - 1)
        scores = scores.round().astype(int)
    return (n_bins + 1 - scores) if reverse else scores


def rfm_segment_label(r: int, f: int) -> str:
    if r >= 4 and f >= 4:
        return "Champions"
    if r <= 2 and f <= 2:
        return "Perdus"
    if r <= 2:
        return "En risque"
    return "À fidéliser"


def compute_rfm(df: pd.DataFrame, customer_col: str, date_col: str, amount_col: str,
                reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Calcule Récence (jours), Fréquence (nb commandes), Montant (somme) + scores et segments."""
    data = df[[customer_col, date_col, amount_col]].dropna().copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", dayfirst=True)
    data[amount_col] = pd.to_numeric(data[amount_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        raise ValueError("Aucune ligne exploitable (dates ou montants invalides).")
    ref = pd.Timestamp(reference_date) if reference_date is not None else data[date_col].max() + pd.Timedelta(days=1)
    rfm = data.groupby(customer_col).agg(
        derniere_commande=(date_col, "max"),
        frequence=(date_col, "count"),
        montant=(amount_col, "sum"),
    )
    rfm["recence_jours"] = (ref - rfm["derniere_commande"]).dt.days
    rfm["R"] = _quantile_score(rfm["recence_jours"], reverse=True)
    rfm["F"] = _quantile_score(rfm["frequence"])
    rfm["M"] = _quantile_score(rfm["montant"])
    rfm["score_RFM"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["score_global"] = rfm[["R", "F", "M"]].mean(axis=1).round(2)
    rfm["segment"] = [rfm_segment_label(r, f) for r, f in zip(rfm["R"], rfm["F"])]
    return rfm.reset_index()[[customer_col, "recence_jours", "frequence", "montant", "R", "F", "M",
                              "score_RFM", "score_global", "segment", "derniere_commande"]]


def rfm_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("segment").agg(
        clients=("segment", "size"),
        recence_moy=("recence_jours", "mean"),
        frequence_moy=("frequence", "mean"),
        montant_moy=("montant", "mean"),
        montant_total=("montant", "sum"),
    )
    summary["% clients"] = (summary["clients"] / summary["clients"].sum() * 100).round(1)
    summary["% CA"] = (summary["montant_total"] / summary["montant_total"].sum() * 100).round(1)
    summary["action recommandée"] = summary.index.map(RFM_SEGMENTS)
    return summary.round(1).reset_index()


# --------------------------------------------------------------------------- #
# Clustering K-Means
# --------------------------------------------------------------------------- #
@dataclass
class ClusteringResult:
    labels: np.ndarray
    projection: pd.DataFrame
    profile: pd.DataFrame
    silhouette: float | None
    explained_variance: list[float]
    index: pd.Index


def prepare_numeric(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Sélection des features numériques, suppression des lignes incomplètes."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X.dropna()


def elbow_analysis(df: pd.DataFrame, features: list[str], k_range=range(2, 11), scale: bool = True,
                   seed: int = 42) -> pd.DataFrame:
    """Inertie (méthode du coude) et silhouette pour chaque k."""
    X = prepare_numeric(df, features)
    Xs = StandardScaler().fit_transform(X) if scale else X.values
    rows = []
    for k in k_range:
        if k >= len(Xs):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        sil = silhouette_score(Xs, km.labels_, sample_size=min(3000, len(Xs)), random_state=seed)
        rows.append({"k": k, "inertie": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def run_kmeans(df: pd.DataFrame, features: list[str], k: int, scale: bool = True,
               n_components: int = 2, seed: int = 42) -> ClusteringResult:
    """K-Means avec StandardScaler intégré + projection ACP 2D/3D."""
    X = prepare_numeric(df, features)
    if len(X) <= k:
        raise ValueError("Pas assez de lignes complètes pour ce nombre de clusters.")
    Xs = StandardScaler().fit_transform(X) if scale else X.values
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    labels = km.labels_
    n_comp = min(n_components, Xs.shape[1])
    if Xs.shape[1] > n_comp:
        pca = PCA(n_components=n_comp, random_state=seed).fit(Xs)
        coords = pca.transform(Xs)
        var = pca.explained_variance_ratio_.tolist()
    else:
        coords, var = Xs[:, :n_comp], []
    cols = [f"Axe {i + 1}" for i in range(coords.shape[1])]
    proj = pd.DataFrame(coords, columns=cols, index=X.index)
    proj["cluster"] = [f"Cluster {c}" for c in labels]
    profile = X.assign(cluster=[f"Cluster {c}" for c in labels]).groupby("cluster").mean().round(2)
    profile.insert(0, "effectif", pd.Series(labels).value_counts().sort_index().values)
    sil = silhouette_score(Xs, labels, sample_size=min(3000, len(Xs)), random_state=seed) if k > 1 else None
    return ClusteringResult(labels, proj, profile, sil, var, X.index)


# --------------------------------------------------------------------------- #
# Modèles supervisés
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    task: str
    algo: str
    metrics: dict
    y_test: pd.Series
    y_pred: np.ndarray
    importance: pd.DataFrame
    confusion: pd.DataFrame | None = None
    cv_scores: np.ndarray | None = None
    pipeline: Pipeline | None = None
    warnings: list[str] = field(default_factory=list)


ALGOS = {
    "classification": ["Régression logistique", "Random Forest"],
    "regression": ["Régression linéaire", "Random Forest"],
}


def detect_task(y: pd.Series) -> str:
    """Classification si cible non numérique ou peu de modalités (<= 10)."""
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10:
        return "classification"
    return "regression"


def _build_pipeline(X: pd.DataFrame, task: str, algo: str, seed: int) -> Pipeline:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=30))]), cat_cols),
    ])
    if task == "classification":
        model = (LogisticRegression(max_iter=2000) if algo == "Régression logistique"
                 else RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))
    else:
        model = (LinearRegression() if algo == "Régression linéaire"
                 else RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1))
    return Pipeline([("pre", pre), ("model", model)])


def _aggregate_importance(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Importance par variable d'origine (somme sur les colonnes one-hot)."""
    pre: ColumnTransformer = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_)
    else:
        coef = np.asarray(model.coef_)
        raw = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    owners: list[str] = []
    for out_name in pre.get_feature_names_out():
        group, base = out_name.split("__", 1)
        if group == "num":
            owners.append(base)
        else:  # one-hot : 'colonne_modalite' -> on retient la colonne d'origine la plus longue qui matche
            candidates = [c for c in X.columns if base == str(c) or base.startswith(f"{c}_")]
            owners.append(max(candidates, key=lambda c: len(str(c))) if candidates else base)
    imp = pd.DataFrame({"variable": owners, "importance": raw}).groupby("variable", as_index=False).sum()
    total = imp["importance"].sum()
    if total > 0:
        imp["importance"] = imp["importance"] / total
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def detect_leakage(df: pd.DataFrame, target: str, features: list[str], threshold: float = 0.95) -> list[str]:
    """Signale les variables quasi identiques à la cible (risque de fuite de données)."""
    alerts = []
    y = df[target]
    for f in features:
        x = df[f]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            pair = pd.concat([x, y], axis=1).dropna()
            if len(pair) > 5 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                corr = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                if corr >= threshold:
                    alerts.append(f"« {f} » est corrélée à {corr:.2f} avec la cible : fuite de données probable.")
        if x.nunique(dropna=True) >= 0.95 * len(x.dropna()) and not pd.api.types.is_float_dtype(x):
            alerts.append(f"« {f} » est quasi unique par ligne (identifiant ?) : risque de sur-apprentissage.")
    return alerts


def train_model(df: pd.DataFrame, target: str, features: list[str], algo: str, task: str | None = None,
                test_size: float = 0.25, seed: int = 42) -> ModelResult:
    """Entraîne un modèle supervisé (pipeline complet sans fuite de données)."""
    data = df[features + [target]].dropna(subset=[target]).copy()
    X, y = data[features], data[target]
    task = task or detect_task(y)
    if task == "classification":
        y = y.astype(str)
        if y.nunique() < 2:
            raise ValueError("La cible ne contient qu'une seule classe.")
    else:
        y = pd.to_numeric(y, errors="coerce")
        X, y = X[y.notna()], y[y.notna()]
    for c in X.columns:  # booléens -> numériques
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    pipe = _build_pipeline(X, task, algo, seed)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    warns = detect_leakage(data, target, features)

    if task == "classification":
        avg = "binary" if y.nunique() == 2 else "weighted"
        pos = sorted(y.unique())[-1]
        kw = {"average": avg, "zero_division": 0}
        if avg == "binary":
            kw["pos_label"] = pos
        metrics = {
            "Accuracy": accuracy_score(y_te, y_pred),
            "Précision": precision_score(y_te, y_pred, **kw),
            "Rappel": recall_score(y_te, y_pred, **kw),
            "F1-Score": f1_score(y_te, y_pred, **kw),
        }
        if avg == "binary" and hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_te)[:, list(pipe.classes_).index(pos)]
            metrics["ROC AUC"] = roc_auc_score((y_te == pos).astype(int), proba)
        baseline = y_te.value_counts(normalize=True).max()
        metrics["Baseline (classe majoritaire)"] = baseline
        labels = sorted(y.unique())
        cm = pd.DataFrame(confusion_matrix(y_te, y_pred, labels=labels), index=labels, columns=labels)
        cv = cross_val_score(_build_pipeline(X, task, algo, seed), X_tr, y_tr, cv=5, scoring="f1_weighted")
        train_score = f1_score(y_tr, pipe.predict(X_tr), average="weighted", zero_division=0)
        test_score = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    else:
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        metrics = {"R²": r2_score(y_te, y_pred), "RMSE": rmse, "MAE": mean_absolute_error(y_te, y_pred)}
        cm = None
        cv = cross_val_score(_build_pipeline(X, task, algo, seed), X_tr, y_tr, cv=5, scoring="r2")
        train_score = r2_score(y_tr, pipe.predict(X_tr))
        test_score = metrics["R²"]

    if train_score - test_score > 0.15:
        warns.append(f"Écart train/test important ({train_score:.2f} vs {test_score:.2f}) : "
                     "sur-apprentissage probable (réduire la profondeur, plus de données, régularisation).")
    return ModelResult(task, algo, metrics, y_te, y_pred, _aggregate_importance(pipe, X), cm, cv, pipe, warns)


# --------------------------------------------------------------------------- #
# Préparation Power BI : schéma en étoile
# --------------------------------------------------------------------------- #
def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rôle suggéré de chaque colonne : clé, mesure, date, attribut de dimension."""
    rows = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        ratio = nunique / n
        name = str(col).lower()
        if pd.api.types.is_datetime64_any_dtype(s) or "date" in name:
            role = "Date (→ DimDate)"
        elif ratio > 0.95 and (name.startswith("id") or name.endswith("id") or "siren" in name or "code" in name
                               or not pd.api.types.is_float_dtype(s)):
            role = "Clé / identifiant"
        elif pd.api.types.is_numeric_dtype(s) and nunique > 15:
            role = "Mesure (→ table de faits)"
        else:
            role = "Attribut de dimension"
        rows.append({"colonne": col, "type": str(s.dtype), "valeurs uniques": nunique,
                     "cardinalité %": round(ratio * 100, 1), "rôle suggéré": role})
    return pd.DataFrame(rows)


def functional_dependencies(df: pd.DataFrame, columns: list[str]) -> list[tuple[str, str]]:
    """Paires (A, B) telles que A détermine B (chaque valeur de A n'a qu'un B).

    Utile pour regrouper des attributs dans une même dimension (ville -> code postal).
    """
    deps = []
    for a in columns:
        if df[a].nunique() < 2:
            continue
        for b in columns:
            if a == b:
                continue
            na, nb = df[a].nunique(), df[b].nunique()
            # Bijection (ex. ville <-> code postal) : on ne garde qu'un sens, selon l'ordre des colonnes
            if nb > na or (nb == na and columns.index(b) < columns.index(a)):
                continue
            if df[[a, b]].dropna().groupby(a)[b].nunique().max() == 1:
                deps.append((a, b))
    return deps


def suggest_dimensions(df: pd.DataFrame, attributes: list[str]) -> dict[str, list[str]]:
    """Regroupe les attributs en dimensions à partir des dépendances fonctionnelles."""
    deps = functional_dependencies(df, attributes)
    determined = {b for _, b in deps}
    roots = [a for a in attributes if a not in determined]
    dims: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for root in sorted(roots, key=lambda c: -df[c].nunique()):
        members = [root] + [b for a, b in deps if a == root and b not in assigned]
        members = list(dict.fromkeys(members))
        dims[f"Dim{_camel(root)}"] = members
        assigned.update(members)
    for a in attributes:
        if a not in assigned:
            dims[f"Dim{_camel(a)}"] = [a]
    return dims


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in str(name).replace("-", "_").split("_"))


def build_calendar(dates: pd.Series) -> pd.DataFrame:
    """Table de dimension calendrier (DimDate) au format Power BI."""
    d = pd.to_datetime(dates, errors="coerce", dayfirst=True).dropna()
    rng = pd.date_range(d.min().normalize(), d.max().normalize(), freq="D")
    cal = pd.DataFrame({"date": rng})
    cal["date_key"] = cal["date"].dt.strftime("%Y%m%d").astype(int)
    cal["annee"] = cal["date"].dt.year
    cal["trimestre"] = "T" + cal["date"].dt.quarter.astype(str)
    cal["mois_num"] = cal["date"].dt.month
    months = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
    cal["mois"] = cal["mois_num"].map(lambda m: months[m - 1])
    cal["semaine_iso"] = cal["date"].dt.isocalendar().week.astype(int)
    days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    cal["jour_semaine"] = cal["date"].dt.dayofweek.map(lambda x: days[x])
    cal["est_week_end"] = cal["date"].dt.dayofweek >= 5
    return cal[["date_key", "date", "annee", "trimestre", "mois_num", "mois", "semaine_iso",
                "jour_semaine", "est_week_end"]]


def build_star_schema(df: pd.DataFrame, dimensions: dict[str, list[str]], measures: list[str],
                      date_col: str | None = None) -> dict[str, pd.DataFrame]:
    """Éclate une table plate en table de faits + dimensions avec clés de substitution."""
    fact = pd.DataFrame(index=df.index)
    tables: dict[str, pd.DataFrame] = {}
    for dim_name, cols in dimensions.items():
        if not cols:
            continue
        key = f"{dim_name.removeprefix('Dim').lower()}_key"
        dim = df[cols].drop_duplicates().reset_index(drop=True)
        dim.insert(0, key, np.arange(1, len(dim) + 1))
        merged = df[cols].merge(dim, on=cols, how="left")
        fact[key] = merged[key].values
        tables[dim_name] = dim
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        fact["date_key"] = dates.dt.strftime("%Y%m%d").astype("Int64")
        tables["DimDate"] = build_calendar(dates)
    for m in measures:
        fact[m] = df[m].values
    return {"FaitPrincipal": fact.reset_index(drop=True), **tables}


def dax_measures(measures: list[str], fact: str = "FaitPrincipal") -> str:
    """Mesures DAX de base pour démarrer le rapport Power BI."""
    lines = []
    for m in measures:
        label = m.replace("_", " ").capitalize()
        lines.append(f"Total {label} = SUM('{fact}'[{m}])")
        lines.append(f"Moyenne {label} = AVERAGE('{fact}'[{m}])")
    lines.append(f"Nb lignes = COUNTROWS('{fact}')")
    if measures:
        m = measures[0]
        lines.append(
            f"{m} N-1 = CALCULATE([Total {m.replace('_', ' ').capitalize()}], "
            "SAMEPERIODLASTYEAR('DimDate'[date]))"
        )
    return "\n".join(lines)
