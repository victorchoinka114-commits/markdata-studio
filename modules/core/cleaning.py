"""Data Refinery : import, audit qualité et nettoyage de données tabulaires.

Fonctions pures (aucune dépendance Streamlit) afin d'être testables et
réutilisables dans des scripts ou notebooks.
"""
from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Colonnes dont le contenu ressemble à un nombre mais qui sont des identifiants
# (on ne doit jamais les convertir en numérique : perte des zéros initiaux).
ID_LIKE_PATTERN = re.compile(
    r"(t[ée]l|phone|mobile|portable|fax|siren|siret|postal|zip|insee|"
    r"(^|_)cp($|_)|(^|_)id($|_)|code|iban|num[ée]ro)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
def read_uploaded_file(name: str, data: bytes, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Lit un fichier .csv / .xlsx / .xls depuis ses octets.

    Pour les CSV : détection de l'encodage (UTF-8, UTF-8 BOM, Latin-1) et du
    séparateur (`,` `;` tabulation `|`) via le sniffer de pandas.
    """
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)
    if lower.endswith((".csv", ".txt", ".tsv")):
        text = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError("Encodage du fichier non reconnu.")
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    raise ValueError(f"Format non supporté : {name}")


def list_excel_sheets(data: bytes) -> list[str]:
    """Renvoie la liste des onglets d'un classeur Excel."""
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _to_number(series: pd.Series) -> pd.Series:
    """Convertit des chaînes au format français ('1 234,56 €') en float."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[\s  €%]", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def infer_types(df: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, dict[str, str]]:
    """Typage automatique des colonnes texte (numérique, date).

    Une colonne est convertie si au moins `threshold` de ses valeurs non vides
    sont interprétables. Les colonnes « identifiants » (téléphone, SIREN, code
    postal…) sont volontairement laissées en texte.
    """
    out = df.copy()
    changes: dict[str, str] = {}
    for col in out.columns:
        s = out[col]
        if not _is_text(s) or ID_LIKE_PATTERN.search(str(col)):
            continue
        non_null = s.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if non_null.empty:
            continue
        # Codes numériques à zéro initial ou numéros "+33…" => identifiants, on ne touche pas.
        if non_null.str.fullmatch(r"0\d+|\+[\d\s\.]+").mean() > 0.2:
            continue
        num = _to_number(non_null)
        if num.notna().mean() >= threshold:
            out[col] = _to_number(s)
            changes[col] = "numérique"
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates = pd.to_datetime(non_null, errors="coerce", dayfirst=True, format="mixed")
        if dates.notna().mean() >= threshold:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[col] = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
            changes[col] = "date"
    return out, changes


# --------------------------------------------------------------------------- #
# Audit qualité
# --------------------------------------------------------------------------- #
@dataclass
class AuditReport:
    n_rows: int
    n_cols: int
    missing: pd.DataFrame
    n_duplicates: int
    outliers: pd.DataFrame
    inconsistent_types: pd.DataFrame
    quality_score: float
    dtypes: pd.DataFrame = field(default_factory=pd.DataFrame)


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Bornes de Tukey (Q1 - k*IQR, Q3 + k*IQR)."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def audit_dataframe(df: pd.DataFrame, iqr_k: float = 1.5) -> AuditReport:
    """Diagnostic complet : manquants, doublons, outliers, types incohérents."""
    n_rows, n_cols = df.shape

    # Valeurs manquantes (les chaînes vides comptent comme manquantes)
    blank = df.apply(lambda s: s.astype(str).str.strip().eq("") if _is_text(s) else pd.Series(False, index=s.index))
    miss_count = df.isna().sum() + blank.sum()
    missing = (
        pd.DataFrame({
            "colonne": df.columns,
            "manquants": miss_count.values,
            "% manquants": (miss_count.values / max(n_rows, 1) * 100).round(2),
        })
        .sort_values("% manquants", ascending=False)
        .reset_index(drop=True)
    )

    n_duplicates = int(df.duplicated().sum())

    # Outliers (méthode IQR) sur colonnes numériques
    rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if s.nunique() < 5:
            continue
        lo, hi = iqr_bounds(s, iqr_k)
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"colonne": col, "outliers": n_out, "% outliers": round(n_out / max(len(s), 1) * 100, 2),
                     "borne basse": round(lo, 3), "borne haute": round(hi, 3)})
    outliers = pd.DataFrame(rows, columns=["colonne", "outliers", "% outliers", "borne basse", "borne haute"])

    # Types incohérents : colonne texte mélangeant nombres et texte libre
    rows = []
    for col in df.columns:
        s = df[col]
        if not _is_text(s) or ID_LIKE_PATTERN.search(str(col)):
            continue
        vals = s.dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) == 0:
            continue
        ratio = _to_number(vals).notna().mean()
        if 0.05 < ratio < 0.95:
            rows.append({"colonne": col, "% numériques": round(ratio * 100, 1),
                         "exemples non numériques": ", ".join(vals[_to_number(vals).isna()].unique()[:3])})
    inconsistent = pd.DataFrame(rows, columns=["colonne", "% numériques", "exemples non numériques"])

    # Score qualité 0-100 : pénalités pondérées
    total_cells = max(n_rows * n_cols, 1)
    pct_missing = miss_count.sum() / total_cells
    pct_dup = n_duplicates / max(n_rows, 1)
    pct_out = outliers["outliers"].sum() / total_cells if not outliers.empty else 0
    pct_incons = len(inconsistent) / max(n_cols, 1)
    score = 100 * (1 - min(1, 1.5 * pct_missing + 1.0 * pct_dup + 0.5 * pct_out + 0.3 * pct_incons))

    dtypes = pd.DataFrame({
        "colonne": df.columns,
        "type": [str(t) for t in df.dtypes],
        "valeurs uniques": [df[c].nunique(dropna=True) for c in df.columns],
        "exemple": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    dtypes["exemple"] = dtypes["exemple"].astype(str)

    return AuditReport(n_rows, n_cols, missing, n_duplicates, outliers, inconsistent, round(score, 1), dtypes)


# --------------------------------------------------------------------------- #
# Actions de nettoyage
# --------------------------------------------------------------------------- #
MISSING_STRATEGIES = {
    "drop_rows": "Supprimer les lignes concernées",
    "drop_cols": "Supprimer les colonnes",
    "mean": "Imputer par la moyenne",
    "median": "Imputer par la médiane",
    "mode": "Imputer par le mode (valeur la plus fréquente)",
    "constant": "Remplacer par une constante",
    "ffill": "Propager la valeur précédente",
}


def blank_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les chaînes vides / espaces en NaN."""
    out = df.copy()
    for col in out.columns:
        if _is_text(out[col]):
            out[col] = out[col].mask(out[col].astype(str).str.strip().eq(""))
    return out


def handle_missing(df: pd.DataFrame, columns: list[str], strategy: str, constant=None) -> pd.DataFrame:
    """Traite les valeurs manquantes des `columns` selon `strategy`."""
    out = blank_to_nan(df)
    if not columns:
        return out
    if strategy == "drop_rows":
        return out.dropna(subset=columns).reset_index(drop=True)
    if strategy == "drop_cols":
        return out.drop(columns=columns)
    for col in columns:
        s = out[col]
        if strategy in ("mean", "median"):
            if not pd.api.types.is_numeric_dtype(s):
                continue  # non applicable à du texte
            out[col] = s.fillna(s.mean() if strategy == "mean" else s.median())
        elif strategy == "mode":
            mode = s.mode(dropna=True)
            if not mode.empty:
                out[col] = s.fillna(mode.iloc[0])
        elif strategy == "constant":
            value = constant
            if pd.api.types.is_numeric_dtype(s):
                try:
                    value = float(str(constant).replace(",", "."))
                except ValueError:
                    pass
            out[col] = s.astype(object).fillna(value) if not isinstance(value, float) else s.fillna(value)
        elif strategy == "ffill":
            out[col] = s.ffill()
        else:
            raise ValueError(f"Stratégie inconnue : {strategy}")
    return out


def _normalized_keys(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Clés normalisées (minuscules, sans espaces ni accents) pour doublons partiels."""
    keys = pd.DataFrame(index=df.index)
    for col in subset:
        s = df[col]
        if _is_text(s):
            s = (s.astype("string").str.normalize("NFKD")
                 .str.encode("ascii", errors="ignore").str.decode("ascii")
                 .str.lower().str.replace(r"[\s\.\-_]", "", regex=True))
        keys[col] = s
    return keys


def find_duplicates(df: pd.DataFrame, subset: list[str] | None = None, fuzzy: bool = False) -> pd.DataFrame:
    """Renvoie toutes les lignes impliquées dans un doublon (strict ou partiel)."""
    subset = subset or list(df.columns)
    keys = _normalized_keys(df, subset) if fuzzy else df[subset]
    mask = keys.duplicated(keep=False)
    groups = keys[mask].apply(lambda row: "|".join(str(v) for v in row), axis=1)
    return df[mask].assign(_groupe=groups).sort_values("_groupe")


def deduplicate(df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first",
                fuzzy: bool = False) -> pd.DataFrame:
    """Dédoublonne selon une ou plusieurs clés (ex : SIREN, email)."""
    subset = subset or list(df.columns)
    keys = _normalized_keys(df, subset) if fuzzy else df[subset]
    keep_arg = False if keep == "none" else keep
    return df[~keys.duplicated(keep=keep_arg)].reset_index(drop=True)


def normalize_text(df: pd.DataFrame, columns: list[str], case: str | None = None,
                   strip: bool = True, remove_accents: bool = False) -> pd.DataFrame:
    """Nettoyage textuel : espaces superflus, casse, accents."""
    out = df.copy()
    for col in columns:
        if not _is_text(out[col]):
            continue
        s = out[col].astype("string")
        if strip:
            s = s.str.replace(r"[\s ]+", " ", regex=True).str.strip()
        if remove_accents:
            s = s.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()
        out[col] = s.astype(object).where(s.notna(), None)
    return out


def _as_digit_string(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return re.sub(r"\D", "", str(value))


def normalize_phone_fr(value, fmt: str = "national"):
    """Standardise un numéro français.

    fmt="national" -> '06 12 34 56 78' ; fmt="e164" -> '+33612345678'.
    Renvoie None si le numéro est invalide.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    digits = _as_digit_string(value)
    if digits.startswith("0033"):
        digits = "0" + digits[4:]
    elif digits.startswith("33") and len(digits) == 11:
        digits = "0" + digits[2:]
    elif len(digits) == 9 and digits[0] in "123456789":
        digits = "0" + digits  # zéro initial perdu (Excel)
    if len(digits) != 10 or not digits.startswith("0") or digits[1] == "0":
        return None
    if fmt == "e164":
        return "+33" + digits[1:]
    return " ".join(digits[i:i + 2] for i in range(0, 10, 2))


def normalize_postal_code_fr(value):
    """Code postal sur 5 chiffres (restaure le zéro perdu : 1000 -> '01000')."""
    if value is None or pd.isna(value):
        return None
    digits = _as_digit_string(value)
    if len(digits) == 4:
        digits = "0" + digits
    if len(digits) != 5 or digits == "00000":
        return None
    return digits


def normalize_column(df: pd.DataFrame, column: str, kind: str, **kwargs) -> tuple[pd.DataFrame, int]:
    """Applique une normalisation 'phone' ou 'postal'. Renvoie (df, nb invalides)."""
    out = df.copy()
    func = normalize_phone_fr if kind == "phone" else normalize_postal_code_fr
    original = out[column]
    normalized = original.map(lambda v: func(v, **kwargs))
    invalid = int((original.notna() & normalized.isna()).sum())
    # Les valeurs invalides sont conservées telles quelles pour revue manuelle
    out[column] = normalized.where(normalized.notna(), original).astype(object)
    return out, invalid


def treat_outliers(df: pd.DataFrame, columns: list[str], method: str = "clip", k: float = 1.5) -> pd.DataFrame:
    """Écrêtage (winsorisation IQR) ou suppression des valeurs aberrantes."""
    out = df.copy()
    mask_drop = pd.Series(False, index=out.index)
    for col in columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        lo, hi = iqr_bounds(out[col].dropna(), k)
        if method == "clip":
            out[col] = out[col].clip(lo, hi)
        else:
            mask_drop |= (out[col] < lo) | (out[col] > hi)
    return out[~mask_drop].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def to_excel_bytes(sheets: dict[str, pd.DataFrame] | pd.DataFrame) -> bytes:
    """Export .xlsx (une ou plusieurs feuilles) en mémoire."""
    if isinstance(sheets, pd.DataFrame):
        sheets = {"donnees": sheets}
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return buffer.getvalue()


def to_csv_bytes(df: pd.DataFrame, sep: str = ";") -> bytes:
    """Export CSV compatible Excel FR (séparateur ; et BOM UTF-8)."""
    return df.to_csv(index=False, sep=sep).encode("utf-8-sig")
