"""MODULE 1 — Data Refinery : import, audit qualité, nettoyage et export."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import ui
from modules.core import cleaning as cl
from modules.core.samples import demo_crm_b2b, demo_transactions

WORK_KEY = "refinery_df"
HISTORY_KEY = "refinery_history"


def _set_work(df: pd.DataFrame, action: str | None = None) -> None:
    """Met à jour le dataset de travail et empile l'historique (annulation possible)."""
    if action and WORK_KEY in st.session_state:
        st.session_state.setdefault(HISTORY_KEY, []).append((action, st.session_state[WORK_KEY]))
    st.session_state[WORK_KEY] = df


def _import_section() -> None:
    st.subheader("1. Import des données")
    c1, c2 = st.columns([3, 2])
    with c1:
        uploaded = st.file_uploader("Glissez-déposez un fichier .xlsx ou .csv",
                                    type=["xlsx", "xls", "csv", "txt"], accept_multiple_files=False)
    with c2:
        st.markdown("**Ou chargez un jeu de démonstration :**")
        if st.button("📇 CRM B2B secteur public (données sales)", width="stretch"):
            _load(demo_crm_b2b(), "demo_crm_b2b")
        if st.button("🧾 Historique de commandes (RFM)", width="stretch"):
            _load(demo_transactions(), "demo_transactions")
    if uploaded is not None and st.session_state.get("refinery_file") != uploaded.file_id:
        data = uploaded.getvalue()
        sheet = 0
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            sheets = cl.list_excel_sheets(data)
            if len(sheets) > 1:
                sheet = st.selectbox("Onglet à importer", sheets)
                if not st.button("Importer cet onglet"):
                    return
        try:
            df = cl.read_uploaded_file(uploaded.name, data, sheet)
        except Exception as exc:  # noqa: BLE001 - message utilisateur
            st.error(f"Lecture impossible : {exc}")
            return
        st.session_state["refinery_file"] = uploaded.file_id
        _load(df, uploaded.name.rsplit(".", 1)[0])


def _load(df: pd.DataFrame, name: str) -> None:
    typed, changes = cl.infer_types(df)
    st.session_state[HISTORY_KEY] = []
    st.session_state["refinery_name"] = name
    st.session_state["refinery_type_changes"] = changes
    _set_work(typed)
    st.rerun()


def _audit_section(df: pd.DataFrame) -> None:
    st.subheader("2. Audit de santé de la base")
    report = cl.audit_dataframe(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score qualité", f"{report.quality_score}/100")
    c2.metric("Lignes", f"{report.n_rows:,}".replace(",", " "))
    c3.metric("Colonnes", report.n_cols)
    c4.metric("Cellules manquantes", int(report.missing["manquants"].sum()))
    c5.metric("Doublons stricts", report.n_duplicates)

    changes = st.session_state.get("refinery_type_changes") or {}
    if changes:
        st.caption("Typage automatique appliqué : " + ", ".join(f"`{c}` → {t}" for c, t in changes.items()))

    t1, t2, t3, t4, t5 = st.tabs(["Valeurs manquantes", "Doublons", "Valeurs aberrantes", "Types", "Aperçu"])
    with t1:
        miss = report.missing[report.missing["manquants"] > 0]
        if miss.empty:
            st.success("Aucune valeur manquante 🎉")
        else:
            fig = px.bar(miss, x="% manquants", y="colonne", orientation="h", text="% manquants",
                         color="% manquants", color_continuous_scale="Reds")
            fig.update_layout(height=max(250, 28 * len(miss)), yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(miss, hide_index=True, width="stretch")
    with t2:
        keys = st.multiselect("Clés de comparaison (vide = toutes les colonnes)", df.columns.tolist(),
                              key="audit_dup_keys")
        fuzzy = st.toggle("Doublons partiels (ignore casse, espaces, accents, ponctuation)", value=True)
        dups = cl.find_duplicates(df, keys or None, fuzzy=fuzzy)
        st.write(f"**{len(dups)} lignes** impliquées dans "
                 f"**{dups['_groupe'].nunique() if not dups.empty else 0} groupes** de doublons.")
        if not dups.empty:
            st.dataframe(dups.head(500), hide_index=True, width="stretch")
    with t3:
        if report.outliers.empty:
            st.info("Pas de colonne numérique analysable.")
        else:
            st.dataframe(report.outliers, hide_index=True, width="stretch")
            col = st.selectbox("Visualiser la distribution", report.outliers["colonne"].tolist())
            st.plotly_chart(px.box(df, x=col, points="outliers", color_discrete_sequence=[ui.PALETTE[0]]),
                            width="stretch")
    with t4:
        st.dataframe(report.dtypes, hide_index=True, width="stretch")
        if not report.inconsistent_types.empty:
            st.warning("Colonnes au typage incohérent (mélange de nombres et de texte) :")
            st.dataframe(report.inconsistent_types, hide_index=True, width="stretch")
    with t5:
        st.dataframe(df.head(200), width="stretch")


def _cleaning_section(df: pd.DataFrame) -> None:
    st.subheader("3. Actions de nettoyage")
    tabs = st.tabs(["🕳️ Manquants", "👯 Dédoublonnage", "🔤 Texte", "📞 Téléphones & CP", "📈 Outliers", "🧩 Colonnes"])

    with tabs[0]:
        cols = st.multiselect("Colonnes à traiter", df.columns.tolist(),
                              default=[c for c in df.columns if df[c].isna().any()][:5], key="miss_cols")
        strat = st.selectbox("Stratégie", list(cl.MISSING_STRATEGIES), format_func=cl.MISSING_STRATEGIES.get)
        const = st.text_input("Valeur constante", "Non renseigné") if strat == "constant" else None
        if strat in ("mean", "median"):
            st.caption("ℹ️ Moyenne / médiane ne s'appliquent qu'aux colonnes numériques.")
        if st.button("Appliquer", key="apply_missing", type="primary", disabled=not cols):
            out = cl.handle_missing(df, cols, strat, const)
            _set_work(out, f"Manquants : {cl.MISSING_STRATEGIES[strat]} sur {len(cols)} colonne(s)")
            st.rerun()

    with tabs[1]:
        keys = st.multiselect("Clé(s) d'unicité (ex : siren, email, id_client)", df.columns.tolist(), key="dedup_keys")
        c1, c2 = st.columns(2)
        keep = c1.radio("Conserver", ["first", "last", "none"], horizontal=True,
                        format_func={"first": "La première", "last": "La dernière", "none": "Aucune"}.get)
        fuzzy = c2.toggle("Comparaison normalisée (doublons partiels)", value=True, key="dedup_fuzzy")
        if st.button("Dédoublonner", key="apply_dedup", type="primary"):
            out = cl.deduplicate(df, keys or None, keep=keep, fuzzy=fuzzy)
            _set_work(out, f"Dédoublonnage ({len(df) - len(out)} lignes supprimées)")
            st.rerun()

    with tabs[2]:
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                     and not pd.api.types.is_datetime64_any_dtype(df[c])]
        cols = st.multiselect("Colonnes texte", text_cols, default=text_cols, key="txt_cols")
        c1, c2, c3 = st.columns(3)
        case = c1.selectbox("Casse", [None, "lower", "upper", "title"],
                            format_func={None: "Inchangée", "lower": "minuscules", "upper": "MAJUSCULES",
                                         "title": "Première Lettre En Majuscule"}.get)
        strip = c2.checkbox("Supprimer les espaces superflus", value=True)
        accents = c3.checkbox("Retirer les accents", value=False)
        st.caption("Astuce : passez les emails en minuscules avant de dédoublonner.")
        if st.button("Normaliser le texte", key="apply_txt", type="primary", disabled=not cols):
            _set_work(cl.normalize_text(df, cols, case, strip, accents), f"Normalisation texte ({len(cols)} col.)")
            st.rerun()

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            guess = next((c for c in df.columns if any(k in str(c).lower() for k in ("tel", "phone", "mobile"))), None)
            col = st.selectbox("Colonne téléphone", [None] + df.columns.tolist(),
                               index=(df.columns.tolist().index(guess) + 1) if guess else 0)
            fmt = st.radio("Format", ["national", "e164"], horizontal=True,
                           format_func={"national": "06 12 34 56 78", "e164": "+33612345678"}.get)
            if st.button("Standardiser les téléphones", disabled=col is None):
                out, invalid = cl.normalize_column(df, col, "phone", fmt=fmt)
                _set_work(out, f"Téléphones standardisés ({invalid} invalides conservés)")
                st.rerun()
        with c2:
            guess = next((c for c in df.columns if any(k in str(c).lower() for k in ("postal", "cp", "zip"))), None)
            col = st.selectbox("Colonne code postal", [None] + df.columns.tolist(),
                               index=(df.columns.tolist().index(guess) + 1) if guess else 0)
            st.caption("Restaure le zéro initial perdu par Excel (1000 → 01000).")
            if st.button("Standardiser les codes postaux", disabled=col is None):
                out, invalid = cl.normalize_column(df, col, "postal")
                _set_work(out, f"Codes postaux standardisés ({invalid} invalides conservés)")
                st.rerun()

    with tabs[4]:
        num_cols = ui.numeric_columns(df)
        cols = st.multiselect("Colonnes numériques", num_cols, key="out_cols")
        c1, c2 = st.columns(2)
        method = c1.radio("Méthode", ["clip", "drop"], horizontal=True,
                          format_func={"clip": "Écrêter aux bornes IQR", "drop": "Supprimer les lignes"}.get)
        k = c2.slider("Coefficient IQR (1,5 = standard ; 3 = extrêmes seulement)", 1.0, 5.0, 1.5, 0.5)
        if st.button("Traiter les outliers", disabled=not cols):
            _set_work(cl.treat_outliers(df, cols, method, k), f"Outliers ({method}, k={k})")
            st.rerun()

    with tabs[5]:
        drop = st.multiselect("Supprimer des colonnes", df.columns.tolist(), key="drop_cols")
        c1, c2 = st.columns(2)
        rename_col = c1.selectbox("Renommer la colonne", [None] + df.columns.tolist())
        new_name = c2.text_input("Nouveau nom")
        if st.button("Appliquer les modifications de colonnes"):
            out = df.drop(columns=drop)
            if rename_col and new_name and rename_col in out:
                out = out.rename(columns={rename_col: new_name})
            _set_work(out, "Modification des colonnes")
            st.rerun()

    history = st.session_state.get(HISTORY_KEY, [])
    if history:
        with st.expander(f"🕘 Historique des actions ({len(history)})", expanded=False):
            for i, (action, _) in enumerate(history, 1):
                st.write(f"{i}. {action}")
            if st.button("↩️ Annuler la dernière action"):
                _, previous = history.pop()
                st.session_state[WORK_KEY] = previous
                st.rerun()


def _export_section(df: pd.DataFrame) -> None:
    st.subheader("4. Export & partage")
    name = st.session_state.get("refinery_name", "dataset")
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Télécharger en .xlsx", cl.to_excel_bytes(df), f"{name}_clean.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    sep = c2.selectbox("Séparateur CSV", [";", ","], label_visibility="collapsed")
    c2.download_button("⬇️ Télécharger en .csv", cl.to_csv_bytes(df, sep), f"{name}_clean.csv", mime="text/csv",
                       width="stretch")
    if c3.button("📤 Envoyer vers Analytics & ML", type="primary", width="stretch"):
        ui.save_dataset(f"{name} (nettoyé)", df.copy())
        st.success(f"« {name} (nettoyé) » est disponible dans les modules Analytics et ML.")


def render() -> None:
    ui.hero("🧪 Data Refinery", "Importez, auditez et assainissez vos fichiers Excel / CSV en quelques clics.")
    _import_section()
    df = st.session_state.get(WORK_KEY)
    if df is None:
        st.info("Commencez par importer un fichier ou charger un jeu de démonstration.")
        return
    st.divider()
    _audit_section(df)
    st.divider()
    _cleaning_section(df)
    st.divider()
    _export_section(df)
