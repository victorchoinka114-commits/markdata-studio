# 📊 MarkData Studio

Application web locale (Python / Streamlit) de **data marketing B2B**, de **data science** et de **sports analytics**.
Elle sert à la fois d'outil métier au quotidien (alternance marketing & communication B2B dans l'édition de logiciels
pour le secteur public), de compagnon de Master 2 Data Marketing & IA, et de laboratoire d'apprentissage sur la data
appliquée au tennis.

## 🗂️ Arborescence

```
markdata-studio/
├── app.py                      # Point d'entrée : navigation, sidebar, page d'accueil
├── requirements.txt
├── pytest.ini
├── .streamlit/
│   └── config.toml             # Thème et options serveur
├── modules/
│   ├── ui.py                   # Style, en-têtes, cartes, datasets partagés entre pages
│   ├── llm.py                  # Intégration Claude (SDK Anthropic) : clé API, streaming, JSON structuré
│   ├── data_refinery.py        # MODULE 1 — Import, audit qualité, nettoyage, export
│   ├── analytics_lab.py        # MODULE 2 — Explorateur, RFM, K-Means, ML, Power BI
│   ├── campaign_lab.py         # MODULE 3 — Emailing, contenus SEO, LinkedIn, personas, diagnostic digital
│   ├── project_hub.py          # MODULE 4 — Cadrage & CdC, CAC/LTV/ROI, scoring prestataires, agile
│   ├── tennis_lab.py           # MODULE 5 — Tennis Data Lab & coach virtuel
│   └── core/                   # Logique métier pure, sans Streamlit (testable)
│       ├── cleaning.py         # Typage auto, audit, manquants, doublons, téléphones/CP, exports
│       ├── analytics.py        # RFM, K-Means + ACP, pipelines scikit-learn, schéma en étoile, DAX
│       ├── marketing.py        # CAC, LTV, payback, ROI, entonnoir, scoring multicritère
│       ├── tennis.py           # Court ITF à l'échelle, données de démo, zones, modèle de Markov
│       └── samples.py          # Jeux de démonstration fictifs (CRM B2B, commandes)
└── tests/
    ├── test_core.py            # Tests unitaires de la logique métier
    └── test_app.py             # Tests de fumée des pages (streamlit.testing)
```

## 🚀 Installation & lancement

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur <http://localhost:8501>.

**Clé API Claude (optionnelle)** : renseignez-la dans la barre latérale (champ masqué, conservé en mémoire
pour la session uniquement), ou définissez la variable d'environnement avant le lancement :

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell : $env:ANTHROPIC_API_KEY="sk-ant-..."
```

Sans clé, tous les modules fonctionnent sauf la génération de texte (Campaign Lab, enrichissement du CdC,
assistant user stories, coach virtuel).

## 🧩 Modules

| Module | Fonctionnalités |
|---|---|
| **🧪 Data Refinery** | Import `.xlsx`/`.csv` par glisser-déposer (détection encodage, séparateur et onglet), typage automatique (formats FR : `1 234,56`, dates `jj/mm/aaaa`) qui laisse intacts les identifiants. Audit : score qualité, % de manquants, doublons stricts et partiels, outliers IQR, types incohérents. Nettoyage en un clic : imputation, dédoublonnage par clés, normalisation du texte, téléphones FR (national/E.164), codes postaux (zéro initial restauré), écrêtage. Historique avec annulation. Export `.xlsx`/`.csv`. |
| **📊 Analytics & ML** | Explorateur Plotly (histogramme, nuage de points, boîte à moustaches, série temporelle, barres, corrélations). RFM (quintiles, segments Champions / À fidéliser / En risque / Perdus). K-Means avec StandardScaler, coude + silhouette, ACP 2D/3D, radar des profils. ML : régression linéaire/logistique et Random Forest en pipelines sans fuite de données, validation croisée, matrice de confusion, importance des variables, alertes sur-apprentissage et fuite. Power BI : rôle de chaque colonne, dimensions déduites des dépendances fonctionnelles, table de faits + clés de substitution, DimDate, mesures DAX, export. |
| **📣 Campaign Lab** | Séquences emailing (AIDA, PAS, BAB, 4U) en sortie JSON structurée : objets A/B, pré-en-têtes, export texte brut + **HTML responsive** prêt pour le CRM. Briefs SEO, plans, articles, métadonnées, déclinaisons LinkedIn. Personas B2B. Canevas d'audit digital avec radar et plan d'action. |
| **🎓 M2 Project Hub** | Formulaire de cadrage (SMART, périmètre, architecture, risques, livrables) qui génère un cahier des charges Markdown, puis une version enrichie par Claude. CAC, LTV, LTV/CAC, payback, ROI et entonnoir. Scoring prestataires pondéré avec radar et analyse de sensibilité. User stories, backlog MoSCoW/points, DoR/DoD, découpage d'un besoin par l'IA. |
| **🎾 Tennis Data Lab** | Données de démo (matchs, tracking, profils). Court 2D à l'échelle ITF avec rebonds et heatmaps. Vitesse vs précision, rallyes. Cas 1 : classification des styles de jeu et recommandation raquette/cordage. Cas 2 : profondeur et zones de vulnérabilité. Cas 3 : probabilité de victoire en direct (Markov point → jeu → set → match) et simulation point par point. Coach virtuel : 5 défis, exécution locale, revue du code par Claude (fuites de données, sur-apprentissage). |

## 🤖 Intégration Claude

- SDK officiel `anthropic` ; modèle par défaut **Claude Opus 5.5** (`claude-opus-5-5`), au choix dans la sidebar
  avec Sonnet 5.5 et Haiku 4.5.
- Streaming des réponses (`st.write_stream`), réflexion adaptative et profondeur réglable (`effort`).
- Sorties structurées par schéma JSON pour les emails (export HTML fiable).
- Repli automatique côté serveur si un garde-fou refuse une requête (`fallbacks: "default"`), gestion des erreurs
  (clé invalide, quota, réseau) avec messages en français.
- Les prompts système imposent de **ne jamais inventer** chiffres, produits ou références clients :
  les manques sont balisés `[À COMPLÉTER : …]`.

## ✅ Tests

```bash
pip install pytest
pytest
```

> Toutes les données de démonstration (organisations, personnes, joueurs, matchs) sont **fictives**.
