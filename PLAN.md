# Plan d'intégration — Research Opportunity Agent

## Étape 1 — Base technique : pipeline retrieval robuste

**Objectif** : stabiliser le pipeline existant et le rendre fiable et reproductible.

### 1.1 Ajouter `requirements.txt`

Créer un manifeste de dépendances complet :

```
sentence-transformers
faiss-cpu
numpy
scikit-learn
scipy
python-dotenv
requests
google-genai
openai
streamlit
tqdm
```

### 1.2 Nettoyer `main.py` et extraire la configuration

- Extraire les constantes (`EMBEDDING_MODEL`, `TOP_K_RETRIEVAL`, `TOP_K_FINAL`) dans un fichier de config centralisé (`src/config.py` ou un fichier YAML/JSON).
- Supprimer la duplication entre `main.py` et `run_evaluation.py`.
- Ajouter `src/__init__.py` et les `__init__.py` manquants dans les sous-dossiers (`builders/`, `builders/opportunity/`, `builders/profil/`, `collectors/`, `embeddings/`, `evaluation/`, `loaders/`, `matching/`, `models/`, `processors/`, `reranking/`, `vector_store/`).

### 1.3 Valider le pipeline sur un petit dataset

- S'assurer que `data/processed/opportunities.json` contient des données exploitables.
- Tester le flux complet : load → build index → embed researcher → retrieve top-K → rerank → output JSON.
- Vérifier que le LLM retourne du JSON valide et que le parsing est robuste (déjà partly fait dans `_parse_response`).

### 1.4 Définir les paramètres par défaut

Documenter et fixer :

| Paramètre | Valeur par défaut |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Retrieval top_k | 20 |
| Final top_k | 3 |
| LLM providers | Gemini + Qwen |
| LLM fallback | activé |

### Livrable

Un pipeline fonctionnel qui retourne des recommandations cohérentes pour un chercheur donné, avec gestion des erreurs LLM.

---

## Étape 2 — Reranking LLM fiable

**Objectif** : rendre le reranking robuste et comparable entre providers.

### 2.1 Structurer le prompt du reranker

Le prompt actuel (`src/reranking/llm_reranker.py:40-90`) est fonctionnel mais peut être amélioré :

- Ajouter des instructions de format strict (enum de `matching_areas` prédéfinis).
- Ajouter un exemple few-shot dans le prompt pour guider le LLM.
- Limiter le nombre de tokens dans la description des opportunités pour éviter les timeouts.

### 2.2 Améliorer le parsing JSON

Le parsing actuel (`_extract_json_payload` + `json.loads`) est minimaliste :

- Ajouter un fallback : si le JSON est invalide, tenter un `json.loads` après extraction de la sous-chaîne entre la première `{` et la dernière `}`.
- Ajouter la validation du schéma : s'assurer que chaque recommandation contient `opportunity_id`, `score`, `reason`, `matching_areas`.
- Retourner un objet typed (`@dataclass` ou `TypedDict`) au lieu d'un `dict` brut.

### 2.3 Comparaison Gemini vs Qwen

- Le dual-provider est déjà en place dans `main.py:97-100`.
- Ajouter un timing par provider (temps de réponse).
- Ajouter un compteur d'erreurs par provider.

### 2.4 Gestion des erreurs LLM

Le système d'erreurs (`src/llm/errors.py`) est bien structuré. Vérifier que :

- Le fallback cosine (déjà dans `main.py:172-180`) fonctionne pour tous les types d'erreur LLM.
- Le flag `LLM_ALLOW_FALLBACK` est bien respecté.

### Livrable

Un reranker qui retourne des JSON valides et exploitables, avec comparaison temps/qualité entre Gemini et Qwen.

---

## Étape 3 — Agent orchestration

**Objectif** : transformer le pipeline script en classe `RecommendationAgent` réutilisable.

### 3.1 Créer `src/agent/recommendation_agent.py`

```python
class RecommendationAgent:
    def __init__(self, config: PipelineConfig):
        self.embedder = Embedder(config.embedding_model)
        self.ranker = OpportunityRanker()
        self.profile_builder = StructuredProfileBuilder()
        self.opportunity_builder = StructuredOpportunityBuilder()
        self.rerankers = [...]  # LLMReranker instances
        self.vector_index = VectorIndex()  # ou FAISSIndex
        self.opportunities_by_id = {}
        self.index_records = []

    def load_data(self, researchers_path, opportunities_path): ...
    def build_index(self): ...
    def query(self, researcher_id: str | int) -> RecommendationResult: ...
    def query_from_profile(self, profile: Researcher) -> RecommendationResult: ...
```

### 3.2 Définir les modèles de sortie

```python
@dataclass
class RecommendationResult:
    researcher_id: str
    researcher_name: str
    recommendations: list[RecommendationItem]
    retrieval_top_k: list[RecommendationItem]
    provider_results: dict[str, list[RecommendationItem]]
    metadata: dict  # timing, errors, etc.

@dataclass
class RecommendationItem:
    opportunity_id: str
    title: str
    organization: str
    score: float
    reason: str
    matching_areas: list[str]
    source: str  # retrieval | reranked
```

### 3.3 Ajouter gestion des erreurs

- LLM indisponible → fallback cosine
- Données incomplètes → logging warning, continuer
- Aucun résultat trouvé → retourner liste vide avec metadata
- Researcher ID inconnu → lever une erreur claire

### 3.4 Ajouter logging

Remplacer tous les `print()` par `logging` structuré avec niveaux (DEBUG, INFO, WARNING, ERROR).

### 3.5 Réécrire `main.py` comme client de l'agent

```python
agent = RecommendationAgent(config)
agent.load_data(researchers_path, opportunities_path)
agent.build_index()

for researcher in researchers:
    result = agent.query(researcher.id)
    print(result)
```

### Livrable

Une classe `RecommendationAgent` utilisable programmatiquement, avec gestion d'erreurs et logging.

---

## Étape 4 — Indexation scalable avec FAISS

**Objectif** : remplacer le numpy par FAISS pour supporter 80K+ vecteurs, avec sérialisation.

### 4.1 Créer `src/vector_store/faiss_vector_index.py`

Remplacer le `VectorIndex` numpy par un vrai index FAISS :

```python
import faiss
import numpy as np
import json
from pathlib import Path

class FAISSVectorIndex:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine si normalisé)
        self._ids: list[str] = []
        self._metadata: list[dict] = []
        self._dimension = dimension

    def build(self, index_records: list[dict]) -> None:
        vectors = np.stack([rec["vector"] for rec in index_records]).astype(np.float32)
        faiss.normalize_L2(vectors)  # normaliser pour cosine via IP
        self.index.add(vectors)
        self._ids = [rec["opportunity_id"] for rec in index_records]
        self._metadata = [rec.get("metadata", {}) for rec in index_records]

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        query = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "opportunity_id": self._ids[idx],
                "score": float(score),
                **self._metadata[idx],
            })
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "opportunities.faiss"))
        with open(directory / "metadata.json", "w") as f:
            json.dump({"ids": self._ids, "metadata": self._metadata}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorIndex":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "opportunities.faiss"))
        with open(directory / "metadata.json") as f:
            data = json.load(f)
        instance = cls(dimension=index.d)
        instance.index = index
        instance._ids = data["ids"]
        instance._metadata = data["metadata"]
        return instance
```

### 4.2 Mapping ID ↔ vecteurs

Le mapping est déjà géré dans la structure actuelle (`opportunity_id` dans chaque record). Avec FAISS, on garde exactement la même logique :

- Position dans l'index FAISS → `_ids[position]` → opportunity_id
- `metadata.json` contient la liste parallèle `_ids` et `_metadata`

### 4.3 Fichier de stockage

```
vector_store/
├── opportunities.faiss    # Index FAISS binaire
└── metadata.json          # IDs + metadata parallèles
```

### 4.4 Modifier `OpportunityRanker`

Adapter `src/matching/ranker.py` pour accepter un `FAISSVectorIndex` au lieu de `index_records` :

```python
class OpportunityRanker:
    def rank(self, researcher_vector, vector_index: FAISSVectorIndex, top_k=None):
        return vector_index.search(researcher_vector, top_k=top_k or 10)
```

### 4.5 Modifier l'agent

Dans `RecommendationAgent.build_index()` :

```python
def build_index(self, persist_dir: str | Path = None):
    index_records = build_opportunity_index(...)
    self.vector_index = FAISSVectorIndex(dimension=embedder.embedding_dim)
    self.vector_index.build(index_records)
    if persist_dir:
        self.vector_index.save(persist_dir)
```

### 4.6 Tests de scalabilité

- Indexer les ~4000 opportunités de `data/processed/opportunities.json`.
- Tester la recherche avec 1, 10, 100 requêtes.
- Comparer les temps avec l'ancien numpy.
- Vérifier la cohérence des résultats (les top-K doivent être identiques ou très proches).

### Livrable

Un index FAISS persisté, avec save/load, et un `OpportunityRanker` qui l'utilise.

---

## Étape 5 — Dashboard Streamlit + démonstration

**Objectif** : rendre l'agent utilisable via une interface web interactive.

### 5.1 Structure du dashboard

```
dashboard/
├── app.py                  # Point d'entrée Streamlit
├── pages/
│   ├── 1_recherche.py      # Page de recherche par chercheur
│   ├── 2_exploration.py    # Exploration des opportunités indexées
│   └── 3_evaluation.py     # Résultats d'évaluation
├── components/
│   ├── sidebar.py          # Configuration sidebar
│   └── cards.py            # Composants d'affichage
└── requirements.txt        # dépendances dashboard
```

### 5.2 Page de recherche (`1_recherche.py`)

- Sélection d'un chercheur dans une sidebar (dropdown ou recherche textuelle).
- Affichage du profil du chercheur (nom, institution, domaines, skills).
- Bouton "Rechercher des opportunités".
- Résultats affichés en cards :
  - Titre de l'opportunité
  - Organisation
  - Score (barre de progression colorée)
  - Raison du matching
  - Domaines correspondants
  - Lien vers la source (URL)
- Comparaison côte à côte : Cosine top-3 vs Gemini reranked vs Qwen reranked.

### 5.3 Page d'exploration (`2_exploration.py`)

- Nombre total d'opportunités indexées.
- Répartition par source (CORDIS, Funding & Tenders).
- Répartition par type.
- Recherche textuelle dans les opportunités.
- Filtres : par source, par type, par deadline.

### 5.4 Page d'évaluation (`3_exploration.py`)

- Lancer un benchmark depuis l'interface.
- Afficher les métriques : Recall@K, MRR, NDCG.
- Comparaison Gemini vs Qwen.
- Graphiques avec `st.chart` ou `plotly`.

### 5.5 Composants sidebar

- Configuration des paramètres :
  - Top-K retrieval (slider)
  - Top-K final (slider)
  - Embedding model (dropdown)
  - LLM providers (checkboxes)
  - Fallback activé (toggle)

### 5.6 Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

### Livrable

Un dashboard Streamlit fonctionnel permettant de :
1. Sélectionner un chercheur et obtenir des recommandations.
2. Explorer les opportunités indexées.
3. Visualiser les résultats d'évaluation.

---

## Ordre d'exécution

| Étape | Priorité | Temps estimé | Dépend de |
|---|---|---|---|
| 1. Pipeline retrieval | Haute | 1-2 jours | Rien |
| 2. Reranking LLM | Haute | 1 jour | Étape 1 |
| 3. Agent orchestration | Haute | 1-2 jours | Étape 1 + 2 |
| 4. FAISS indexation | Moyenne | 1-2 jours | Étape 3 |
| 5. Dashboard Streamlit | Moyenne | 2-3 jours | Étape 3 + 4 |

**Total estimé : 6-10 jours**

## Fichiers à modifier/créer

### Modifier
- `main.py` → refactoriser comme client de l'agent
- `src/matching/ranker.py` → adapter pour FAISSVectorIndex
- `src/vector_store/faiss_index.py` → remplacer par FAISS réel
- `src/reranking/llm_reranker.py` → améliorer parsing + retour typed
- `src/evaluation/evaluator.py` → adapter pour utiliser l'agent
- `run_evaluation.py` → refactoriser avec l'agent

### Créer
- `requirements.txt`
- `src/config.py` (ou `config.yaml`)
- `src/__init__.py` + tous les `__init__.py` manquants
- `src/agent/__init__.py`
- `src/agent/recommendation_agent.py`
- `src/agent/models.py` (RecommendationResult, RecommendationItem)
- `src/vector_store/faiss_vector_index.py` (nouveau, remplace l'ancien)
- `dashboard/app.py`
- `dashboard/pages/1_recherche.py`
- `dashboard/pages/2_exploration.py`
- `dashboard/pages/3_evaluation.py`
- `dashboard/components/sidebar.py`
- `dashboard/components/cards.py`

### Supprimer / archiver
- `src/matching/similarity.py` (redondant avec le ranker)
- `src/data/` (dossier vide, données dans `data/`)
- `data/loader.py` (dossier vide, doublon de `src/loaders/`)
