# Claim verification pipeline

End-to-end project for **evidence retrieval** and **claim classification**: dense retrieval with optional cross-encoder reranking, followed by a TF–IDF + logistic regression classifier. Outputs JSON predictions and retrieval artifacts under `output/` for scoring with `eval.py`.

## Project layout

| Path | Role |
| ------ | ------ |
| `run.py` | Main entry: retrieval (and optionally reranking) → classifier → artifacts |
| `eval.py` | Compares predictions to ground-truth labels and retrieved evidence IDs |
| `util.py` | Loads `data/*.json`; loads retrieval results from `output/` for the classifier |
| `src/config.py` | Paths for data and `output/` (resolved from repo root) |
| `src/classifier.py` | `TfidfClaimClassifier`: claim + `[SEP]` + retrieved evidence texts |
| `src/retreiver/retreiver.py` | `TransformerRetreiver` (embeddings), `TfidfRetreiver`, `BM25Retreiver` |
| `src/retreiver/embedder.py` | `SentenceEmbedder` (`sentence-transformers`, default MiniLM) |
| `src/retreiver/reranker.py` | `DistilBertReranker`: DistilBERT cross-encoder reranking |

## Data

Place these JSON files under `data/` (see `src/config.py`):

- `train-claims.json` — training claims with labels and gold evidence IDs  
- `dev-claims.json` — development claims with labels and gold evidence IDs  
- `evidence.json` — map of evidence ID → passage text  

The classifier reads dev retrieval output from:

`output/retreived_data/dev-claims-transformer-retreival-reranked.json`

(as set in `util.load_retrieval_results()`). Run `run.py` first so this file exists.

## Environment

Use Python **3.10+** (the repo may include a local `.venv`). Install typical dependencies:

- `numpy`, `scikit-learn`, `tqdm`, `matplotlib`
- `sentence-transformers` (for dense retrieval)
- `torch`, `transformers`, `datasets` (for reranker training/inference — if reranking enabled)

Install from your environment manager of choice, for example:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy scikit-learn tqdm matplotlib sentence-transformers torch transformers accelerate
```

## Running the pipeline

From the **repository root** (`project_implementation/`):

```bash
python run.py --retreiver-top-k 200 --reranker-top-k 50
```

Important: `run.py` currently calls `main(include_reranking=False)` at the bottom, so **DistilBERT reranking may be skipped** even though the reranker is wired inside `main()`. Set `include_reranking=True` in `run.py` if you want reranking on the full path.

Artifacts written include:

- `output/retreived_data/dev-claims-transformer-retreival-reranked.json` — per-claim retrieval (and rerank) output consumed by the classifier  
- `output/predictions.json` — predicted `claim_label` and `evidences` per dev claim  
- `output/dev-groundtruth.json` — copy of `data/dev-claims.json` for evaluation convenience  
- Embedding cache (when using `TransformerRetreiver`): under `output/temp_cache/` as configured in `retreiver.py`

First dense retrieval pass may download embedding weights and optionally compute embeddings for all evidence (can be heavy for large `evidence.json`).

### Optional lexical retrievers

`TfidfRetreiver` and `BM25Retreiver` in `src/retreiver/retreiver.py` follow the same `fit` / `retrieve` shape as TF–IDF; `run.py` contains commented examples for TF–IDF-only retrieval paths you can swap in for experiments.

## Evaluation

After `run.py` finishes, from the repo root:

```bash
python eval.py --predictions output/predictions.json --groundtruth output/dev-groundtruth.json
```

Optional per-claim debug:

```bash
python eval.py --predictions output/predictions.json --groundtruth output/dev-groundtruth.json --verbose
```

The script reports mean evidence retrieval precision, recall, F-score over the retrieved set versus gold evidence IDs, claim classification accuracy, and a harmonic mean of F and accuracy.

## Note on naming

Directories and identifiers use **`retreiver`** (alternate spelling of “retriever”) consistently in this codebase; imports expect that package name (`from retreiver import ...`) with `SRC_ROOT = src` injected in `run.py`.
