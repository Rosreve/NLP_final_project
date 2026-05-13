from pathlib import Path

# Resolve paths from the project root so imports work from notebooks/scripts anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

OUTPUT_PATH = PROJECT_ROOT / "output"

TRAIN_CLAIMS_PATH = DATA_PATH / "train-claims.json"
DEV_CLAIMS_PATH = DATA_PATH / "dev-claims.json"
EVIDENCE_PATH = DATA_PATH / "evidence.json"


# OUTPUT retreiver file name
BM25_RETRIEVAL_FILENAME = "dev-claims-bm25-retreival.json"

TFIDF_RETRIEVAL_FILENAME = "dev-claims-tfidf-retreival.json"

TRANSFORMER_RETRIEVAL_FILENAME = "dev-claims-transformer-retreival-reranked.json"

COMBINED_RETRIEVAL_FILENAME = "dev-claims-combined-retreival.json"