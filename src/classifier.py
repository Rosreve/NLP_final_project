import json

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from util import load_data

from config import OUTPUT_PATH

from tqdm import tqdm

class TfidfClaimClassifier:
    def __init__(self, retrieval_filename: str ='dev-claims-bm25-retreival.json'):
        self.train_claims, _, self.evidence = load_data()
        self.dev_claims = self._load_retrieval_results(retrieval_filename)


        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50000
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

    def _build_classification_data(self, claim):
        claim_text: str = claim.get("claim_text", "")
        claim_label: str = claim.get("claim_label", None)

        evidence_texts = [
            self.evidence[evidence_id]
            for evidence_id in claim.get("evidences", [])
            if evidence_id in self.evidence
        ]

        combined_text: str = claim_text + " [SEP] " + " ".join(evidence_texts)

        return combined_text, claim_label

    def _prepare_training_data(self):
        texts, labels = [], []

        for claim_data in tqdm(self.train_claims.values(), desc="Preparing training data"):
            text, label = self._build_classification_data(claim_data)
            if label is not None:
                texts.append(text)
                labels.append(label)

        return texts, labels

    def _prepare_dev_data(self):
        texts, labels = [], []

        for claim_data in tqdm(self.dev_claims.values(), desc="Preparing dev data"):
            text, label = self._build_classification_data(claim_data)
            if label is not None:
                texts.append(text)
                labels.append(label)

        return texts, labels

    def train_model(self):
        train_texts, train_labels = self._prepare_training_data()
        self.model.fit(train_texts, train_labels)

    def evaluate(self):
        # write the predictions into a new json file combined with the original dev claims data for error analysis
        predictions = {}
        for claim_id, claim_data in self.dev_claims.items():
            claim_text = claim_data.get("claim_text", "")
            evidence_ids = claim_data.get("evidences", [])
            pred_label = self.predict_one(claim_text, evidence_ids)
            predictions[claim_id] = {
                "claim_label": pred_label,
                "evidences": list(evidence_ids),
            }

        predictions_path = OUTPUT_PATH / "predictions.json"
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Wrote eval predictions to {predictions_path}")
        return

    def predict_one(self, claim_text, evidence_ids):
        evidence_texts = [
            self.evidence[eid]
            for eid in evidence_ids
            if eid in self.evidence
        ]

        combined_text = claim_text + " [SEP] " + " ".join(evidence_texts)
        return self.model.predict([combined_text])[0]

    def _load_retrieval_results(self, filename: str ='dev-claims-bm25-retreival.json'):
        output_file = OUTPUT_PATH / "retreived_data" / filename
        with open(output_file, "r") as f:
            retrieval_results = json.load(f)
        return retrieval_results
    
if __name__ == "__main__":
    classifier = TfidfClaimClassifier()
    classifier.train_model()
    classifier.evaluate()