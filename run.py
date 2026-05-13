"""
End-to-end pipeline: TF-IDF retrieval -> TF-IDF + LR claim classification.
Writes artifacts under output/ so eval.py can be run with --predictions and --groundtruth.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import (DEV_CLAIMS_PATH, 
                    OUTPUT_PATH,
                    BM25_RETRIEVAL_FILENAME,
                    TFIDF_RETRIEVAL_FILENAME,
                    TRANSFORMER_RETRIEVAL_FILENAME,
                    COMBINED_RETRIEVAL_FILENAME)

from classifier import TfidfClaimClassifier

from util import load_data
 
from retreiver import (TransformerRetreiver, DistilBertReranker, BM25Retreiver, TfidfCharRetreiver, CombinedRetreiver)
 
def main():
    parser = argparse.ArgumentParser(description="Run retrieval + classification pipeline.")
    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=50,
        help="Number of evidence passages to retrieve per claim (eval uses retrieved set as a set).",
    )
    parser.add_argument(
        "--retreiver-top-k",
        type=int,
        default=340,
        help="Number of evidence passages to retrieve per claim.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha value for combined retreiver.",
    ) # best 0.5
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Beta value for combined retreiver.",
    ) # best 0.3
    parser.add_argument(
        "--include_reranking",
        type=bool,
        default=True,
        help="Include reranking step.",
    )
    args = parser.parse_args()

    gamma = 1.0 - args.alpha - args.beta
    if gamma < 0:
        parser.error(
            "alpha + beta must be at most 1.0 (gamma = 1 - alpha - beta must be non-negative)."
        )

    # ============================================
    # Step 1: Retrieve evidence for each claim
    # ============================================
    train_claims, dev_claims, evidence = load_data()

    retreiver = CombinedRetreiver(
        top_k=args.retreiver_top_k, alpha=args.alpha, beta=args.beta, gamma=gamma
    )
    retreiver.fit(evidence)

    retreived_data: dict = retreiver.retreive_all(dev_claims, visual_sim=False)

    if args.include_reranking:
        ranker = DistilBertReranker(evidence=evidence, retreiver=retreiver, top_k=args.reranker_top_k)
        reranked_data: dict = ranker.rerank_all(retreived_data)
    else:
        reranked_data = retreived_data

    if args.include_reranking:
        file_name = f"{COMBINED_RETRIEVAL_FILENAME.replace('.json', '_reranked.json')}"
    else:
        file_name = COMBINED_RETRIEVAL_FILENAME
    output_file = OUTPUT_PATH / "retreived_data" / file_name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(reranked_data, f, indent=2)

    print(f"Saved reranked data to {output_file}") 
    # ============================================
    # Step 2: Classify each claim
    # ============================================

    classifier = TfidfClaimClassifier(retrieval_filename=file_name)
    classifier.train_model()
    classifier.evaluate()
    predictions_path = OUTPUT_PATH / "predictions.json"
    groundtruth_path = OUTPUT_PATH / "dev-groundtruth.json"
    output_file_name = file_name.replace(".json", "_predictions.json")
    shutil.copyfile(DEV_CLAIMS_PATH, groundtruth_path)
    print(f"Wrote ground truth copy to {groundtruth_path}")

    print()
    print("Evaluate with:")
    print(
        f'  python eval.py --predictions "{predictions_path}" '
        f'--groundtruth "{groundtruth_path}"'
        f' --output_filename "{output_file_name}"'
    )


if __name__ == "__main__":
    main()
