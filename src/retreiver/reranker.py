from retreiver import TransformerRetreiver
from torch.utils.data import Dataset
import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoConfig
)

from sklearn.model_selection import train_test_split
from transformers import Trainer

from util import load_data
from config import OUTPUT_PATH
import random

class DistilBertReranker:
    def __init__(self,
                evidence: dict,
                model_name: str = "distilbert-base-uncased",
                device: str = None,
                max_length: int = 256,
                retreiver: TransformerRetreiver = None,
                negative_sample_size: int = 1,
                top_k: int = 5,
                retrain: bool = False):

        self.evidence = evidence    
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.retreiver = retreiver
        self.negative_sample_size = negative_sample_size
        self.top_k = top_k

        # prepare the training arguments and the trainer
        self.training_args = TrainingArguments(
            output_dir= OUTPUT_PATH / "reranker" / "cross_encoder_ranker",
            eval_strategy="epoch", # evaluate the model at the end of each epoch
            save_strategy="epoch", # save the model at the end of each epoch
            learning_rate=2e-5, # learning rate for the optimizer
            per_device_train_batch_size=16, # batch size for training
            per_device_eval_batch_size=16, # batch size for validation
            num_train_epochs=1, # number of training epochs
            weight_decay=0.01, # weight decay for the optimizer
            warmup_ratio=0.1, # warmup ratio for the optimizer
            logging_steps=50, # log the training and evaluation metrics every 50 steps
            load_best_model_at_end=True, # load the best model at the end of the training
            metric_for_best_model="eval_loss", # use the evaluation loss as the metric for the best model
            report_to="none", # don't report the training and evaluation metrics to the console
            remove_unused_columns=False # don't remove unused columns from the dataset
        )

        self.model = None
        self.tokenizer = None
        self.retrain = retrain
        self._start_fine_tuning(retrain=self.retrain)

    def _start_fine_tuning(self, retrain):
        """
        Start the fine-tuning process. Currenty fine-tuning is "full fine-tuning" of the model, i.e. we do not freeze the model. 
        """
        def _train():
            # load the model and tokenizer
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=1,
                dropout=0.1,
                attention_dropout=0.1
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # load the data for finetuning
            train_dataset, val_dataset = self._load_data_for_finetune()

            # prepare the trainer
            trainer = PairwiseRankingTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            # start training
            trainer.train()

            # save the model and the tokenizer
            path = OUTPUT_PATH / "reranker" / "cross_encoder_ranker"
            path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_model(path)
            trainer.save_state()
            self.tokenizer.save_pretrained(path)
            print(f"Saved model and tokenizer to {path}")
            pass
            
        def _load():
            try:
                save_path = OUTPUT_PATH / "reranker" / "cross_encoder_ranker"

                self.model = AutoModelForSequenceClassification.from_pretrained(save_path)
                self.tokenizer = AutoTokenizer.from_pretrained(save_path)
                print(f"Loaded model and tokenizer from {OUTPUT_PATH / "reranker" / "cross_encoder_ranker"}")
            except Exception as e:
                print(f"Error loading model and tokenizer: {e}")
                print(f"No model and tokenizer found at {OUTPUT_PATH / "reranker" / "cross_encoder_ranker"}. Training from scratch.")
                _train()

        if retrain:
            print("----------------------------------------------------------")
            print("Retraining the model from scratch.")
            print("----------------------------------------------------------")
            _train()
        else:
            print("----------------------------------------------------------")
            print("Loading the model and tokenizer from the cache.")
            print("----------------------------------------------------------")
 
            _load()

        # set the device
        self.model.to(self.device)
        self.model.eval()

        return

    def rerank_all(self, claims: dict) -> dict:
        """
        Rerank all claims in the provided dictionary.
        args:
            claims (dict): The dictionary of claims to rerank.
        returns:
            Dictionary of reranked claims.
        """
        reranked_claims = {}
        for claim_id, claim_data in tqdm(claims.items(), desc="Reranking claims"):
            claim_text = claim_data["claim_text"]
            candidate_ids = claim_data["evidences"]
            reranked_claims[claim_id] = {
                **claim_data,
                "evidences": self.rerank(claim_text, candidate_ids),
            }
        return reranked_claims

    def rerank(self, claim_text: str, candidate_ids: list) -> list:
        scores = self._score_pairs(claim_text, candidate_ids)

        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [eid for eid, score in ranked[:self.top_k]]

    def _score_pairs(self, claim_text: str, evidence_ids: list, batch_size: int = 16) -> list:
        """
        Score a list of evidence pairs for a given claim text.
        args:
            claim_text (str): The text of the claim for which we want to score the evidence pairs.
            evidence_ids (list): The list of evidence IDs to score.
            batch_size (int): The batch size for the scoring.
        returns:
            List of scores for the evidence pairs (higher score means more relevant).
        """
        scores: list[float] = []

        for i in range(0, len(evidence_ids), batch_size):
            batch_ids: list[str] = evidence_ids[i:i + batch_size]
            batch_evidence: list[str] = [self.evidence[eid] for eid in batch_ids]

            # encode the claim and evidence pairs into tokens
            encoded = self.tokenizer(
                [claim_text] * len(batch_ids),
                batch_evidence,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # move the tokens to the device
            encoded = {
                k: v.to(self.device)
                for k, v in encoded.items()
            }

            # start inference
            with torch.no_grad():
                outputs = self.model(**encoded)

            # get the scores for the relevant evidence pairs
            relevant_scores = outputs.logits.squeeze(-1).detach().cpu().numpy()
            
            # extend the scores list with the relevant scores
            scores.extend(relevant_scores.tolist())

        return scores

    def _build_ranker_triplets(
        self,
        claims: dict,
        evidence: dict,
        num_negatives_per_positive: int = 1
    ):
        triplets = []

        for claim_id, claim_data in tqdm(claims.items(), desc="Building ranking triplets"):
            claim_text = claim_data["claim_text"]
            gold_ids = [
                eid for eid in claim_data["evidences"]
                if eid in evidence
            ]

            if len(gold_ids) == 0:
                continue

            retrieved_ids, _ = self.retreiver.retreive(claim_text, top_k=340)

            negative_ids = [
                eid for eid in retrieved_ids
                if eid not in set(gold_ids) and eid in evidence
            ]

            if len(negative_ids) == 0:
                continue

            for pos_id in gold_ids:
                sampled_negatives = random.sample(
                    negative_ids,
                    min(num_negatives_per_positive, len(negative_ids))
                )

                for neg_id in sampled_negatives:
                    triplets.append({
                        "claim": claim_text,
                        "positive_evidence": evidence[pos_id],
                        "negative_evidence": evidence[neg_id],
                    })

        return triplets

    def _load_data_for_finetune(self):
        """
        Load the data for finetuning.
        returns:
            train_dataset (RankerDataset): The train dataset.
            val_dataset (RankerDataset): The validation dataset.
            evidence (dict): The evidence data.
        """

        try:
            train_dataset, val_dataset = self.load_dataset()
            print("Loaded train and val datasets from cache.")
        except FileNotFoundError:
            print("No train and val datasets found in cache. Building from scratch.")
            train_claims, _, _ = load_data()

            claim_ids = list(train_claims.keys())

            train_claim_ids, val_claim_ids = train_test_split(
                claim_ids,
                test_size=0.15,
                random_state=42,
                shuffle=True
            )

            train_claims_split = {
                cid: train_claims[cid]
                for cid in train_claim_ids
            }

            val_claims_split = {
                cid: train_claims[cid]
                for cid in val_claim_ids
            }

            train_triplets = self._build_ranker_triplets(
                train_claims_split,
                self.evidence,
                num_negatives_per_positive=self.negative_sample_size
            )

            val_triplets = self._build_ranker_triplets(
                val_claims_split,
                self.evidence,
                num_negatives_per_positive=self.negative_sample_size
            )

            random.shuffle(train_triplets)
            random.shuffle(val_triplets)

            train_dataset = PairwiseRankerDataset(
                train_triplets,
                self.tokenizer,
                self.max_length
            )

            val_dataset = PairwiseRankerDataset(
                val_triplets,
                self.tokenizer,
                self.max_length
            )
            self.save_dataset(train_dataset, val_dataset)

        return train_dataset, val_dataset

    def save_dataset(self, train_dataset: list, val_dataset: list):
        path = OUTPUT_PATH / "temp_cache" / "train_dataset.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(train_dataset, path)
        print(f"Saved train dataset to {path}")
        path = OUTPUT_PATH / "temp_cache" / "val_dataset.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(val_dataset, path)
        print(f"Saved val dataset to {path}")

    def load_dataset(self):
        # Cached datasets are torch pickles (RankerDataset objects), not pure tensor checkpoints.
        path = OUTPUT_PATH / "temp_cache" / "train_dataset.pt"
        train_dataset = torch.load(path, weights_only=False)
        print(f"Loaded train dataset from {path}")
        path = OUTPUT_PATH / "temp_cache" / "val_dataset.pt"
        val_dataset = torch.load(path, weights_only=False)
        print(f"Loaded val dataset from {path}")
        return train_dataset, val_dataset

class PairwiseRankerDataset(Dataset):
    def __init__(self, triplets: list, tokenizer: AutoTokenizer, max_length: int = 256):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]

        pos_encoded = self.tokenizer(
            item["claim"],
            item["positive_evidence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        neg_encoded = self.tokenizer(
            item["claim"],
            item["negative_evidence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pos_input_ids": pos_encoded["input_ids"].squeeze(0),
            "pos_attention_mask": pos_encoded["attention_mask"].squeeze(0),
            "neg_input_ids": neg_encoded["input_ids"].squeeze(0),
            "neg_attention_mask": neg_encoded["attention_mask"].squeeze(0),
        }
 
class PairwiseRankingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_inputs = {
            "input_ids": inputs["pos_input_ids"],
            "attention_mask": inputs["pos_attention_mask"],
        }

        neg_inputs = {
            "input_ids": inputs["neg_input_ids"],
            "attention_mask": inputs["neg_attention_mask"],
        }

        pos_scores = model(**pos_inputs).logits.squeeze(-1)
        neg_scores = model(**neg_inputs).logits.squeeze(-1)

        margin = 1.0
        loss = torch.relu(margin + neg_scores - pos_scores).mean()

        if return_outputs:
            return loss, {
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
            }

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None
    ):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)

        return (loss, None, None)