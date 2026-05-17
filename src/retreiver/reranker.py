from .retreiver import TransformerRetreiver, CombinedRetreiver
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
from typing import Union

import time

# Seed used for negative sampling, group shuffling, and the train/val split so
# repeat training runs produce the same fine-tuning data.
RANDOM_SEED = 42

class DistilBertReranker:
    def __init__(self,
                evidence: dict,
                model_name: str = "distilbert-base-uncased",
                device: str = None,
                max_length: int = 256,
                retreiver: Union[TransformerRetreiver, CombinedRetreiver, None] = None,
                negative_sample_size: int = 8,
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
            per_device_train_batch_size=2, # batch size for training
            per_device_eval_batch_size=2, # batch size for validation
            num_train_epochs=2, # number of training epochs
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
        Start the fine-tuning process. Currently fine-tuning is "full fine-tuning" of the model, i.e. we do not freeze the model.
        """
        save_path = OUTPUT_PATH / "reranker" / "cross_encoder_ranker"

        def _train():
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=1,
                dropout=0.1,
                attention_dropout=0.1
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            train_dataset, val_dataset = self._load_data_for_finetune()

            trainer = MultiNegativeRankingTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            print("----------------------------------------------------------")
            print("Starting the training of the model.")
            print("----------------------------------------------------------")
            start_time = time.time()

            trainer.train()

            print("----------------------------------------------------------")
            print("Training completed.")
            print(f'time taken: {time.time() - start_time} seconds')
            print("----------------------------------------------------------")

            # Trainer.save_model() creates the destination directory, so no manual mkdir is needed.
            trainer.save_model(save_path)
            trainer.save_state()
            self.tokenizer.save_pretrained(save_path)
            print(f"Saved model and tokenizer to {save_path}")

        def _load():
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(save_path)
                self.tokenizer = AutoTokenizer.from_pretrained(save_path)
                print(f"Loaded model and tokenizer from {save_path}")
            except Exception as e:
                print(f"Error loading model and tokenizer: {e}")
                print(f"No model and tokenizer found at {save_path}. Training from scratch.")
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

        Returns a list of scores aligned 1:1 with ``evidence_ids``. Any id missing
        from ``self.evidence`` is given ``-inf`` so it ranks last without breaking
        alignment with the caller's id list.

        args:
            claim_text (str): The text of the claim for which we want to score the evidence pairs.
            evidence_ids (list): The list of evidence IDs to score.
            batch_size (int): The batch size for the scoring.
        returns:
            List of scores for the evidence pairs (higher score means more relevant).
        """
        # Inference uses dynamic padding (`padding=True`) for speed; the training
        # path uses `padding="max_length"` because `default_data_collator` needs a
        # fixed shape to stack candidates across the batch axis.
        scores: list[float] = [float("-inf")] * len(evidence_ids)

        for i in range(0, len(evidence_ids), batch_size):
            # get the batch of evidence ids
            batch_ids: list[str] = evidence_ids[i:i + batch_size]

            # get the valid local indices and evidence
            valid_local_indices: list[int] = []
            valid_evidence: list[str] = []

            # filter out the evidence ids that are not in the evidence dictionary
            for j, eid in enumerate(batch_ids):
                if eid in self.evidence:
                    valid_local_indices.append(j)
                    valid_evidence.append(self.evidence[eid])

            if not valid_evidence:
                continue

            # encode the claim text and the valid evidence
            encoded = self.tokenizer(
                [claim_text] * len(valid_evidence),
                valid_evidence,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # move the encoded tensors to the device
            encoded = {
                k: v.to(self.device)
                for k, v in encoded.items()
            }

            # inference
            with torch.no_grad():
                outputs = self.model(**encoded)

            # AutoModelForSequenceClassification with num_labels=1 -> logits (n, 1).
            relevant_scores = outputs.logits.squeeze(-1).detach().cpu().tolist()

            # handle edge case where the relevant scores is a 0-d tensor
            if not isinstance(relevant_scores, list):  # 0-d tensor when n==1 in some configs
                relevant_scores = [relevant_scores]

            # put the scores back into the scores list to maintain the alignment with the evidence ids (since we filtered out the evidence ids that are not in the evidence dictionary)
            for local_idx, score in zip(valid_local_indices, relevant_scores):
                scores[i + local_idx] = score

        return scores

    def _build_ranker_groups(
        self,
        claims: dict,
        evidence: dict,
        num_negatives_per_positive: int = 8
    ) -> list[dict]:

        groups: list[dict] = []
        # Local RNG keeps negative sampling reproducible without leaking state into the global random module
        rng: random.Random = random.Random(RANDOM_SEED)

        for claim_id, claim_data in tqdm(claims.items(), desc="Building ranking groups"):
            claim_text = claim_data["claim_text"]

            gold_ids = [
                eid for eid in claim_data["evidences"]
                if eid in evidence
            ]

            if len(gold_ids) == 0:
                continue

            gold_set = set(gold_ids)

            retrieved_ids, _ = self.retreiver.retreive(claim_text)

            negative_ids = [
                eid for eid in retrieved_ids
                if eid not in gold_set and eid in evidence
            ]

            if len(negative_ids) < num_negatives_per_positive:
                continue

            for pos_id in gold_ids:
                sampled_negatives = rng.sample(negative_ids, num_negatives_per_positive)

                groups.append({
                    "claim": claim_text,
                    "positive_evidence": evidence[pos_id],
                    "negative_evidences": [
                        evidence[neg_id]
                        for neg_id in sampled_negatives
                    ],
                })

        return groups

    _TRAIN_GROUPS_CACHE = OUTPUT_PATH / "temp_cache" / "train_groups.pt"
    _VAL_GROUPS_CACHE = OUTPUT_PATH / "temp_cache" / "val_groups.pt"

    def _load_data_for_finetune(self):
        """
        Load the data for finetuning.
        returns:
            train_dataset (MultiNegativeRankerDataset): The train dataset.
            val_dataset (MultiNegativeRankerDataset): The validation dataset.
        """

        try:
            train_groups, val_groups = self._load_groups()
            print("Loaded train and val groups from cache.")
        except FileNotFoundError:
            print("No cached groups found. Building from scratch.")
            train_claims, _, _ = load_data()

            claim_ids = list(train_claims.keys())

            train_claim_ids, val_claim_ids = train_test_split(
                claim_ids,
                test_size=0.15,
                random_state=RANDOM_SEED,
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

            train_groups = self._build_ranker_groups(
                train_claims_split,
                self.evidence,
                num_negatives_per_positive=self.negative_sample_size
            )

            val_groups = self._build_ranker_groups(
                val_claims_split,
                self.evidence,
                num_negatives_per_positive=self.negative_sample_size
            )

            shuffle_rng = random.Random(RANDOM_SEED)
            shuffle_rng.shuffle(train_groups)
            shuffle_rng.shuffle(val_groups)

            self._save_groups(train_groups, val_groups)

        train_dataset = MultiNegativeRankerDataset(
            train_groups,
            self.tokenizer,
            self.max_length
        )

        val_dataset = MultiNegativeRankerDataset(
            val_groups,
            self.tokenizer,
            self.max_length
        )

        return train_dataset, val_dataset

    def _save_groups(self, train_groups: list[dict], val_groups: list[dict]):
        cache_dir = self._TRAIN_GROUPS_CACHE.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train_groups, self._TRAIN_GROUPS_CACHE)
        print(f"Saved train groups to {self._TRAIN_GROUPS_CACHE}")

        torch.save(val_groups, self._VAL_GROUPS_CACHE)
        print(f"Saved val groups to {self._VAL_GROUPS_CACHE}")

    def _load_groups(self) -> tuple[list[dict], list[dict]]:
        if not self._TRAIN_GROUPS_CACHE.exists() or not self._VAL_GROUPS_CACHE.exists():
            raise FileNotFoundError(
                f"Cached groups not found under {self._TRAIN_GROUPS_CACHE.parent}"
            )

        train_groups = torch.load(self._TRAIN_GROUPS_CACHE, weights_only=False)
        print(f"Loaded train groups from {self._TRAIN_GROUPS_CACHE}")

        val_groups = torch.load(self._VAL_GROUPS_CACHE, weights_only=False)
        print(f"Loaded val groups from {self._VAL_GROUPS_CACHE}")

        return train_groups, val_groups

class MultiNegativeRankerDataset(Dataset):
    def __init__(self, groups: list[dict], tokenizer: AutoTokenizer, max_length: int = 256):
        # Each group is {"claim": str, "positive_evidence": str, "negative_evidences": list[str]}.
        self.groups: list[dict] = groups
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        item: dict = self.groups[idx]

        # The positive must be at index 0 so the cross-entropy target below is correct.
        evidences: list[str] = [item["positive_evidence"]] + item["negative_evidences"]
        claims: list[str] = [item["claim"]] * len(evidences)

        encoded = self.tokenizer(
            claims,
            evidences,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],          # (num_candidates, max_length)
            "attention_mask": encoded["attention_mask"],  # (num_candidates, max_length)
            # Cross-entropy target: index of the positive among the candidate row.
            # The positive is always placed first, so this is always 0.
            "labels": torch.tensor(0, dtype=torch.long),
        }

class MultiNegativeRankingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]                   # (batch_size, num_candidates, seq_len)
        attention_mask = inputs["attention_mask"]         # (batch_size, num_candidates, seq_len)
        labels = inputs["labels"]                         # (batch_size,) — index of the positive among candidates (always 0 here)

        batch_size, num_candidates, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)

        outputs = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        # AutoModelForSequenceClassification with num_labels=1 returns logits of shape (N, 1).
        scores = outputs.logits.squeeze(-1)               # (batch_size * num_candidates,)
        scores = scores.view(batch_size, num_candidates)  # (batch_size, num_candidates)

        # Multiple-negative ranking loss: softmax across candidates with the positive at index 0.
        loss = torch.nn.functional.cross_entropy(scores, labels)

        if return_outputs:
            return loss, {"scores": scores}

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss = self.compute_loss(model, inputs)

        return (loss.detach(), None, None)