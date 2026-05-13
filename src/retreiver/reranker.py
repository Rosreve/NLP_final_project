from retreiver import TransformerRetreiver
from torch.utils.data import Dataset
import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig
)

from sklearn.model_selection import train_test_split
import numpy as np

from util import load_data
from config import OUTPUT_PATH

from joblib import Parallel, delayed

class DistilBertReranker:
    def __init__(self,
                evidence: dict,
                model_name: str = "distilbert-base-uncased",
                device: str = None,
                max_length: int = 256,
                retreiver: TransformerRetreiver = None,
                top_k: int = 3):

        self.evidence = evidence    
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.retreiver = retreiver
        self.top_k = top_k

        # prepare the training arguments and the trainer
        self.training_args = TrainingArguments(
            output_dir= OUTPUT_PATH / "reranker" / "cross_encoder_ranker",
            eval_strategy="epoch", # evaluate the model at the end of each epoch
            save_strategy="epoch", # save the model at the end of each epoch
            learning_rate=2e-5, # learning rate for the optimizer
            per_device_train_batch_size=16, # batch size for training
            per_device_eval_batch_size=16, # batch size for validation
            num_train_epochs=3, # number of training epochs
            weight_decay=0.01, # weight decay for the optimizer
            warmup_ratio=0.1, # warmup ratio for the optimizer
            logging_steps=50, # log the training and evaluation metrics every 50 steps
            load_best_model_at_end=True, # load the best model at the end of the training
            metric_for_best_model="eval_loss", # use the evaluation loss as the metric for the best model
            report_to="none" # don't report the training and evaluation metrics to the console
        )

        self.model = None
        self.tokenizer = None
        self._start_fine_tuning(retrain=True)

    def _start_fine_tuning(self, retrain: bool = False):
        """
        Start the fine-tuning process. Currenty fine-tuning is "full fine-tuning" of the model, i.e. we do not freeze the model. 
        """
        def _train():
            # load the model and tokenizer
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=2,
                dropout=0.1,
                attention_dropout=0.1
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # load the data for finetuning
            train_dataset, val_dataset = self._load_data_for_finetune()

            # prepare the trainer
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
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
            print("Retraining the model from scratch.")
            _train()
        else:
            print("Loading the model and tokenizer from the cache.")
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
        """
        Rerank a list of evidence IDs for a given claim text.
        args:
            claim_text (str): The text of the claim for which we want to rerank the evidence IDs.
            candidate_ids (list): The list of evidence IDs to rerank.
            top_k (int): The number of top evidence IDs to return.
        returns:
            List of reranked evidence IDs.
        """
        # score the evidence pairs
        scores: list[float] = self._score_pairs(claim_text, candidate_ids)

        # sort the evidence pairs by the scores
        ranked: list[tuple[str, float]] = sorted(
            zip(candidate_ids, scores),
            key=lambda x: x[1],
            reverse=True # sort in descending order of scores
        )
        
        # return the top k evidence IDs
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
                probs = torch.softmax(outputs.logits, dim=1)

            # get the scores for the relevant evidence pairs
            relevant_scores: np.ndarray = probs[:, 1].detach().cpu().numpy()
            
            # extend the scores list with the relevant scores
            scores.extend(relevant_scores.tolist())

        return scores

    def _build_ranker_pairs(
        self,
        claims: dict,
        evidence: dict,
        num_negatives: int = 200
    ):
        pairs = []

        for claim_id, claim_data in tqdm(claims.items(), desc="Building ranker pairs for finetuning"):
            claim_text = claim_data["claim_text"]
            gold_ids = set(claim_data["evidences"])

            # positive examples
            for eid in gold_ids:
                if eid in evidence:
                    pairs.append({
                        "claim": claim_text,
                        "evidence": evidence[eid],
                        "label": 1
                    })

            # hard negative examples from retreiver
            retreived_ids, _ = self.retreiver.retreive(claim_text, top_k=340)

            # get the negative evidence ids
            # we sample the top consine similarity instances that are not in the gold set, as the "hard" negative examples
            # in hope that algorithm to learn better than pure random sampling
            negative_ids = [
                eid for eid in retreived_ids
                if eid not in gold_ids
            ]

            negative_ids = negative_ids[:num_negatives]

            for eid in negative_ids:
                pairs.append({
                    "claim": claim_text,
                    "evidence": evidence[eid],
                    "label": 0
                })

        return pairs

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

            # build the ranker pairs
            train_pairs: list[dict] = self._build_ranker_pairs(
                train_claims,
                self.evidence,
                num_negatives=200
            )

            # split the data into train and dev
            train_pairs, val_pairs = train_test_split(
                train_pairs, # split the train pairs into train and val
                test_size=0.15,  # 15% of the data for validation
                random_state=42, # set the random state for reproducibility
                stratify=[p["label"] for p in train_pairs] # stratify the data by the label to deal with class imbalance
            )

            print(f"Number of train pairs: {len(train_pairs)}")
            print(f"Number of val pairs: {len(val_pairs)}")

            # prepare the datasets
            train_dataset: RankerDataset = RankerDataset(train_pairs, self.tokenizer, self.max_length) # train dataset
            val_dataset: RankerDataset = RankerDataset(val_pairs, self.tokenizer, self.max_length) # validation dataset

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

class RankerDataset(Dataset):
    # dataset for the ranker model
    def __init__(self, pairs: list, tokenizer: AutoTokenizer, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        # tokenize the claim and evidence
        encoded: dict = self.tokenizer(
            item["claim"], # claim text
            item["evidence"], # evidence text
            truncation=True, # truncate the text to the max length
            padding="max_length", # pad the text to the max length
            max_length=self.max_length, # set the max length
            return_tensors="pt" # return the tensors 
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0), # squeeze the tensor to reduce the batch dimension
            # attension mask is used to mask the padding tokens to tell model which tokens are padded
            "attention_mask": encoded["attention_mask"].squeeze(0), # squeeze the tensor to reduce the batch dimension
            "labels": torch.tensor(item["label"], dtype=torch.long) # convert the label to a tensor for the loss function
        }
 