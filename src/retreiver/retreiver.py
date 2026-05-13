from typing import Any


import json
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from util import load_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import OUTPUT_PATH

from retreiver.embedder import SentenceEmbedder

import pickle
import os

from nltk.stem import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()

def lemma_tokenizer(text: str):
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    return [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

def _plot_similarity_scatter(similarities: dict, title: str | None = None, ylabel: str | None = None):
    # flatten similarities into x/y points
    x = []
    y = []

    for claim_idx, (claim_id, sims) in enumerate(similarities.items()):
        for sim in sims:
            x.append(claim_idx)   # or claim_id if numeric
            y.append(sim)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6)

    plt.title(title or "Cosine Similarities for Retreived Evidence")
    plt.xlabel("Claim Index")
    plt.ylabel(ylabel or "Cosine Similarity")

    # hide x-axis ticks for better visualization
    plt.xticks([])

    plt.grid(alpha=0.3)
    plt.show()

class CombinedRetreiver:
    def __init__(self, top_k=5, alpha=0.3, beta=0.3, gamma=0.4):
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._normalize_weights()

        self.transformer_retreiver = TransformerRetreiver(top_k=top_k*2)
        self.bm25_retreiver = BM25Retreiver(top_k=top_k*2)
        self.tfidf_char_retreiver = TfidfCharRetreiver(top_k=top_k*2)

    def fit(self, evidence):
        try:
            self._load_retreiver()
            print("Loaded existing retreiver. Skipping fit.")
        except FileNotFoundError:
            print("No existing retreiver found. Fitting new retreiver...")
            self.transformer_retreiver.fit(evidence)
            self.bm25_retreiver.fit(evidence)
            self.tfidf_char_retreiver.fit(evidence)

            # temp cache the retreiver as pickle
            self._save_retreiver() 
            print("Retreiver fitted and saved.")
    
    def _save_retreiver(self, path= OUTPUT_PATH / "temp_cache" / "combined_retreiver.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "transformer_retreiver": self.transformer_retreiver,
                "bm25_retreiver": self.bm25_retreiver,
                "tfidf_char_retreiver": self.tfidf_char_retreiver
            }, f)

        print(f"Saved retreiver to {path}")

    def _load_retreiver(self, path= OUTPUT_PATH / "temp_cache" / "combined_retreiver.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Retreiver file not found at {path}. Please run fit() and save_retreiver() first.")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.transformer_retreiver = data["transformer_retreiver"]
            self.bm25_retreiver = data["bm25_retreiver"]
            self.tfidf_char_retreiver = data["tfidf_char_retreiver"]
        print(f"Loaded retreiver from {path}")

    def retreive(self, claim_text: str, top_k: int | None = None) -> tuple[list[Any], np.ndarray]:
        # retreive evidence ids and similarities from the transformer and bm25 retreivers
        retrieved_ids, transformer_similarities = self.transformer_retreiver.retreive(claim_text)
        bm25_ids, bm25_scores = self.bm25_retreiver.retreive(claim_text)
        tfidf_char_ids, tfidf_char_similarities = self.tfidf_char_retreiver.retreive(claim_text)

        # normalize the scores
        transformer_similarities = self._minmax_normalize(transformer_similarities)
        bm25_scores = self._minmax_normalize(bm25_scores)
        tfidf_char_similarities = self._minmax_normalize(tfidf_char_similarities)

        # combine the scores by evidence id for safer merging
        transformer_by_id = dict(zip(retrieved_ids, transformer_similarities))
        bm25_by_id = dict(zip(bm25_ids, bm25_scores))
        tfidf_char_by_id = dict(zip(tfidf_char_ids, tfidf_char_similarities))


        # combine the scores by evidence id for safer merging
        combined: dict[Any, float] = {}
        for eid in set(transformer_by_id) | set(bm25_by_id):
            t = float(transformer_by_id.get(eid, 0.0))
            b = float(bm25_by_id.get(eid, 0.0))
            c = float(tfidf_char_by_id.get(eid, 0.0))
            combined[eid] = self.alpha * t + self.beta * b + self.gamma * c

        # sort and filter
        k = top_k if top_k is not None else self.top_k
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]

        # return the top-k evidence ids and scores for consistency
        top_ids = [eid for eid, _ in ranked]
        top_scores = np.asarray([score for _, score in ranked], dtype=np.float64)
        return top_ids, top_scores
    
    def retreive_all(self, claims: dict, visual_sim=True) -> dict:
        """
        Retreive evidence for all claims in the provided dictionary of claims.
        args:
            claims (dict): A dictionary where keys are claim IDs and values are claim data, including the claim text.
        returns:
            A dictionary where keys are claim IDs and values are dictionaries containing the claim text, a dummy claim label, and the list of retreived evidence IDs.
        """
        results = {}
        top_similarities_all = {}
        for claim_id, claim_data in tqdm(claims.items(), desc="Retrieving evidence for claims"):
            claim_text = claim_data["claim_text"]
            retreived_evidence, top_similarities = self.retreive(claim_text)

            results[claim_id] = {
                "claim_text": claim_text,
                "claim_label": "NOT_ENOUGH_INFO",  # temporary dummy label
                "evidences": retreived_evidence
            }

            if visual_sim:
                top_similarities_all[claim_id] = top_similarities
        
        if visual_sim:
            _plot_similarity_scatter(top_similarities_all, title="Combined Score for Retreived Evidence", ylabel="Combined Score")    

        return results

    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.max() == scores.min():
            return np.ones_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def _normalize_weights(self):
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        
        # normalize the weights to sum to 1
        total = alpha + beta + gamma

        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        

class TransformerRetreiver:
    def __init__(self, top_k=50):
        self.top_k = top_k
        self.embedder = SentenceEmbedder()

        self.evidence = None
        self.evidence_ids = None
        self.evidence_texts = None
        self.evidence_embeddings = None

    def fit(self, evidence):
        try:
            self._load_embeddings()
            print("Loaded existing evidence embeddings. Skipping embedding computation.")
        except FileNotFoundError:
            # initialize the retreiver with the evidence data, creating an embedding matrix for efficient retrieval
            print("No existing embeddings found. Computing evidence embeddings...")
            self.evidence = evidence
            self.evidence_ids = list(evidence.keys())
            self.evidence_texts = list(evidence.values())
            self.evidence_embeddings: np.ndarray = self.embedder.encode_texts(self.evidence_texts) # dimension: (1208827, 384)
            self._save_embeddings()

    def retreive(self, claim_text: str, top_k: int | None = None):
        k = top_k or self.top_k

        # 1. Encode claim
        claim_embedding = self.embedder.encode_texts([claim_text]).astype(np.float32)

        # 2. Normalize claim embedding
        claim_embedding /= np.linalg.norm(claim_embedding, axis=1, keepdims=True) + 1e-12

        # 3. Cosine similarity = dot product if evidence embeddings are already normalized
        similarities = claim_embedding @ self.evidence_embeddings.T
        similarities = similarities.ravel()

        # 4. Faster top-k selection, avoids sorting the whole array
        top_indices = np.argpartition(similarities, -k)[-k:]

        # 5. Sort only the top-k results
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        retreived = [self.evidence_ids[i] for i in top_indices]
        top_similarities = similarities[top_indices]

        return retreived, top_similarities

    # def retreive(self, claim_text: str, top_k: int | None = None) -> tuple[list[Any], np.ndarray]:
    #     """
    #     Retreive the top-k most relevant evidence entries for a given claim text based on cosine similarity of embeddings.
    #     args:
    #         claim_text (str): The text of the claim for which we want to retreive evidence.
    #     returns:
    #         List of evidence IDs corresponding to the top-k most relevant evidence entries.
    #     """
    #     # encode the claim text into an embedding vector
    #     claim_embedding = self.embedder.encode_texts([claim_text])

    #     # compute cosine similarities between the claim embedding and all evidence embeddings
    #     similarities: np.ndarray = cosine_similarity(
    #         claim_embedding,
    #         self.evidence_embeddings
    #     ).flatten()

    #     # sort
    #     # top k indices with highest similarity scores (descending order)
    #     top_indices: np.ndarray = similarities.argsort()[::-1][:top_k or self.top_k]
 
    #     # map the top indices back to evidence IDs
    #     retreived = [
    #         self.evidence_ids[i]
    #         for i in top_indices
    #     ]

    #     top_similarities: np.ndarray = similarities[top_indices]
    #     return retreived, top_similarities
    
    def retreive_all(self, claims: dict, visual_sim=True) -> dict:
        """
        Retreive evidence for all claims in the provided dictionary of claims.
        args:
            claims (dict): A dictionary where keys are claim IDs and values are claim data, including the claim text.
        returns:
            A dictionary where keys are claim IDs and values are dictionaries containing the claim text, a dummy claim label, and the list of retreived evidence IDs.
        """
        results = {}
        top_similarities_all = {}
        for claim_id, claim_data in tqdm(claims.items(), desc="Retrieving evidence for claims"):
            claim_text = claim_data["claim_text"]
            retreived_evidence, top_similarities = self.retreive(claim_text)

            results[claim_id] = {
                "claim_text": claim_text,
                "claim_label": "NOT_ENOUGH_INFO",  # temporary dummy label
                "evidences": retreived_evidence
            }

            if visual_sim:
                top_similarities_all[claim_id] = top_similarities
        
        if visual_sim:
            _plot_similarity_scatter(top_similarities_all, title="Transformer Score for Retreived Evidence", ylabel="Transformer Score")    

        return results

    def _save_embeddings(self, path= OUTPUT_PATH / "temp_cache" / "evidence_embeddings.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "evidence_ids": self.evidence_ids,
                "evidence_texts": self.evidence_texts,
                "evidence_embeddings": self.evidence_embeddings
            }, f)
        print(f"Saved evidence embeddings to {path}")

    def _load_embeddings(self, path= OUTPUT_PATH / "temp_cache" / "evidence_embeddings.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found at {path}. Please run fit() and save_embeddings() first.")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.evidence_ids = data["evidence_ids"]
            self.evidence_texts = data["evidence_texts"]
            self.evidence_embeddings = data["evidence_embeddings"]
        print(f"Loaded evidence embeddings from {path}")


class TfidfCharRetreiver:
    def __init__(self, top_k=5):
        self.top_k = top_k

        # Configure the TF-IDF vectorizer with common settings for text retrieval.
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            max_features=100000
            )

        self.evidence = None
        self.evidence_ids = None
        self.evidence_texts = None
        self.evidence_matrix = None

    def fit(self, evidence):
        # initialize the retreiver with the evidence data, creating a TF-IDF matrix for efficient retrieval
        self.evidence = evidence
        self.evidence_ids = list(evidence.keys())
        self.evidence_texts = list(evidence.values())
        self.evidence_matrix = self._vectorizer.fit_transform(self.evidence_texts)

    # def retreive(self, claim_text: str, top_k: int | None = None) -> tuple[list[Any], np.ndarray]:
    #     """
    #     Retreive the top-k most relevant evidence entries for a given claim text based on cosine similarity of TF-IDF vectors.
    #     args:
    #         claim_text (str): The text of the claim for which we want to retreive evidence.
    #     returns:
    #         List of evidence IDs corresponding to the top-k most relevant evidence entries.
    #     """
    #     # transform the claim text into a TF-IDF vector
    #     claim_vector = self._vectorizer.transform([claim_text])

    #     # compute cosine similarities between the claim vector and all evidence vectors
    #     # returns a 2D array of shape (1, num_evidence), so we flatten it to get a 1D array of similarities
    #     similarities: np.ndarray = cosine_similarity(
    #         claim_vector,
    #         self.evidence_matrix
    #     ).flatten()

    #     # sort
    #     # top k indices with highest similarity scores (descending order)
    #     top_indices: np.ndarray = similarities.argsort()[::-1][:top_k or self.top_k]
 
    #     # map the top indices back to evidence IDs
    #     retreived = [
    #         self.evidence_ids[i]
    #         for i in top_indices
    #     ]

    #     top_similarities = similarities[top_indices]
    #     return retreived, top_similarities

    def retreive(
        self,
        claim_text: str,
        top_k: int | None = None
    ) -> tuple[list[Any], np.ndarray]:
        """
        Retrieve the top-k most relevant evidence entries for a given claim
        using cosine similarity over TF-IDF vectors.
        """

        k = self.top_k if top_k is None else top_k

        # transform claim into TF-IDF vector
        claim_vector = self._vectorizer.transform([claim_text])
 
        # sparse matrix multiplication
        similarities = (claim_vector @ self.evidence_matrix.T).toarray().ravel()

        top_indices = np.argpartition(similarities, -k)[-k:]

        # sort only the selected top-k indices
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        # map back to evidence ids
        retreived = [
            self.evidence_ids[i]
            for i in top_indices
        ]

        top_similarities = similarities[top_indices]

        return retreived, top_similarities

    def retreive_all(self, claims: dict, visual_sim=True) -> dict:
        """
        Retreive evidence for all claims in the provided dictionary of claims.
        args:
            claims (dict): A dictionary where keys are claim IDs and values are claim data, including the claim text.
        returns:
            A dictionary where keys are claim IDs and values are dictionaries containing the claim text, a dummy claim label, and the list of retreived evidence IDs.
        """
  
        results = {}
        top_similarities_all = {}
        for claim_id, claim_data in tqdm(claims.items(), desc="Retreiving evidence for claims"):
            claim_text = claim_data["claim_text"]
            retreived_evidence, top_similarities = self.retreive(claim_text)

            results[claim_id] = {
                "claim_text": claim_text,
                "claim_label": "NOT_ENOUGH_INFO",  # temporary dummy label
                "evidences": retreived_evidence
            }

            if visual_sim:
                top_similarities_all[claim_id] = top_similarities
        
        if visual_sim:
            _plot_similarity_scatter(top_similarities_all, title="TF-IDF Score for Retreived Evidence", ylabel="TF-IDF Score")    

        return results


class BM25Retreiver:
    """
    Sparse lexical retrieval with Okapi BM25. Tokenization and vocabulary match
    ``TfidfRetreiver``: a TF-IDF vectorizer is fit once to define the vocab,
    then BM25 weights are computed over that vocabulary.

    BM25 boosts documents where query terms occur often, while down-weighting very
    long documents via length normalization (parameters ``k1`` and ``b``).
    """

    def __init__(self, top_k=5, k1=1.5, b=0.75):
        # How many evidence passages to return per query claim.
        self.top_k = top_k
        # BM25 saturation: higher k1 lets term frequency saturate more slowly.
        self.k1 = k1
        # BM25 length normalization strength in [0, 1]; b=1 fully normalizes by doc length.
        self.b = b

        self._vectorizer = TfidfVectorizer(
            tokenizer=lemma_tokenizer,
            stop_words="english",
            lowercase=False,
            ngram_range=(1, 2),
            max_features=50000,
            )

        self.evidence = None
        self.evidence_ids = None
        self.evidence_texts = None
 
        self._text_analyzer = None
        self._term_string_to_index = None
        self._num_vocab_terms = None
        self._num_evidence_documents = None
        self._document_token_lengths = None
        self._average_document_length = None
        self._IDF = None
        self._inverted_index_by_term = None

    def fit(self, evidence):
        """Traing stage: Index all evidence passages: vocabulary, postings, lengths, IDF.
        args:
            evidence (dict): A dictionary where keys are evidence IDs and values are evidence texts.
        """

        # ============Step 1. fit the vectorizer to get the vocabulary============
        self.evidence = evidence
        self.evidence_ids = list(evidence.keys())
        self.evidence_texts = list(evidence.values())
        num_documents = len(self.evidence_texts)

        # Learn vocab from the corpus (identical footprint to TF-IDF retriever).
        self._vectorizer.fit(self.evidence_texts)
        self._text_analyzer = self._vectorizer.build_analyzer()

        self._term_string_to_index = self._vectorizer.vocabulary_ # dictionary mapping terms to their indices in the vocabulary

        self._num_vocab_terms = len(self._term_string_to_index) # number of terms in the vocabulary
        self._num_evidence_documents = num_documents # number of evidence documents

        # ============Step 2. Initialize the inverted index============
        document_frequency_per_term = np.zeros(self._num_vocab_terms, dtype=np.int32)   # Document frequency (DF) per term
        self._document_token_lengths = np.zeros(num_documents, dtype=np.float64)        # token count for each evidence document
        self._inverted_index_by_term = [[] for _ in range(self._num_vocab_terms)]       # inverted index: term -> list of (document_index, term_count)(for faster retrieval)

        # ============Step 3. Process each evidence text by counting statistics============
        for document_index, evidence_text in tqdm(enumerate(self.evidence_texts), desc="Processing evidence texts"):

            # Count each vocab term inside this passage.
            term_frequency = defaultdict(int)
            for token in self._text_analyzer(evidence_text):
                term_index = self._term_string_to_index.get(token)
                if term_index is None:
                    continue
                term_frequency[term_index] += 1 # count the number of times the term appears in the evidence text (TF)

            passage_length_tokens = sum(term_frequency.values())            # record the token count for the evidence text
            self._document_token_lengths[document_index] = passage_length_tokens    # store 

            # ============Step 4. Update the inverted index and document frequencies============
            for term_index, frequency in term_frequency.items():
                self._inverted_index_by_term[term_index].append(
                    (document_index, frequency)
                ) # record the TF for the evidence text in the inverted index
                document_frequency_per_term[term_index] += 1 # update the document frequency for the term

        # ============Step 5. Compute the average document length============
        total_tokens = float(self._document_token_lengths.sum())
        self._average_document_length = (
            total_tokens / num_documents if num_documents else 0.0
        )

        # ============Step 6. Compute the inverse document frequency (IDF)============
        document_frequency_float: np.ndarray = document_frequency_per_term.astype(np.float64) # convert to float for numerical operations
        numerator: np.ndarray = num_documents - document_frequency_float + 0.5
        denominator: np.ndarray = document_frequency_float + 0.5
        self._IDF: np.ndarray = np.log(numerator / denominator + 1.0) # compute the inverse document frequency

    def retreive(self, claim_text: str, top_k: int | None = None) -> tuple[list[Any], np.ndarray]:
        """
        Retrieve the top-k evidence IDs for ``claim_text`` using BM25 scores.
        Same return shape as ``TfidfRetreiver.retreive``.
        args:
            claim_text (str): The text of the claim for which we want to retreive evidence.
        returns:
            List of evidence IDs corresponding to the top-k most relevant evidence entries.
        """

        # ============Step 1. initialize the BM25 scores vector============
        bm25_scores = np.zeros(self._num_evidence_documents, dtype=np.float64) # for each evidence document, init the BM25 score
        length_norm_k1 = self.k1
        length_norm_b = self.b

        # ============Step 2. get the average document length with safe fallback============
        corpus_avg_length = (
            self._average_document_length
            if self._average_document_length and self._average_document_length > 0
            else 1.0
        )

        # ============Step 3. tokenize and count the term frequencies (TF) in the query============
        query_term_frequencies = defaultdict(int)
        for token in self._text_analyzer(claim_text):
            term_index = self._term_string_to_index.get(token)
            if term_index is None:
                continue
            query_term_frequencies[term_index] += 1

        # ============Step 4. compute BM25 scores============
        # iterate over each term in the query
        for term_index, times_in_query in query_term_frequencies.items():
            idf_weight = self._IDF[term_index] # IDF
            postings = self._inverted_index_by_term[term_index]
            for document_index, term_freq in postings:
                document_length_tokens = self._document_token_lengths[document_index] # current document length

                # BM25 denominator: length-normalized component
                length_component = (
                    1.0
                    - length_norm_b
                    + length_norm_b
                    * (document_length_tokens / corpus_avg_length)
                )
                term_frequency_denominator = (
                    term_freq + length_norm_k1 * length_component
                )

                numerator = term_freq * (length_norm_k1 + 1.0)
                bm25_term_score = idf_weight * numerator / term_frequency_denominator
                bm25_scores[document_index] += times_in_query * bm25_term_score # upate the BM25 score for the evidence document    
        
        # ============Step 5. sort the BM25 scores and get the top-k evidence IDs============
        # top_ranked_indices: np.ndarray = bm25_scores.argsort()[::-1][:top_k or self.top_k]

        # retrieved_ids = [
        #     self.evidence_ids[position] for position in top_ranked_indices
        # ]
        # top_bm25_scores = bm25_scores[top_ranked_indices]
        # return retrieved_ids, top_bm25_scores
        k = self.top_k if top_k is None else top_k
        k = min(k, self._num_evidence_documents)

        top_ranked_indices = np.argpartition(bm25_scores, -k)[-k:]

        # sort only the selected top-k indices
        top_ranked_indices = top_ranked_indices[
            np.argsort(bm25_scores[top_ranked_indices])[::-1]
        ]

        retrieved_ids = [
            self.evidence_ids[position]
            for position in top_ranked_indices
        ]

        top_bm25_scores = bm25_scores[top_ranked_indices]

        return retrieved_ids, top_bm25_scores

    def retreive_all(self, claims: dict, visual_sim=True) -> dict:
        """
        Retrieve evidence for all claims. Same structure as ``TfidfRetreiver.retreive_all``.
        """

        results = {}
        top_scores_all_claims = {}
        for claim_id, claim_data in tqdm(claims.items(), desc="Retreiving evidence for claims"):
            claim_text = claim_data["claim_text"]
            retrieved_evidence_ids, top_bm25_scores_for_claim = self.retreive(
                claim_text
            )

            results[claim_id] = {
                "claim_text": claim_text,
                "claim_label": "NOT_ENOUGH_INFO",
                "evidences": retrieved_evidence_ids,
            }

            if visual_sim:
                top_scores_all_claims[claim_id] = top_bm25_scores_for_claim

        if visual_sim:
            _plot_similarity_scatter(top_scores_all_claims, title="BM25 Score for Retreived Evidence", ylabel="BM25 Score")    

        return results