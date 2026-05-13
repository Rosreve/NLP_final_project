import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    """
    Sentence embedding wrapper using all-MiniLM-L6-v2.
    """
    def __init__(self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into sentence embeddings.

        Args:
            texts: list of input strings

        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Input texts list is empty.")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)
