import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class CaptionRetriever:
    def __init__(self, index_path, model_name=EMBEDDING_MODEL_NAME):
        data = np.load(index_path, allow_pickle=True)
        self.images = data["images"]
        self.captions = data["captions"]
        self.embeddings = data["embeddings"]
        self.embedder = SentenceTransformer(model_name)

    def search(self, query, top_k=5):
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "image": str(self.images[i]),
                "caption": str(self.captions[i]),
                "score": float(scores[i]),
            }
            for i in top_indices
        ]
