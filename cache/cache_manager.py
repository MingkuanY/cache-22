import faiss
import numpy as np

class CacheManager:
  def __init__(self):
    self.index = faiss.IndexFlatIP(384)  # IP = inner product ≈ cosine similarity
    self.stored_vectors = []
    self.responses = []

  def add_to_cache(self, vec, response):
    vec = vec / np.linalg.norm(vec)
    vec_np = np.array([vec]).astype("float32")
    self.index.add(vec_np)
    self.stored_vectors.append(vec)
    self.responses.append(response)

  def check_cache(self, vec, threshold=0.8):  # bump up threshold since IP goes from -1 to 1
    if len(self.stored_vectors) == 0:
        return None

    vec = vec / np.linalg.norm(vec)
    vec_np = np.array([vec]).astype("float32")
    D, I = self.index.search(vec_np, 1)  # top-1

    similarity = D[0][0]
    print(f"Similarity: {similarity:.4f}")
    if similarity >= threshold:
        return self.responses[I[0][0]]

    return None

