from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> np.ndarray:
    # Return a 1D numpy array of shape (384,)
    vec = model.encode([text], convert_to_numpy=True)  # This returns shape (1, 384)
    return vec[0].astype(np.float32)  # Return the first row

if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "What is gravity"
    vec = embed(text)
    print(f"Vector shape: {vec.shape}")
    print(f"Vector preview: {vec[:5]}")

