def embed(text):
    # Fake embedding: hash to simulate different vectors
    return hash(text) % 10000
