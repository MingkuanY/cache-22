from cache.cache_manager import CacheManager
from embedding.real_embedder import embed
import time
import random
import numpy as np

cache = CacheManager()

def show_metrics_terminal(components, hits, misses, time_taken, tokens_saved):
    print("\n=== GPT Cache Performance ===")
    print(f"Total components: {components}")
    print(f"Cache hits      : {hits}")
    print(f"Cache misses    : {misses}")
    print(f"Time taken (s)  : {time_taken:.2f}")
    print(f"Tokens saved    : {tokens_saved}")
    hit_rate = (hits / components * 100) if components > 0 else 0
    print(f"Hit rate        : {hit_rate:.1f}%\n")

def simulate_prompt_flow(prompt):
    components = prompt.split(" and ")
    hits, misses = 0, 0
    tokens_saved = 0
    start_time = time.perf_counter()

    print("\n--- Processing Components ---")
    for comp in components:
        vec = embed(comp)
        print(f"Vector shape: {np.array(vec).shape}")

        cached = cache.check_cache(vec)

        if cached:
            hits += 1
            tokens_saved += random.randint(50, 150)
            print(f"[HIT]   \"{comp}\" → {cached}")
        else:
            fake_response = f"This is a generated answer for: '{comp}'"
            print(f"[MISS]  \"{comp}\" → {fake_response}")
            cache.add_to_cache(vec, fake_response)
            misses += 1

    time_taken = time.perf_counter() - start_time
    show_metrics_terminal(len(components), hits, misses, time_taken, tokens_saved)

if __name__ == "__main__":
    print("Welcome to Cache-22!")
    print("Enter 'exit' to quit.\n")

    while True:
        user_prompt = input("Enter your prompt: ")
        if user_prompt.lower() in {"exit", "quit"}:
            break
        simulate_prompt_flow(user_prompt)
