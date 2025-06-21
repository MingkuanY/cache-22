from cache.cache_manager import CacheManager
from embedding.embedder import embed
import time
import random

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
    cache = CacheManager()
    hits, misses = 0, 0
    tokens_saved = 0
    start_time = time.perf_counter()

    for comp in components:
        vec = embed(comp)
        if cache.check_cache(vec):
            hits += 1
            tokens_saved += random.randint(50, 150)
        else:
            cache.add_to_cache(vec)
            misses += 1

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    show_metrics_terminal(len(components), hits, misses, time_taken, tokens_saved)

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    simulate_prompt_flow(user_prompt)
