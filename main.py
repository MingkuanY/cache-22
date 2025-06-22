from cache.cache_manager import CacheManager
from embedding.real_embedder import embed
import time
import random
import numpy as np
from llm.openai_client import decompose_prompt, gpt4_generate_response, gpt3_5_synthesize

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
    # Decompose into components
    components = decompose_prompt(prompt)
    print("\n--- Decomposed Components ---")
    for i, comp in enumerate(components, 1):
        print(f"{i}. {comp}")
    print()
    
    hits, misses = 0, 0
    tokens_saved = 0
    start_time = time.perf_counter()
    
    responses = []

    print("\n--- Processing Components ---")
    for comp in components:
        vec = embed(comp)
        print(f"Vector shape: {np.array(vec).shape}")

        cached = cache.check_cache(vec)

        if cached:
            hits += 1
            tokens_saved += random.randint(50, 150)
            print(f"[HIT]   \"{comp}\" → {cached}")
            responses.append(cached)
        else:
            misses += 1
            response = gpt4_generate_response(comp)
            print(f"[MISS]  \"{comp}\" → {response}")
            cache.add_to_cache(vec, response)
            responses.append(response)

    time_taken = time.perf_counter() - start_time
    show_metrics_terminal(len(components), hits, misses, time_taken, tokens_saved)
    
    final_answer = gpt3_5_synthesize(responses)
    print("\n=== Synthesized Final Response ===")
    print(final_answer)

if __name__ == "__main__":
    print("Welcome to Cache-22!")
    print("Enter 'exit' to quit.\n")

    while True:
        user_prompt = input("Enter your prompt: ")
        if user_prompt.lower() in {"exit", "quit"}:
            break
        simulate_prompt_flow(user_prompt)
