# Only used for CLI testing, obsolete for full stack app

from cache.cache_manager import CacheManager
from embedding.real_embedder import embed
import time
import numpy as np
import tiktoken
from llm.openai_client import decompose_prompt, gpt4_generate_response, gpt3_5_synthesize

cache = CacheManager()
encoder = tiktoken.encoding_for_model("gpt-4-turbo")  # For local token counting

def count_tokens(text):
    return len(encoder.encode(text))

def show_metrics_terminal(
    components, hits, misses, time_taken, tokens_used, tokens_saved, api_calls
):
    print("\n=== GPT Cache Performance ===")
    print(f"Total components processed   : {components}")
    print(f"Cache hits                   : {hits}")
    print(f"Cache misses                 : {misses}")
    print(f"API Calls (compute used)     : {api_calls}")
    print(f"Tokens used (compute spent)  : {tokens_used}")
    print(f"Tokens saved (cache reuse)   : {tokens_saved}")
    print(f"Time taken (s)               : {time_taken:.2f}")
    
    hit_rate = (hits / components * 100) if components > 0 else 0
    print(f"Cache hit rate               : {hit_rate:.1f}%")

    estimated_total_tokens = tokens_used + tokens_saved
    token_savings_pct = (tokens_saved / estimated_total_tokens * 100) if estimated_total_tokens else 0
    print(f"→ Token savings              : {token_savings_pct:.1f}% of total token usage")

    estimated_total_api_calls = components
    api_call_savings = estimated_total_api_calls - api_calls
    api_savings_pct = (api_call_savings / estimated_total_api_calls * 100) if estimated_total_api_calls else 0
    print(f"→ API call savings           : {api_savings_pct:.1f}% fewer calls than full generation")

    print("\n💡 Efficiency Summary:")
    print(f"By reusing cached results, we reduced compute by {int(token_savings_pct)}% and avoided {api_call_savings} full model calls.")
    print(f"That means faster answers, lower cost, and less energy use. 🚀")
    print()


def simulate_prompt_flow(prompt):
    # Decompose into components
    components = decompose_prompt(prompt)
    print("\n--- Decomposed Components ---")
    for i, comp in enumerate(components, 1):
        print(f"{i}. {comp}")
    print()
    
    hits, misses = 0, 0
    tokens_used_api = 0
    tokens_saved_by_cache = 0
    api_calls = 0
    start_time = time.perf_counter()
    
    responses = []

    print("\n--- Processing Components ---")
    for comp in components:
        vec = embed(comp)
        print(f"Vector shape: {np.array(vec).shape}")

        cached = cache.check_cache(vec)

        if cached:
            hits += 1
            estimated_saved = len(encoder.encode(cached))
            tokens_saved_by_cache += estimated_saved
            print(f"[HIT]   \"{comp}\" → Using cached response (estimated tokens saved: {estimated_saved})")
            responses.append(cached)
        else:
            misses += 1
            api_calls += 1
            response, tokens_used = gpt4_generate_response(comp)
            tokens_used_api += tokens_used
            print(f"[MISS]  \"{comp}\" → Generated new response (tokens used: {tokens_used})")
            cache.add_to_cache(vec, response)
            responses.append(response)

    time_taken = time.perf_counter() - start_time

    final_answer = gpt3_5_synthesize(responses)

    return {
        "final_answer": final_answer,
        "metrics": {
            "components": len(components),
            "hits": hits,
            "misses": misses,
            "api_calls": api_calls,
            "tokens_used": tokens_used_api,
            "tokens_saved": tokens_saved_by_cache,
            "time_taken": time_taken,
        }
    }



if __name__ == "__main__":
    print("Welcome to Cache-22!")
    print("Enter 'exit' to quit.\n")

    while True:
        user_prompt = input("Enter your prompt: ")
        if user_prompt.lower() in {"exit", "quit"}:
            break
        result = simulate_prompt_flow(user_prompt)
        print("\n=== Synthesized Final Response ===")
        print(result["final_answer"])
        show_metrics_terminal(**result["metrics"])

