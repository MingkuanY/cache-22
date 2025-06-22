# backend/simulate.py
from cache.cache_manager import CacheManager
from embedding.real_embedder import embed
from llm.openai_client import decompose_prompt, gpt4_generate_response, gpt3_5_synthesize
import time
import tiktoken

cache = CacheManager()
encoder = tiktoken.encoding_for_model("gpt-4-turbo")

def count_tokens(text):
    return len(encoder.encode(text))

def simulate_prompt_flow(prompt):
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

    for comp in components:
        vec = embed(comp)
        cached = cache.check_cache(vec)

        if cached:
            hits += 1
            estimated_saved = count_tokens(cached)
            tokens_saved_by_cache += estimated_saved
            print(f"[HIT]   \"{comp}\" → Using cached response (estimated tokens saved: {estimated_saved})")
            responses.append(cached)
        else:
            misses += 1
            api_calls += 1
            response, tokens_used = gpt4_generate_response(comp)
            tokens_used_api += tokens_used
            cache.add_to_cache(vec, response)
            print(f"[MISS]  \"{comp}\" → Generated new response (tokens used: {tokens_used})")
            responses.append(response)

    time_taken = time.perf_counter() - start_time
    final_answer, synth_tokens_used = gpt3_5_synthesize(responses)
    tokens_used_api += synth_tokens_used
    api_calls += 1

    return {
        "final_response": final_answer,
        "metrics": {
            "components": len(components),
            "hits": hits,
            "misses": misses,
            "api_calls": api_calls,
            "tokens_used": tokens_used_api,
            "tokens_saved": tokens_saved_by_cache,
            "time_taken": round(time_taken, 2),
        }
    }

