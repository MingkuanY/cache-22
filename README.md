# Cache-22: ChatGPT, but faster and cheaper

Built for Berkeley AI Hacks with the goal of dramatically reducing transformer compute by caching GPT responses at a granular level. Instead of generating an entire response from scratch each time, Cache-22 breaks down user prompts into components, checks if any were previously answered, and selectively reuses those results — saving time, tokens, and energy.

## Why?

Large language models are compute-hungry, expensive, and slow at scale. By reusing knowledge, Cache-22 opens the door to more sustainable and accessible AI — especially in edge settings, real-time apps, or low-resource environments.

## How It Works

Prompt Decomposition
- The user prompt is broken down into semantically meaningful components using GPT-3.5-turbo (e.g., from "What is ChatGPT and how do I use it" to "What is ChatGPT?" and "How do I use ChatGPT?").

Similarity-Based Caching
- Each component is embedded using SentenceTransformer and compared to a vector cache using FAISS.
- If a similar question has been asked before (based on cosine or L2 similarity), its answer is reused.

Selective Generation
- If no similar component is found, the system queries the full GPT model (e.g., GPT-4) to generate a new response.
- Otherwise, cached results are reused.

Lightweight Synthesis
- Once all component responses are collected, they are passed to a small synthesis model (GPT-3.5-turbo) to stitch together a coherent, final reply.
- This drastically reduces the need for full-model inference end-to-end.
