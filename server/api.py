from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from server.simulate import simulate_prompt_flow  # or however you import it

app = FastAPI()

# CORS settings for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # You can limit to ["http://localhost:3000"] if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query")
async def query(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return {"error": "No prompt provided"}

    result = simulate_prompt_flow(prompt)
    return result
