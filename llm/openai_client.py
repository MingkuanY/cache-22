from openai import OpenAI

client = OpenAI()

def decompose_prompt(prompt: str, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You're a tool that rewrites a long user prompt into 2–4 reusable subquestions. Do not number or label the subquestions. Just return each one on a new line as plain text."
            },
            {
                "role": "user",
                "content": f"Break this prompt down into atomic subquestions. Do not number them:\n{prompt}"
            }
        ],
        temperature=0.3,
    )
    text = response.choices[0].message.content
    return [line.strip("- ").strip() for line in text.split("\n") if line.strip()]