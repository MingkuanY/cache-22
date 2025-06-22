# File: utils/run_prompt.py

import sys
from main import simulate_prompt_flow_return  # Make sure this returns a string

if __name__ == "__main__":
    prompt = sys.argv[1]
    result = simulate_prompt_flow_return(prompt)
    print(result)
