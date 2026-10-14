PROMPT_SOLVER = """You are an expert solver for math word problems. Solve ONLY the specific question to get a single integer between 0-22.

Rules:
1. Perform all arithmetic in Z_23 (modulo 23).
2. Show step-by-step derivation briefly.
3. Output EXACTLY ONE integer inside \\boxed{}, e.g., \boxed{5}. No text outside.

Question: 
{input}
"""

PROMPT_ALT = """You are an expert solver for math word problems. Solve ONLY the specific question to get a single integer between 0-22 using a different reasoning strategy.

Rules:
1. Perform all arithmetic in Z_23 (modulo 23).
2. Verify intermediate sums are reduced mod 23 at each step.
3. Output EXACTLY ONE integer inside \\boxed{}, e.g., \boxed{5}. No text outside.

Question: 
{input}
"""