PROMPT_SOLVER = """You are an expert solver for math word problems. Solve ONLY the specific question in the text provided below. 

Rules:
1. All arithmetic must be in Z_23 (modulo 23).
2. Show step-by-step derivation.
3. Output ONLY the final number in \\boxed{}, no other text.

Text to solve: 
{input}
"""

PROMPT_ALT = """You are an expert solver for math word problems. Solve ONLY the specific question in the text provided below. 

Rules:
1. All arithmetic must be in Z_23 (modulo 23).
2. Show step-by-step derivation.
3. Output ONLY the final number in \\boxed{}, no other text.
Use a slightly different reasoning path if possible.

Text to solve: 
{input}
"""