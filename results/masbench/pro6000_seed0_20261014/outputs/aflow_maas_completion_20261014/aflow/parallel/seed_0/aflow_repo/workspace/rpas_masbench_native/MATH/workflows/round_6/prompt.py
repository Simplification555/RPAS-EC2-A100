BATCH_PARSING_PROMPT = """You are an expert math solver in Z_23 (modulus 23). 
Input: {user_problem}
Task: 
1. Solve all independent subproblems.
2. For each problem, find the final answer.
3. Output EXACTLY one line per problem in this format: "Problem X Was: <answer>" where <answer> is an integer between 0 and 22.
4. Do not write any explanation text, derivations, or boxed symbols. Only the specified lines.
5. If a line cannot be determined, output "Problem X Was: 0".
Example: Problem 1 Was: 10
        Problem 2 Was: 16
""".strip()

BATCH_PARSING_PROMPT_RETRY = """You are an expert math solver in Z_23 (modulus 23). 
Input: {user_problem}
Task: 
1. Solve all independent subproblems rigorously.
2. Output EXACTLY one line per problem in this format: "Problem X Was: <answer>".
3. <answer> must be an integer between 0 and 22 (e.g., 0, 1, ..., 22).
4. NO EXPLANATIONS. NO BOXED SYMBLOS. NO EXTRA TEXT.
5. Format strictly: "Problem 1 Was: <val>"
Example: Problem 1 Was: 10
        Problem 2 Was: 16
""".strip()