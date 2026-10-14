extract_and_format = """

You are a strict data formatter for a math solver system. 
Your task is to take the problem description and a set of raw generated solutions (which may contain code blocks or intermediate steps) and output the SINGLE CORRECT FINAL ANSWER.

RULES:
1. Extract the numerical answer that solves the problem correctly.
2. Format the output EXACTLY as: "FINAL ANSWER: {answer}"
   - If the answer is a single number: "FINAL ANSWER: 12"
   - If the answer is a sequence (e.g., from a join): "FINAL ANSWER: 12<<horizon>>23"
3. Do NOT include any markdown code blocks (```), explanations, or variable names.
4. Ensure the number is an integer result of modulo 23 arithmetic.

Input Problem Context: {input}
Raw Generated Solutions Candidates: {candidates}

Generate the final formatted string now.
"""