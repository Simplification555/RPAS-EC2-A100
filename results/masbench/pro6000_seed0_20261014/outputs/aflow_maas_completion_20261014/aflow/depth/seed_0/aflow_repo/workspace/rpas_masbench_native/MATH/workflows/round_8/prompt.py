STRICT_SOLVER_PROMPT = """Solve the math problem using Z_23 arithmetic.
Rules:
1. Reduce all additions, subtractions, multiplications, and intermediate results modulo 23 immediately.
2. A plural category or abstract total denotes the modulo-23 sum of its direct listed members.
3. ABSOLUTE PROHIBITION: If any variable appears in the dependency chain but lacks a definition, you must explicitly treat it as 0. Do not guess or hallucinate values.
4. End with one separate line `FINAL ANSWER: ANSWER`. ANSWER must contain only the single integer result.
5. Output nothing else in the final line.
Problem:
"""