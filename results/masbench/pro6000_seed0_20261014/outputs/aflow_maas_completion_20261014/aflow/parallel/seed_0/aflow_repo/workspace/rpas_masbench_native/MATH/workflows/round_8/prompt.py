MATH_BATCH_SOLUTION_PROMPT = """Solve the following independent subproblems strictly following these rules:
1. Arithmetic is in Z_23 (modulus 23). Reduce ALL intermediate results to integers 0-22.
2. Each problem is isolated. Do not use answers from Problem N to solve Problem N+1 unless explicitly dependent within the text of that specific problem.
3. For any undefined variable, assume value 0.
4. Format Requirement: For each problem, provide the derivation, then end with a line exactly as: `Problem <N> Answer: <integer_value>`. 
   - Do NOT use LaTeX boxed notation inside the explanation.
   - Ensure there are NO `<<horizon>>` separators inside individual problem blocks; use only the integer value.
5. Final Output Construction: After all problems, you must generate a single line containing ONLY the joined string of answers separated by `<<horizon>>`.
   Example structure for internal generation:
   Problem 1 Derivation...
   Problem 1 Answer: 22
   Problem 2 Derivation...
   Problem 2 Answer: 3
   <-- SYSTEM WILL EXTRACT ABOVE LINES TO FORM FINAL ANSWER: 22<<horizon>>3 -->
6. Do not include "FINAL ANSWER:" in the model output text itself; generate the problem-specific answer lines as described.

Subproblems:
{user_problem}
"""