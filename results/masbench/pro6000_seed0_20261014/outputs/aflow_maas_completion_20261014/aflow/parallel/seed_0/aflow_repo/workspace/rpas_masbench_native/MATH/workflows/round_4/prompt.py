MATH_BATCH_SOLUTION_PROMPT = """Solve the following set of independent subproblems carefully. Follow the MASBench/iGSM constraints strictly:
1. All arithmetic is in Z_23 (modulus 23). Reduce all intermediate results to integers 0-22.
2. Do NOT solve problems sequentially based on previous answers unless explicitly required by a dependency chain within a single problem. Treat each problem as an isolated system.
3. If a variable is referenced but not defined in the text for a specific problem, assume its value is 0.
4. Strictly separate the answers for different problems using the `<<horizon>>` separator.
5. For each problem, provide the step-by-step derivation, then end the derivation section with exactly one line: `Problem <N> Answer: <boxed_value>`. Do not use the format `\\boxed{...}` inside the explanation text, only in this specific final line for the problem.
6. Example of correct structure for non-dependent questions:
"Problem 1: [Derivation] Problem 1 Answer: <value>
Problem 2: [Derivation] Problem 2 Answer: <value>"
7. Apply everything modulo 23 at every step.

Subproblems to solve:
{user_problem}
"""