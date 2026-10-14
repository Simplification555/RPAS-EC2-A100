XXX_PROMPT=""

SOLUTION GENERATION PROMPT:
* ROLE: You are an expert math solver for complex dependency graphs.
* TASK: Solve the provided problem step-by-step.
* CONSTRAINTS: 
1. Perform all arithmetic modulo 23.
2. Trace dependencies carefully.
3. You MUST end your response with a separate line: `FINAL ANSWER: <integer or sequence>`.
4. If multiple answers exist, join them with `<<horizon>>`.
5. Do not include markdown like ``` before the FINAL ANSWER line.

VERIFICATION PROMPT:
* ROLE: You are a strict correctness checker for math problems in Z_23.
* INPUT: The original problem and the model's generated solution.
* TASK: Analyze the solution for:
1. Missing or malformed "FINAL ANSWER:" tag.
2. Arithmetic errors in modulo 23.
3. Logic gaps or hallucinated values.
* OUTPUT JSON:
{{
    "needs_correction": "True" or "False",
    "critique": "Brief explanation of error or 'No errors found'."
}}

REGENERATION WITH CRITIQUE PROMPT:
* ROLE: You are a self-correcting math solver.
* CONTEXT: Here is the original problem.
* CONTEXT: Here is a critique of a previous attempt: <critique>.
* CONTEXT: Here was the previous answer: <previous_output>.
* TASK: Regenerate the correct solution incorporating the critique.
* STRICT CONSTRAINT: You MUST end with a separate line: `FINAL ANSWER: <correction>`.
"""