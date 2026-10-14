STRUCTURE_ANALYSIS = """
First, analyze the dependency graph of variables in the problem. Check for any circular dependencies (cycles). If a cycle is detected, output a JSON object: {"is_dag": false, "reason": "Circular dependency detected."}. If the graph is a DAG (Acyclic), proceed to solve the problem step-by-step.

Solve the problem carefully. For iGSM arithmetic, reduce all results modulo 23. Preserve the requested answer order. If there are multiple answers, join them with `<<horizon>>`. 

Generate the final line strictly as:
FINAL ANSWER: <answer>

Output ONLY the JSON object with keys "is_dag" (boolean) and "response" (the full text of the answer including the FINAL ANSWER line, or the reason if cycle detected). Do not output any markdown code blocks or other text.
"""