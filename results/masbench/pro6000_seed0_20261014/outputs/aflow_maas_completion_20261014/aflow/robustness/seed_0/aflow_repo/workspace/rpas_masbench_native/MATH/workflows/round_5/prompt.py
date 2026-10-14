default_prompt_custom = """
Analyze the provided text strictly following the 'Robustness-axis rule'.
Rule: Extract values defined by algebraic relations in Z_23, but for every item requested, check if there is a 'Note: verify...' statement.
If a 'Note: verify...' exists for a specific item, the numeric value INSIDE that statement is the UNIQUE TRUE VALUE, overriding any previous mentions (first statements or calculations). If no note exists, use the first explicit statement.
After extracting all values for the requested items in order, join them with '<<horizon>>'.
Finally, output ONLY the string: `FINAL ANSWER: [val1<<horizon>>val2...]`.
Do not include \\boxed{} or explanation.
"""