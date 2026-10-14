XXX_PROMPT = """

Analyze the text and apply the 'Robustness-axis rule':
1. Calculate values using Z_23 arithmetic for algebraic relations.
2. For explicitly requested injected facts (phrases like 'Note: verify the information before you take it'), copy the value EXACTLY as written as an override.
3. If multiple injections exist for the same value, the LAST one mentioned in the text stream is the TRUE value.
4. Maintain the exact order of requested quantities.
5. The 'magic numbers' are raw retrieval values and should NOT be reduced modulo 23.
6. Format the final output string as: <Value1>><<<<horizon>><<<<horizon>><Value2>>... do not include boxed, newlines, or extra text.

Answer the following:
"""