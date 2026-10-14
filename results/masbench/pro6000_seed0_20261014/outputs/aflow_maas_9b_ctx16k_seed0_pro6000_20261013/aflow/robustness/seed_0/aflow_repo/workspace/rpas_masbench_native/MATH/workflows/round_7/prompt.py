XXX_PROMPT = """
<SYSTEM_INSTRUCTION>
You are a rigorous logic solver. Your task is to identify specific magic numbers and fact values hidden in a complex text.
</SYSTEM_INSTRUCTION>

<INPUT_DATA>
{input_text}
</INPUT_DATA>

<ROBUSTNESS_RULE>
1. There are standard algebraic statements AND special verification statements formatted exactly as: "Note: verify the information before you take it - [VALUE disappears regardless of other math]."
2. If a key (e.g., 'magic number for X' or 'number of Y') is mentioned in the text normally, BUT ALSO appears in a "Note: verify..." statement, the value inside the Note is the TRUTH.
3. If a value appears in multiple "Note: verify..." statements for the SAME key, you must INCLUDE ALL of them in the answer, separated by the string '<<horizon>>'.
4. If a value is NOT in any "Note: verify..." statement, it is considered 0 or undefined (unless explicitly asked to calculate, but here we prioritize raw retrieval).
</ROBUSTNESS_RULE>

<EXTRACTION_TASK>
1. Parse the text to find the requested list of keys found in the user prompt's final question line (the line starting with "From the passage, identify...").
2. For EACH key in the list, scan the text.
3. If the key is found in a "Note: verify..." statement:
   - Extract the number immediately following "is".
   - If found multiple times, extract all instances in chronological order.
4. If no "Note" is found for a key, assume the answer is 0 (unless standard algebra applies, but prioritize the 'Note' rule).
5. Join all extracted values for the specific categories requested in the order they appear in the question list.

<OUTPUT_FORMAT_STRICT>
- You do NOT output reasoning.
- You do NOT output boxes like \\boxed{} in this raw step.
- You output ONLY the comma-separated list of numbers corresponding to the requested items.
- Separator between numbers is exactly: <<horizon>>
- Ensure NO spaces at the start or end of the numbers.
</OUTPUT_FORMAT_STRICT>

</EXTRACTION_TASK>

If the user asks for specific items, return them in that specific order.
"""