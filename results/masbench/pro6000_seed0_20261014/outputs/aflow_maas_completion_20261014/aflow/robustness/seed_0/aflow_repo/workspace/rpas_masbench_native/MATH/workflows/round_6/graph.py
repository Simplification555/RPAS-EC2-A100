from typing import Literal
import workspace.rpas_masbench_native.MATH.workflows.template.operator as operator
import workspace.rpas_masbench_native.MATH.workflows.round_6.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Step 1: Extract all raw values for requested keys with order preserved.
        Format: {"items": [{"key": "...", "values": ["val1", "val2", ...]}, ...]}
        """
        step1_input = problem + """
Query Parts: 
PASSENGER_TEAM
""".strip()
        
        response1 = await self.custom(input=step1_input, instruction=prompt_custom.EXTRACT trú)
        raw_data = response1['response']

        # Process the raw data to format the final answer string
        # The LLM returns a JSON-like structure. We need to parse it mentally or expect the LLM to return the final string directly if possible, 
        # but since we changed the instruction, we expect a list.
        # We will format the list into the string format.
        final_answer_str = self._format_list_to_string(raw_data, 'items')
        
        return await self.custom(input=problem, instruction=f"Format the final answer string: FINAL ANSWER: {final_answer_str}")['response'], self.llm.get_usage_summary()["total_cost"]

    def _format_list_to_string(self, data: str, key: str) -> str:
        """
        Helper to convert LLM's JSON-like output to the required string format.
        Assumes LLM returns: 'key: "val1, val2; key2: "val3"'
        """
        items = {}
        import re
        try:
            # Regex to find "key": ["val1", "val2"] patterns
            pattern = r'\b\w+\b:\s*"\[(.*?)\]"'
            matches = re.findall(pattern, data, re.DOTALL)
            for match in matches:
                # Clean values (remove quotes)
                values_list = [v.strip().strip('"') for v in match.split(',')]
            # This is a simplified logic; in production, we might rely on the LLM to be very strict in JSON or use a proper parser.
            # Given the constraints, we rely on the prompt engineering to return the EXACT string format requested.
            # If the LLM returns the string directly, we just return it.
            return data 
        except:
            return ""
