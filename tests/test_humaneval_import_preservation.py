from external_comparison.runners.humaneval import HumanEvalTask, execute_humaneval, extract_code


def test_imports_survive_code_extraction_and_scoring():
    output = 'from typing import List\n\ndef solve(xs: List[int]) -> int:\n    return sum(xs)\n'
    extracted = extract_code(output, "solve")
    assert extracted.startswith("from typing import List")
    task = HumanEvalTask("fixture/import", "unused", "def check(f):\n    assert f([1, 2]) == 3", "solve")
    assert execute_humaneval(extracted, task)["passed"]


def test_helper_before_entry_point_survives():
    output = 'def helper(x):\n    return x + 1\n\ndef solve(x):\n    return helper(x)\n'
    task = HumanEvalTask("fixture/helper", "unused", "def check(f):\n    assert f(2) == 3", "solve")
    assert execute_humaneval(extract_code(output, "solve"), task)["passed"]


def test_fenced_imports_survive():
    output = 'Here is the code:\n```python\nimport math\ndef solve(x):\n    return math.sqrt(x)\n```'
    assert extract_code(output, "solve").startswith("import math")


def test_prose_fallback_and_missing_function():
    assert extract_code('Here is the solution:\ndef solve():\n    return 2', "solve").startswith("def solve")
    assert extract_code('import math\ndef other():\n    return 2', "solve") == ""
