"""Task-level MASBench instructions shared by the external baselines.

This file intentionally contains only the benchmark task contract, not the
RPAS optimizer/search implementation.
"""

MASBENCH_ANSWER_SEPARATOR = "<<horizon>>"

MASBENCH_INSTRUCTION = (
    "/no_think\n"
    "Solve every subproblem in the MASBench task carefully and preserve the requested answer order. Follow only "
    "the dependency paths needed for the requested quantities; do not restate every fact or repeatedly speculate "
    "about missing definitions. For iGSM arithmetic, every quantity is in Z_23: reduce additions, subtractions, "
    "multiplications, and intermediate results modulo 23 to an integer from 0 through 22. A plural category or "
    "abstract total denotes the modulo-23 sum of its direct listed members. "
    f"If there are multiple answers, join them using exactly `{MASBENCH_ANSWER_SEPARATOR}` with no omitted "
    "items or reordering. End with one separate line `FINAL ANSWER: ANSWER`; ANSWER must contain only the "
    "single answer or the ordered separator-delimited answer sequence.\n\n"
)
