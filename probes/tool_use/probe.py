"""
Tool selection probe — pick the best abstract tool for a described task.

Given a task description and a list of available tools with short descriptions,
the model must output the single best tool name. Uses abstract/generic tool
names to test reasoning rather than memorization.

Output: single tool name.
Scoring: exact match = 1.0.
Maps to: frontal lobe / executive function circuits.
"""

from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {
        "task": "Convert a document from English to French.",
        "tools": {
            "summarizer": "Condenses text to key points",
            "translator": "Converts text between languages",
            "formatter": "Adjusts text layout and style",
            "validator": "Checks text for errors",
        },
        "answer": "translator",
    },
    {
        "task": "Reduce a 10-page report to a single paragraph.",
        "tools": {
            "translator": "Converts text between languages",
            "summarizer": "Condenses text to key points",
            "encryptor": "Secures data with encryption",
            "sorter": "Orders items by criteria",
        },
        "answer": "summarizer",
    },
    {
        "task": "Determine the compound interest on a $5000 loan at 4% over 3 years.",
        "tools": {
            "calculator": "Performs mathematical computations",
            "searcher": "Finds information in databases",
            "formatter": "Adjusts text layout and style",
            "compressor": "Reduces data size",
        },
        "answer": "calculator",
    },
    {
        "task": "Find all research papers mentioning 'CRISPR' published in 2024.",
        "tools": {
            "calculator": "Performs mathematical computations",
            "converter": "Changes data between formats",
            "searcher": "Finds information in databases",
            "encryptor": "Secures data with encryption",
        },
        "answer": "searcher",
    },
    {
        "task": "Protect a file containing passwords before sending it over email.",
        "tools": {
            "compressor": "Reduces data size",
            "encryptor": "Secures data with encryption",
            "formatter": "Adjusts text layout and style",
            "searcher": "Finds information in databases",
        },
        "answer": "encryptor",
    },
    {
        "task": "Arrange a list of student records by GPA from highest to lowest.",
        "tools": {
            "filterer": "Selects items matching conditions",
            "sorter": "Orders items by criteria",
            "merger": "Combines multiple data sources",
            "calculator": "Performs mathematical computations",
        },
        "answer": "sorter",
    },
    {
        "task": "Extract only the rows where the status column says 'active'.",
        "tools": {
            "sorter": "Orders items by criteria",
            "filterer": "Selects items matching conditions",
            "converter": "Changes data between formats",
            "summarizer": "Condenses text to key points",
        },
        "answer": "filterer",
    },
    {
        "task": "Check whether an XML configuration file is well-formed and valid.",
        "tools": {
            "formatter": "Adjusts text layout and style",
            "validator": "Checks text for errors",
            "converter": "Changes data between formats",
            "compressor": "Reduces data size",
        },
        "answer": "validator",
    },
]

HARD_ITEMS = [
    {
        "task": "Make a large log file smaller for archival storage.",
        "tools": {
            "compressor": "Reduces data size",
            "summarizer": "Condenses text to key points",
            "formatter": "Adjusts text layout and style",
            "validator": "Checks text for errors",
        },
        "answer": "compressor",
    },
    {
        "task": "Reformat a CSV into a properly indented JSON structure.",
        "tools": {
            "sorter": "Orders items by criteria",
            "validator": "Checks text for errors",
            "formatter": "Adjusts text layout and style",
            "converter": "Changes data between formats",
        },
        "answer": "converter",
    },
    {
        "task": "Combine three separate customer databases into one unified dataset.",
        "tools": {
            "filterer": "Selects items matching conditions",
            "sorter": "Orders items by criteria",
            "merger": "Combines multiple data sources",
            "compressor": "Reduces data size",
        },
        "answer": "merger",
    },
    {
        "task": "Reindent and clean up a messy HTML document for readability.",
        "tools": {
            "validator": "Checks text for errors",
            "converter": "Changes data between formats",
            "formatter": "Adjusts text layout and style",
            "summarizer": "Condenses text to key points",
        },
        "answer": "formatter",
    },
    {
        "task": "A user uploaded a spreadsheet and wants to see only rows from Q4 2024 sorted by revenue.",
        "tools": {
            "filterer": "Selects items matching conditions",
            "sorter": "Orders items by criteria",
            "summarizer": "Condenses text to key points",
            "converter": "Changes data between formats",
        },
        "answer": "filterer",
    },
    {
        "task": "A developer needs to detect if incoming API payloads conform to a JSON schema.",
        "tools": {
            "formatter": "Adjusts text layout and style",
            "converter": "Changes data between formats",
            "validator": "Checks text for errors",
            "compressor": "Reduces data size",
        },
        "answer": "validator",
    },
    {
        "task": "A researcher wants to condense 50 interview transcripts into key themes, then find the original quotes.",
        "tools": {
            "summarizer": "Condenses text to key points",
            "searcher": "Finds information in databases",
            "filterer": "Selects items matching conditions",
            "formatter": "Adjusts text layout and style",
        },
        "answer": "summarizer",
    },
    {
        "task": "Transform a legacy SOAP XML response into a modern REST JSON format while preserving all fields.",
        "tools": {
            "converter": "Changes data between formats",
            "formatter": "Adjusts text layout and style",
            "validator": "Checks text for errors",
            "translator": "Converts text between languages",
        },
        "answer": "converter",
    },
]

# Legacy alias
SCENARIOS = EASY_ITEMS + HARD_ITEMS

PROMPT_TEMPLATE = (
    "Task: {task}\n\n"
    "Available tools:\n{tool_list}\n\n"
    "Select the best tool for this task. Answer with only the tool name."
)


def score_tool_use(response: str, expected: str) -> float:
    """Score tool selection. Exact match = 1.0."""
    response = response.strip().lower()
    expected = expected.lower()
    # Check if the expected tool name appears in the response
    if expected in response:
        return 1.0
    return 0.0


@register_probe
class ToolUseProbe(BaseProbe):
    name = "tool_use"
    description = "Tool selection routing — frontal lobe circuits"

    def run(self, model) -> dict:
        easy_scores = []
        for scenario in EASY_ITEMS:
            tool_list = "\n".join(
                f"  - {name}: {desc}" for name, desc in scenario["tools"].items()
            )
            prompt = PROMPT_TEMPLATE.format(task=scenario["task"], tool_list=tool_list)
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_tool_use(response, scenario["answer"])
            easy_scores.append(score)

        hard_scores = []
        for scenario in HARD_ITEMS:
            tool_list = "\n".join(
                f"  - {name}: {desc}" for name, desc in scenario["tools"].items()
            )
            prompt = PROMPT_TEMPLATE.format(task=scenario["task"], tool_list=tool_list)
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_tool_use(response, scenario["answer"])
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
