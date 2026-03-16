"""
Planning probe — order steps to achieve a goal.

Given a goal and a set of unordered steps (labeled A-D or A-E), the model must
output the correct letter sequence. Scored by pairwise ordering correctness:
each correctly ordered adjacent pair earns 1/n_pairs.

Output: ordered letter sequence (e.g., "BDAC").
Scoring: fraction of correctly ordered adjacent pairs.
Maps to: prefrontal executive / planning circuits.
"""

import re
from probes.registry import BaseProbe, register_probe

EASY_ITEMS = [
    {
        "goal": "Make breakfast (scrambled eggs and toast)",
        "steps": {
            "A": "Crack eggs into a bowl and whisk them",
            "B": "Put bread in the toaster",
            "C": "Heat butter in a pan",
            "D": "Pour eggs into the hot pan and stir",
        },
        "correct_order": "CADB",
    },
    {
        "goal": "Change a flat tire",
        "steps": {
            "A": "Lower the jack and tighten the lug nuts fully",
            "B": "Loosen the lug nuts slightly while the car is on the ground",
            "C": "Remove the flat tire and mount the spare",
            "D": "Jack up the car until the flat tire is off the ground",
        },
        "correct_order": "BDCA",
    },
    {
        "goal": "Launch a basic website",
        "steps": {
            "A": "Register a domain name",
            "B": "Write the HTML/CSS content",
            "C": "Upload files to the hosting server",
            "D": "Purchase web hosting",
        },
        "correct_order": "ADBC",
    },
    {
        "goal": "Build a campfire",
        "steps": {
            "A": "Add larger logs once the fire is established",
            "B": "Clear a safe area and create a fire ring",
            "C": "Arrange tinder and kindling in the fire ring",
            "D": "Light the tinder with a match",
        },
        "correct_order": "BCDA",
    },
    {
        "goal": "Administer first aid for a deep cut",
        "steps": {
            "A": "Apply direct pressure with a clean cloth",
            "B": "Wash hands or put on gloves",
            "C": "Bandage the wound securely",
            "D": "Clean the wound with water once bleeding slows",
        },
        "correct_order": "BADC",
    },
    {
        "goal": "Move to a new apartment",
        "steps": {
            "A": "Pack all belongings into boxes",
            "B": "Sign the lease for the new apartment",
            "C": "Unpack and arrange furniture",
            "D": "Hire movers and transport everything",
        },
        "correct_order": "BADC",
    },
    {
        "goal": "Bake a chocolate cake",
        "steps": {
            "A": "Preheat the oven to the correct temperature",
            "B": "Mix dry and wet ingredients together",
            "C": "Pour batter into a greased pan and bake",
            "D": "Let the cake cool before frosting",
        },
        "correct_order": "ABCD",
    },
    {
        "goal": "Prepare for a job interview",
        "steps": {
            "A": "Research the company and role",
            "B": "Practice answering common interview questions",
            "C": "Arrive 10 minutes early at the location",
            "D": "Choose and prepare professional clothing",
        },
        "correct_order": "ABDC",
    },
]

HARD_ITEMS = [
    {
        "goal": "Deploy a machine learning model to production",
        "steps": {
            "A": "Collect and label training data",
            "B": "Train the model and tune hyperparameters",
            "C": "Evaluate model on held-out test set",
            "D": "Package model into a Docker container",
            "E": "Run integration tests in staging environment",
        },
        "correct_order": "ABCDE",
    },
    {
        "goal": "Publish a mobile app to an app store",
        "steps": {
            "A": "Create developer account and signing certificates",
            "B": "Implement core features and fix critical bugs",
            "C": "Run automated test suite and fix failures",
            "D": "Build release binary with production signing",
            "E": "Submit build for app store review",
        },
        "correct_order": "ABCDE",
    },
    {
        "goal": "Set up a home aquarium with live fish",
        "steps": {
            "A": "Assemble the tank, filter, and heater",
            "B": "Fill tank with dechlorinated water",
            "C": "Run the nitrogen cycle for 4-6 weeks",
            "D": "Test water parameters (ammonia, nitrite, nitrate)",
            "E": "Introduce fish slowly over several days",
        },
        "correct_order": "ABCDE",
    },
    {
        "goal": "Organize a community fundraiser event",
        "steps": {
            "A": "Secure a venue and date",
            "B": "Obtain necessary permits and insurance",
            "C": "Recruit volunteers and assign roles",
            "D": "Promote the event through local media",
            "E": "Set up the venue and run the event",
        },
        "correct_order": "ABCDE",
    },
    {
        "goal": "Restore a vintage car engine",
        "steps": {
            "A": "Document and photograph the engine before disassembly",
            "B": "Remove the engine from the vehicle",
            "C": "Clean, inspect, and machine worn parts",
            "D": "Reassemble with new gaskets and seals",
            "E": "Reinstall engine and perform break-in procedure",
        },
        "correct_order": "ABCDE",
    },
    {
        "goal": "Set up a home solar power system",
        "steps": {
            "A": "Get a structural assessment of the roof",
            "B": "Design the system and obtain permits",
            "C": "Install mounting hardware and panels",
            "D": "Wire panels to inverter and electrical panel",
            "E": "Pass inspection and connect to the grid",
        },
        "correct_order": "ABCDE",
    },
    {
        "goal": "Produce a short documentary film",
        "steps": {
            "A": "Research the subject and write a treatment",
            "B": "Secure funding and equipment",
            "C": "Conduct interviews and film B-roll",
            "D": "Edit footage and add music/narration",
            "E": "Submit to film festivals and distribute",
        },
        "correct_order": "ABCDE",
    },
    {
        "goal": "Launch a new product line for a small business",
        "steps": {
            "A": "Conduct market research and identify target audience",
            "B": "Develop prototypes and test with focus groups",
            "C": "Finalize design and set up manufacturing",
            "D": "Create marketing materials and pricing strategy",
            "E": "Launch product and monitor initial sales feedback",
        },
        "correct_order": "ABCDE",
    },
]

# Legacy alias
SCENARIOS = EASY_ITEMS + HARD_ITEMS

PROMPT_TEMPLATE = (
    "Goal: {goal}\n\n"
    "Steps (in random order):\n{step_list}\n\n"
    "Reorder these steps into the correct sequence. "
    "Answer with only the letters in the right order (e.g., \"BDAC\"). No explanation."
)


def score_planning(response: str, correct_order: str) -> float:
    """
    Score step ordering by fraction of correctly ordered adjacent pairs.

    For a sequence of length L, there are L-1 adjacent pairs.
    Each pair (x, y) is correct if x appears before y in the correct order.
    """
    import re as _re

    response_stripped = response.strip()
    valid_labels = set(correct_order)

    # Only accept responses that look like step orderings:
    # A compact sequence of letters (possibly with commas/spaces/arrows)
    # Reject natural language prose that happens to contain step letters
    match = _re.search(r'\b([A-E](?:[,\s>→\-]*[A-E]){1,})\b', response_stripped.upper())
    if not match:
        # Try: response is just the letters (e.g. "ABCDE")
        clean = _re.sub(r'[^A-E]', '', response_stripped.upper())
        if len(clean) < 2 or len(clean) > len(correct_order) + 2:
            return 0.0
        # Only accept if >50% of the response chars are step labels
        alpha_count = sum(1 for ch in response_stripped if ch.isalpha())
        label_count = sum(1 for ch in response_stripped.upper() if ch in valid_labels)
        if alpha_count > 0 and label_count / alpha_count < 0.5:
            return 0.0
        letters = [ch for ch in clean if ch in valid_labels]
    else:
        letters = [ch for ch in match.group() if ch in valid_labels]

    if len(letters) < 2:
        return 0.0

    # Remove duplicates while preserving order
    seen = set()
    unique_letters = []
    for ch in letters:
        if ch not in seen:
            seen.add(ch)
            unique_letters.append(ch)
    letters = unique_letters

    if len(letters) < 2:
        return 0.0

    # Build position map from correct order
    correct_pos = {ch: idx for idx, ch in enumerate(correct_order)}

    # Count correctly ordered adjacent pairs in the response
    n_pairs = len(letters) - 1
    correct_pairs = 0
    for k in range(n_pairs):
        if correct_pos.get(letters[k], -1) < correct_pos.get(letters[k + 1], -1):
            correct_pairs += 1

    return correct_pairs / n_pairs


@register_probe
class PlanningProbe(BaseProbe):
    name = "planning"
    description = "Step ordering / planning — prefrontal executive circuits"

    def run(self, model) -> dict:
        easy_scores = []
        for scenario in self._limit(EASY_ITEMS):
            step_list = "\n".join(
                f"  {label}: {desc}" for label, desc in scenario["steps"].items()
            )
            prompt = PROMPT_TEMPLATE.format(
                goal=scenario["goal"], step_list=step_list
            )
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_planning(response, scenario["correct_order"])
            easy_scores.append(score)

        hard_scores = []
        for scenario in self._limit(HARD_ITEMS):
            step_list = "\n".join(
                f"  {label}: {desc}" for label, desc in scenario["steps"].items()
            )
            prompt = PROMPT_TEMPLATE.format(
                goal=scenario["goal"], step_list=step_list
            )
            response = model.generate_short(prompt, max_new_tokens=10, temperature=0.0)
            score = score_planning(response, scenario["correct_order"])
            hard_scores.append(score)

        return self._make_result(easy_scores, hard_scores)
