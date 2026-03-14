"""
Mock adapter for probe development and unit testing.
Returns canned or random responses without loading a real model.
Use this to develop and test all probes without GPU/model dependency.

Modes:
  - "random":   random plausible outputs
  - "perfect":  correct answers detected from prompt content
  - "terrible": nonsense that should score near 0
"""

import random
import re


class MockAdapter:
    def __init__(self, model_path=None, mode="random", seed=None, num_layers=32):
        self.model_path = model_path  # accepted for API compatibility, ignored
        self.mode = mode
        self.num_layers = num_layers
        self._layer_path = list(range(num_layers))
        self._rng = random.Random(seed)

    def set_layer_path(self, path):
        self._layer_path = path

    def generate_short(self, prompt, max_new_tokens=20, temperature=0.0):
        if self.mode == "perfect":
            return self._perfect_response(prompt)
        if self.mode == "terrible":
            return self._terrible_response(prompt)
        return self._random_response(prompt)

    def get_logprobs(self, prompt, target_tokens=None):
        import math as _math
        tokens = target_tokens or [str(i) for i in range(10)]
        raw = {t: self._rng.random() for t in tokens}
        total = sum(raw.values())
        return {t: _math.log(v / total) for t, v in raw.items()}

    # ------------------------------------------------------------------ #
    #  Mode implementations                                               #
    # ------------------------------------------------------------------ #

    def _terrible_response(self, prompt: str) -> str:
        """Return nonsense that should score ~0 on every probe."""
        options = [
            "banana walrus carpet",
            "!!!???***",
            "I don't understand the question",
            "The answer is purple",
            "42 42 42 42 42",
            "yes no maybe",
            "lorem ipsum dolor sit amet",
        ]
        return self._rng.choice(options)

    def _random_response(self, prompt: str) -> str:
        """Return random but plausible-format responses."""
        p = prompt.lower()
        if "digit" in p or "0-9" in p or "intensity" in p:
            return str(self._rng.randint(0, 9))
        if "coordinate" in p or "battleship" in p or "grid" in p or "next shot" in p:
            col = self._rng.choice("ABCDEFGHIJ")
            row = self._rng.randint(1, 10)
            return f"{col}{row}"
        if "grammatical" in p:
            return self._rng.choice(["grammatical", "ungrammatical"])
        if "tool" in p and ("select" in p or "best" in p or "choose" in p or "pick" in p):
            return "tool_" + str(self._rng.randint(1, 5))
        if "analogy" in p or "is to" in p:
            return self._rng.choice(["word", "thing", "concept"])
        if "order" in p or "steps" in p or "sequence" in p:
            letters = list("ABCDE")
            self._rng.shuffle(letters)
            return "".join(letters[:4])
        if "def " in p or "function" in p:
            return "\n    return x"
        if "answer with only" in p and "number" in p:
            return str(self._rng.randint(0, 1000))
        return str(self._rng.randint(0, 100))

    def _perfect_response(self, prompt: str) -> str:
        """Detect which probe is calling and return the correct answer."""
        p = prompt.lower()

        # --- Math probe ---
        if "what is 17 * 23" in p:
            return "391"
        if "what is 144 / 12" in p:
            return "12"
        if "what is 2^10" in p:
            return "1024"
        if "sum of integers from 1 to 20" in p:
            return "210"
        if "999 - 573" in p:
            return "426"
        if "15! / 14!" in p:
            return "15"
        if "sides 3, 4, 5" in p and "area" in p:
            return "6"
        if "7th prime" in p:
            return "17"
        if "what is 3^5" in p:
            return "243"
        if "pentagon" in p and "interior angle" in p:
            return "108"
        if "gcd of 48 and 36" in p:
            return "12"
        if "256 in base 2" in p:
            return "9"

        # --- Code probe ---
        # Return complete function definitions so score_code's reconstruction works
        if "def double(x):" in prompt:
            return "def double(x):\n    return x * 2"
        if "def is_even(n):" in prompt:
            return "def is_even(n):\n    return n % 2 == 0"
        if "def max_of_three(a,b,c):" in prompt:
            return "def max_of_three(a,b,c):\n    return max(a, b, c)"
        if "def factorial(n):" in prompt:
            return "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)"
        if "def reverse_string(s):" in prompt:
            return "def reverse_string(s):\n    return s[::-1]"
        if "def count_vowels(s):" in prompt:
            return "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"

        # --- EQ probe ---
        # Detect EQ scenarios by emotion keywords and digit request
        if "digit" in p and ("intensity" in p or "0 to 9" in p or "0-9" in p or "0 (none)" in p):
            return self._perfect_eq(prompt)

        # --- Spatial/Battleship probe ---
        if ("battleship" in p or "next shot" in p) and ("coordinate" in p or "grid" in p):
            return self._perfect_spatial(prompt)

        # --- Factual probe ---
        if "how many bones" in p and "adult human" in p:
            return "206"
        if "melting point of tungsten" in p:
            return "3422"
        if "speed of light" in p and "km/s" in p:
            return "299792"
        if "atomic number of gold" in p:
            return "79"
        if "tallest mountain" in p and "solar system" in p:
            return "Olympus"
        if "deepest ocean trench" in p:
            return "Mariana"
        if "half-life of carbon-14" in p:
            return "5730"
        if "chromosomes" in p and "human" in p:
            return "46"
        if "boiling point of nitrogen" in p:
            return "-196"
        if "electron" in p and "kev" in p:
            return "511"
        if "distance" in p and "earth" in p and "moon" in p:
            return "384400"
        if "planck" in p and "constant" in p:
            return "6.626"

        # --- Language probe ---
        if "grammatical" in p and "ungrammatical" in p:
            return self._perfect_language(prompt)

        # --- Tool use probe ---
        if "available tools" in p or "select the best tool" in p or "which tool" in p:
            return self._perfect_tool_use(prompt)

        # --- Holistic/analogy probe ---
        if "is to" in p and ("as" in p or "analogy" in p):
            return self._perfect_holistic(prompt)

        # --- Planning probe ---
        if ("correct order" in p or "reorder" in p or "right sequence" in p) and "steps" in p:
            return self._perfect_planning(prompt)

        # --- Instruction probe ---
        if "constraint" in p or "instruction" in p or "follow these rules" in p or "requirements" in p:
            return self._perfect_instruction(prompt)

        # Fallback
        return "42"

    # ------------------------------------------------------------------ #
    #  Per-probe perfect response helpers                                  #
    # ------------------------------------------------------------------ #

    def _perfect_eq(self, prompt: str) -> str:
        """Return expected digit for EQ scenarios based on prompt content."""
        p = prompt.lower()
        # Map emotion keywords to expected intensity digits
        eq_map = {
            "guilt": "2", "anger": "7", "surprise": "7", "relief": "9",
            "jealousy": "5", "pride": "8", "betrayal": "9", "gratitude": "8",
            "shame": "8", "anxiety": "7", "contentment": "7", "frustration": "8",
        }
        for emotion, digit in eq_map.items():
            if emotion in p:
                return digit
        return "5"

    def _perfect_spatial(self, prompt: str) -> str:
        """Parse the ASCII board, find hits, and suggest an adjacent unknown cell."""
        lines = prompt.split("\n")
        board_lines = []
        for line in lines:
            if line.startswith("|") and line.endswith("|"):
                # Parse row: |. . H H . . . . . .|
                inner = line[1:-1]
                cells = inner.split()
                if len(cells) == 10:
                    board_lines.append(cells)

        if len(board_lines) != 10:
            return "E5"

        cols = "ABCDEFGHIJ"
        # Find hits and look for adjacent unknowns
        for r in range(10):
            for c in range(10):
                if board_lines[r][c] == 'H':
                    # Check adjacent cells for unknowns
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 10 and 0 <= nc < 10 and board_lines[nr][nc] == '.':
                            return f"{cols[nc]}{nr + 1}"
        # No hits found — pick center unknown
        return "E5"

    def _perfect_language(self, prompt: str) -> str:
        """Detect grammaticality from the sentence in the prompt."""
        p = prompt.lower()
        # The probe will include the sentence and we detect from known items
        # We look for known ungrammatical patterns
        ungrammatical_markers = [
            "the children plays",
            "him went",
            "the dog that the cat that the rat bit chased died",
            "she don't",
            "they was",
            "a books",
            "he goed",
            "more taller",
        ]
        for marker in ungrammatical_markers:
            if marker in p:
                return "ungrammatical"
        return "grammatical"

    def _perfect_tool_use(self, prompt: str) -> str:
        """Return the correct tool name based on task description keywords."""
        p = prompt.lower()
        # Extract just the task line (before "Available tools:")
        task = p.split("available tools")[0] if "available tools" in p else p
        # Map task keywords to correct tool answers
        task_map = [
            (["english to french", "convert", "language", "translate"], "translator"),
            (["reduce", "report", "paragraph", "condense", "summarize"], "summarizer"),
            (["interest", "compound", "loan", "calculate", "math"], "calculator"),
            (["find", "research", "papers", "search", "mentioning"], "searcher"),
            (["smaller", "archival", "log file", "compress", "reduce data"], "compressor"),
            (["protect", "password", "encrypt", "secure"], "encryptor"),
            (["csv", "json", "reformat", "convert"], "converter"),
            (["arrange", "gpa", "highest to lowest", "order", "sort"], "sorter"),
            (["extract", "status", "active", "filter", "rows where"], "filterer"),
            (["combine", "databases", "unified", "merge"], "merger"),
            (["well-formed", "valid", "xml", "check", "validate"], "validator"),
            (["reindent", "clean up", "messy", "readability", "format"], "formatter"),
        ]
        for keywords, tool in task_map:
            if any(kw in task for kw in keywords):
                return tool
        return "calculator"

    def _perfect_holistic(self, prompt: str) -> str:
        """Return correct analogy answer based on prompt content."""
        p = prompt.lower()
        analogy_map = {
            "painter": "composer",
            "fish": "bird",
            "eye": "ear",
            "glove": "sock",
            "library": "museum",
            "captain": "conductor",
            "sword": "pen",
            "winter": "night",
            "architect": "playwright",
            "telescope": "microscope",
            "oasis": "island",
            "key": "password",
        }
        for keyword, answer in analogy_map.items():
            if keyword in p:
                return answer
        return "unknown"

    def _perfect_planning(self, prompt: str) -> str:
        """Return the correct step ordering based on prompt content."""
        p = prompt.lower()
        # Each planning scenario will embed the correct order as metadata
        # For mock: extract from the known scenarios
        planning_answers = {
            "breakfast": "CADB",
            "flat tire": "BDCA",
            "website": "ADBC",
            "move to a new apartment": "BADC",
            "plant a garden": "DCAB",
            "research paper": "ABCD",
            "build a campfire": "BCDA",
            "job interview": "ABDC",
            "bake a cake": "ABCD",
            "first aid": "BADC",
            "machine learning": "ABCDE",
            "mobile app": "ABCDE",
            "aquarium": "ABCDE",
        }
        for keyword, order in planning_answers.items():
            if keyword in p:
                return order
        return "ABCD"

    def _perfect_instruction(self, prompt: str) -> str:
        """Return a response that satisfies all constraints in the prompt."""
        p = prompt.lower()
        # Match against actual instruction probe scenarios
        # Conflicting-constraint scenarios (lowercase, no spaces, etc.)
        if "lowercase" in p and "question mark" in p and "3 words" in p:
            return "what is 7?"
        if "no spaces" in p and "number" in p and "letter" in p:
            return "abc42."
        if "lowercase" in p and "1 word" in p and "number" in p:
            return "go4it!"
        if "exactly 3 words" in p and "uppercase" in p and "number" in p:
            return "HELLO WORLD 42"
        if "starts with" in p and "letter z" in p:
            return "ZEPHYR 99!"
        if "no vowels" in p:
            return "RHYTHMS 7!"
        if "palindrome" in p:
            return "RACECAR 5!"
        if "color" in p and "animal" in p:
            return "RED FOX 42"
        if "alliteration" in p:
            return "PETER PIPER PICKED"
        if "ends with a period" in p and "exactly 4 words" in p:
            return "HELLO WORLD NUMBER 7."
        if "exactly 2 words" in p and "same letter" in p:
            return "SUPER 7 STAR."
        if "starts with" in p and "letter a" in p:
            return "AWESOME 42!"
        if "uppercase" in p and "number" in p and "exclamation" in p:
            return "HELLO WORLD 42!"
        return "HELLO WORLD 42!"
