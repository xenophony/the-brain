"""
Calibration questions for logprob probe discovery.
Run against API to find questions with 40-70% accuracy (sweet spot for circuit mapping).

120 questions across 10 cognitive domains, 12 each.
Mix of easy (~80% expected accuracy) and hard (~20%) to land domains around 40-70%.
"""

DOMAINS = {
    "factual_recall": {
        "description": "Memory retrieval — hippocampal circuits",
        "answer_choices": None,  # free single word
        "items": [
            # Easy (well-known facts)
            {"prompt": "Capital of Japan?", "answer": "tokyo", "difficulty": "easy"},
            {"prompt": "Chemical symbol for gold?", "answer": "au", "difficulty": "easy"},
            {"prompt": "Largest planet in our solar system?", "answer": "jupiter", "difficulty": "easy"},
            {"prompt": "Who wrote Romeo and Juliet?", "answer": "shakespeare", "difficulty": "easy"},
            {"prompt": "What gas do plants absorb from the atmosphere?", "answer": "carbon dioxide", "difficulty": "easy"},
            {"prompt": "How many continents are there?", "answer": "7", "difficulty": "easy"},
            # Hard (obscure facts)
            {"prompt": "Chemical symbol for tungsten?", "answer": "w", "difficulty": "hard"},
            {"prompt": "What element has atomic number 76?", "answer": "osmium", "difficulty": "hard"},
            {"prompt": "Capital of Burkina Faso?", "answer": "ouagadougou", "difficulty": "hard"},
            {"prompt": "What is the smallest bone in the human body?", "answer": "stapes", "difficulty": "hard"},
            {"prompt": "Who was the second person to walk on the moon?", "answer": "aldrin", "difficulty": "hard"},
            {"prompt": "What is the only letter not appearing in any US state name?", "answer": "q", "difficulty": "hard"},
        ],
    },
    "logical_deduction": {
        "description": "Syllogistic reasoning — prefrontal circuits",
        "answer_choices": ["yes", "no"],
        "items": [
            # Easy (straightforward syllogisms)
            {"prompt": "All dogs are animals. Rex is a dog. Is Rex an animal?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "All cats have tails. Whiskers is a cat. Does Whiskers have a tail?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "No fish can fly. A salmon is a fish. Can a salmon fly?", "answer": "no", "difficulty": "easy"},
            {"prompt": "All squares are rectangles. Is every square a rectangle?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "If it is raining, the ground is wet. It is raining. Is the ground wet?", "answer": "yes", "difficulty": "easy"},
            # Hard (invalid syllogisms, tricky negation, contrapositive)
            {"prompt": "All roses are flowers. Some flowers fade quickly. Do all roses fade quickly?", "answer": "no", "difficulty": "hard"},
            {"prompt": "All birds have feathers. A penguin has feathers. Is a penguin definitely a bird based only on this?", "answer": "no", "difficulty": "hard"},
            {"prompt": "No reptiles are mammals. Some pets are mammals. Are no pets reptiles?", "answer": "no", "difficulty": "hard"},
            {"prompt": "If it snows, schools close. Schools are closed. Did it snow?", "answer": "no", "difficulty": "hard"},
            {"prompt": "All A are B. All B are C. No C are D. Can any A be D?", "answer": "no", "difficulty": "hard"},
            {"prompt": "Some dogs are large. Some large things are dangerous. Are some dogs dangerous?", "answer": "no", "difficulty": "hard"},
            {"prompt": "If P then Q. If Q then R. Not R. Is P true?", "answer": "no", "difficulty": "hard"},
        ],
    },
    "numerical_comparison": {
        "description": "Mathematical/numerical circuits",
        "answer_choices": ["a", "b"],
        "items": [
            # Easy (obvious comparisons)
            {"prompt": "Which is larger: (A) 100 or (B) 50?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which is larger: (A) 3.14 or (B) 2.71?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which is larger: (A) 1000 or (B) 999?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which is larger: (A) 0.5 or (B) 0.25?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which is larger: (A) -1 or (B) -10?", "answer": "a", "difficulty": "easy"},
            # Hard (fractions, close values, negatives, counterintuitive)
            {"prompt": "Which is larger: (A) 3/7 or (B) 5/12?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which is larger: (A) 7/11 or (B) 5/8?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which is larger: (A) 0.99^100 or (B) 0.5?", "answer": "b", "difficulty": "hard"},
            {"prompt": "Which is larger: (A) sqrt(2) + sqrt(3) or (B) sqrt(10)?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which is larger: (A) 2^10 or (B) 10^3?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which is larger: (A) -3^2 or (B) (-3)^2?", "answer": "b", "difficulty": "hard"},
            {"prompt": "Which is larger: (A) 99/100 or (B) 999/1000?", "answer": "b", "difficulty": "hard"},
        ],
    },
    "causal_reasoning": {
        "description": "World model / physics — causal circuits",
        "answer_choices": ["yes", "no"],
        "items": [
            # Easy (obvious cause-effect)
            {"prompt": "You drop a glass on concrete. Will it likely break?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "You leave ice cream in the sun. Will it melt?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "You water a plant regularly. Will it likely grow?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "You unplug a lamp. Will it stay on?", "answer": "no", "difficulty": "easy"},
            {"prompt": "A ball is thrown upward. Will it eventually come back down?", "answer": "yes", "difficulty": "easy"},
            # Hard (counterintuitive, subtle, common misconceptions)
            {"prompt": "A heavier object falls faster than a lighter one in a vacuum. True?", "answer": "no", "difficulty": "hard"},
            {"prompt": "Adding salt to water raises its boiling point. True?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "A person standing still in a moving train is moving relative to the ground. True?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "Hot water freezes faster than cold water under certain conditions. True?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "If you spin a coin that lands heads 10 times in a row, is the next flip more likely to be tails?", "answer": "no", "difficulty": "hard"},
            {"prompt": "A mirror reverses left and right but not up and down. True?", "answer": "no", "difficulty": "hard"},
            {"prompt": "Lightning never strikes the same place twice. True?", "answer": "no", "difficulty": "hard"},
        ],
    },
    "sentiment": {
        "description": "Social/emotional tone — limbic circuits",
        "answer_choices": ["positive", "negative"],
        "items": [
            # Easy (obvious sentiment)
            {"prompt": "I love this beautiful sunny day!", "answer": "positive", "difficulty": "easy"},
            {"prompt": "This is the worst experience of my life.", "answer": "negative", "difficulty": "easy"},
            {"prompt": "What a wonderful surprise!", "answer": "positive", "difficulty": "easy"},
            {"prompt": "I'm so grateful for your help.", "answer": "positive", "difficulty": "easy"},
            {"prompt": "This product is terrible and broke immediately.", "answer": "negative", "difficulty": "easy"},
            {"prompt": "I'm devastated by the news.", "answer": "negative", "difficulty": "easy"},
            # Hard (sarcasm, irony, ambiguity, mixed signals)
            {"prompt": "Oh great, another meeting that could have been an email.", "answer": "negative", "difficulty": "hard"},
            {"prompt": "Well, that went exactly as planned. [said after a disaster]", "answer": "negative", "difficulty": "hard"},
            {"prompt": "I'm not unhappy with the results.", "answer": "positive", "difficulty": "hard"},
            {"prompt": "Sure, because what I really needed was more work.", "answer": "negative", "difficulty": "hard"},
            {"prompt": "The surgery was successful but the recovery will be painful.", "answer": "positive", "difficulty": "hard"},
            {"prompt": "At least it can't get any worse.", "answer": "negative", "difficulty": "hard"},
        ],
    },
    "semantic_opposites": {
        "description": "Antonym retrieval — association cortex circuits",
        "answer_choices": None,  # free single word
        "items": [
            # Easy (common opposites)
            {"prompt": "Opposite of hot?", "answer": "cold", "difficulty": "easy"},
            {"prompt": "Opposite of big?", "answer": "small", "difficulty": "easy"},
            {"prompt": "Opposite of fast?", "answer": "slow", "difficulty": "easy"},
            {"prompt": "Opposite of happy?", "answer": "sad", "difficulty": "easy"},
            {"prompt": "Opposite of light?", "answer": "dark", "difficulty": "easy"},
            {"prompt": "Opposite of up?", "answer": "down", "difficulty": "easy"},
            # Hard (abstract, uncommon, multiple valid but one canonical)
            {"prompt": "Opposite of entropy?", "answer": "order", "difficulty": "hard"},
            {"prompt": "Opposite of ephemeral?", "answer": "permanent", "difficulty": "hard"},
            {"prompt": "Opposite of zenith?", "answer": "nadir", "difficulty": "hard"},
            {"prompt": "Opposite of verbose?", "answer": "concise", "difficulty": "hard"},
            {"prompt": "Opposite of prosaic?", "answer": "poetic", "difficulty": "hard"},
            {"prompt": "Opposite of nascent?", "answer": "dying", "difficulty": "hard"},
        ],
    },
    "categorization": {
        "description": "Taxonomic knowledge — categorization circuits",
        "answer_choices": ["yes", "no"],
        "items": [
            # Easy (obvious category membership)
            {"prompt": "Is a dog a mammal?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "Is the sun a planet?", "answer": "no", "difficulty": "easy"},
            {"prompt": "Is a carrot a vegetable?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "Is water an element?", "answer": "no", "difficulty": "easy"},
            {"prompt": "Is a violin a string instrument?", "answer": "yes", "difficulty": "easy"},
            # Hard (edge cases, common misconceptions, tricky taxonomy)
            {"prompt": "Is a penguin a bird?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "Is a whale a fish?", "answer": "no", "difficulty": "hard"},
            {"prompt": "Is a tomato a fruit?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "Is a strawberry a berry in botanical terms?", "answer": "no", "difficulty": "hard"},
            {"prompt": "Is Pluto a planet?", "answer": "no", "difficulty": "hard"},
            {"prompt": "Is a banana a berry in botanical terms?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "Is glass a liquid?", "answer": "no", "difficulty": "hard"},
        ],
    },
    "temporal_ordering": {
        "description": "Sequence knowledge — temporal/planning circuits",
        "answer_choices": ["a", "b"],
        "items": [
            # Easy (well-known historical order)
            {"prompt": "Which came first: (A) World War I or (B) World War II?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which came first: (A) invention of the telephone or (B) invention of the internet?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which came first: (A) Ancient Egypt or (B) the Roman Empire?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which came first: (A) the Renaissance or (B) the Industrial Revolution?", "answer": "a", "difficulty": "easy"},
            {"prompt": "Which came first: (A) the Moon landing or (B) the fall of the Berlin Wall?", "answer": "a", "difficulty": "easy"},
            # Hard (close dates, counterintuitive, obscure)
            {"prompt": "Which came first: (A) Oxford University founded or (B) Aztec Empire founded?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which came first: (A) the fax machine patent or (B) the telephone patent?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which came first: (A) Harvard University or (B) calculus?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which came first: (A) Nintendo founded or (B) the Eiffel Tower built?", "answer": "a", "difficulty": "hard"},
            {"prompt": "Which came first: (A) the samurai abolished in Japan or (B) the American Civil War ended?", "answer": "b", "difficulty": "hard"},
            {"prompt": "Which came first: (A) the guillotine last used in France or (B) Star Wars released?", "answer": "b", "difficulty": "hard"},
            {"prompt": "Which came first: (A) Cleopatra or (B) the construction of the Great Pyramid of Giza?", "answer": "b", "difficulty": "hard"},
        ],
    },
    "sarcasm_detection": {
        "description": "Theory of mind / pragmatics — social inference circuits",
        "answer_choices": ["yes", "no"],  # yes = sarcastic, no = sincere
        "items": [
            # Easy (obvious sarcasm or sincerity)
            {"prompt": "After burning dinner: 'Well, I'm clearly a master chef.' Is this sarcastic?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "'Thank you so much for helping me move, I really appreciate it.' Is this sarcastic?", "answer": "no", "difficulty": "easy"},
            {"prompt": "After getting a flat tire in the rain: 'What a perfect day!' Is this sarcastic?", "answer": "yes", "difficulty": "easy"},
            {"prompt": "'Your presentation was really well-organized and clear.' Is this sarcastic?", "answer": "no", "difficulty": "easy"},
            {"prompt": "Stuck in traffic: 'Oh wonderful, just how I wanted to spend my evening.' Is this sarcastic?", "answer": "yes", "difficulty": "easy"},
            # Hard (ambiguous, subtle, context-dependent)
            {"prompt": "'Nice weather we're having.' (said on a mild, overcast day) Is this sarcastic?", "answer": "no", "difficulty": "hard"},
            {"prompt": "'You're so smart.' (said after someone solves a hard problem) Is this sarcastic?", "answer": "no", "difficulty": "hard"},
            {"prompt": "'Thanks for letting me know.' (said after receiving bad news) Is this sarcastic?", "answer": "no", "difficulty": "hard"},
            {"prompt": "'That's an interesting choice.' (said about someone's outfit) Is this sarcastic?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "'Sure, take your time.' (said to someone who is late) Is this sarcastic?", "answer": "yes", "difficulty": "hard"},
            {"prompt": "'Well, that was educational.' (said after a failed experiment) Is this sarcastic?", "answer": "no", "difficulty": "hard"},
            {"prompt": "'I love how you always remember everything.' (said after someone forgets again) Is this sarcastic?", "answer": "yes", "difficulty": "hard"},
        ],
    },
    "error_detection": {
        "description": "Self-monitoring / verification — metacognition circuits",
        "answer_choices": ["correct", "incorrect"],
        "items": [
            # Easy (obvious errors)
            {"prompt": "2 + 2 = 5", "answer": "incorrect", "difficulty": "easy"},
            {"prompt": "The Earth orbits the Sun.", "answer": "correct", "difficulty": "easy"},
            {"prompt": "There are 24 hours in a day.", "answer": "correct", "difficulty": "easy"},
            {"prompt": "The capital of Australia is Sydney.", "answer": "incorrect", "difficulty": "easy"},
            {"prompt": "Humans have 206 bones.", "answer": "correct", "difficulty": "easy"},
            # Hard (subtle errors, near-misses, off-by-one)
            {"prompt": "The speed of light is approximately 3 x 10^9 m/s.", "answer": "incorrect", "difficulty": "hard"},
            {"prompt": "There are 52 weeks in a year.", "answer": "correct", "difficulty": "hard"},
            {"prompt": "A century is 1000 years.", "answer": "incorrect", "difficulty": "hard"},
            {"prompt": "The human body is approximately 60% water.", "answer": "correct", "difficulty": "hard"},
            {"prompt": "Abraham Lincoln was the 15th president of the United States.", "answer": "incorrect", "difficulty": "hard"},
            {"prompt": "Pi rounded to two decimal places is 3.15.", "answer": "incorrect", "difficulty": "hard"},
            {"prompt": "The Great Wall of China is visible from space with the naked eye.", "answer": "incorrect", "difficulty": "hard"},
        ],
    },
}


def get_all_items():
    """Yield (domain, item) tuples for all questions."""
    for domain_name, domain in DOMAINS.items():
        for item in domain["items"]:
            yield domain_name, item


def get_domain_items(domain_name):
    """Get all items for a specific domain."""
    return DOMAINS[domain_name]["items"]


def summary():
    """Print a summary of question counts and difficulty distribution."""
    total = 0
    for name, domain in DOMAINS.items():
        items = domain["items"]
        easy = sum(1 for i in items if i["difficulty"] == "easy")
        hard = sum(1 for i in items if i["difficulty"] == "hard")
        print(f"  {name:25s}  easy={easy}  hard={hard}  total={len(items)}")
        total += len(items)
    print(f"  {'TOTAL':25s}  {total}")


if __name__ == "__main__":
    summary()
