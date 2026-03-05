# main.py
from tokenizer import tokenize_conversation

if __name__ == "__main__":
    sample = """
Doctor: How are you feeling today?
Senior: I have a headache and dizziness.
Doctor: Let's check your blood pressure.
"""
    for turn in tokenize_conversation(sample):
        print(f"[{turn['speaker']}] {turn['utterance']}")
        terms = [t for t, is_med in turn['medical'].items() if is_med]
        phrases = turn['medical_phrases']
        if terms: print(f"  🔍 Terms: {', '.join(terms)}")
        if phrases: print(f"  💊 Phrases: {', '.join(phrases)}")
        print()
