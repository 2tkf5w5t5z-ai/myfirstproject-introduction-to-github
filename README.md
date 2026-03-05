# Medical Conversation Tokenizer

A lightweight Python tool to parse senior-doctor conversation transcripts. It identifies speakers, splits text into sentences/tokens, and flags medical terms.

## ✨ Features
- **Speaker Detection**: Identifies roles (Doctor, Senior, Patient, etc.).
- **Tokenization**: Splits text into sentences and clean word tokens.
- **Medical Flagging**: Highlights single-word terms (e.g., `fever`) and phrases (e.g., `blood pressure`).
- **Zero Dependencies**: Uses only the Python standard library.

## 🚀 Usage

```python
from tokenizer import tokenize_conversation

text = """
Doctor: How is your headache?
Senior: It hurts, and I feel dizzy.
Doctor: Let's check your blood pressure.
"""

for turn in tokenize_conversation(text):
    print(f"[{turn['speaker']}] {turn['utterance']}")
    
    # Show detected medical keywords
    terms = [t for t, is_med in turn['medical'].items() if is_med]
    phrases = turn['medical_phrases']
    
    if terms: print(f"  🔍 Terms: {', '.join(terms)}")
    if phrases: print(f"  💊 Phrases: {', '.join(phrases)}")
