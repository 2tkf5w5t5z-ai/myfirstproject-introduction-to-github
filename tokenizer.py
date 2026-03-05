"""
Simple tokenizer for a senior's conversation with a doctor.

Splits a conversation transcript into speaker turns, sentences, and word tokens.
Medical-related terms are flagged for easy identification.
"""

import re

# Single-word medical terms matched against individual tokens.
MEDICAL_TERMS = {
    "pulse", "medication", "prescription", "diagnosis", "symptom", "symptoms",
    "pain", "fever", "nausea", "fatigue", "dizziness", "headache", "diabetes",
    "hypertension", "cholesterol", "arthritis", "dementia", "alzheimers",
    "dosage", "dose", "refill", "pharmacy", "hospital", "clinic", "appointment",
    "vaccine", "injection", "surgery", "therapy", "rehabilitation",
    "allergies", "allergy",
}

# Multi-word medical phrases matched against the full utterance string.
MEDICAL_PHRASES = {
    "blood pressure", "heart rate", "chest pain", "shortness of breath",
    "blood test", "urine test", "follow-up", "x-ray", "lab test",
}

SPEAKER_PATTERN = re.compile(
    r"^(doctor|dr\.?|physician|senior|patient|elderly|grandma|grandpa)\s*:",
    re.IGNORECASE,
)

SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+")


def tokenize_words(text: str) -> list[str]:
    """Split *text* into word tokens (lowercase, punctuation stripped)."""
    return [token for token in re.split(r"\W+", text.lower()) if token]


def tokenize_sentences(text: str) -> list[str]:
    """Split *text* into individual sentences."""
    return [s.strip() for s in SENTENCE_SPLITTER.split(text) if s.strip()]


def detect_speaker(line: str) -> tuple[str | None, str]:
    """Return *(speaker, utterance)* extracted from a transcript line.

    If no speaker label is found, *speaker* is ``None`` and the full line is
    returned as the utterance.
    """
    match = SPEAKER_PATTERN.match(line.strip())
    if match:
        speaker = match.group(1).rstrip(".").capitalize()
        utterance = line[match.end():].strip()
        return speaker, utterance
    return None, line.strip()


def flag_medical_terms(tokens: list[str]) -> dict[str, bool]:
    """Return a mapping of each token to whether it is a known medical term.

    Tokens are expected to be lowercase (as produced by :func:`tokenize_words`).
    """
    return {token: token in MEDICAL_TERMS for token in tokens}


def detect_medical_phrases(utterance: str) -> list[str]:
    """Return multi-word medical phrases found in *utterance*.

    The search is case-insensitive and matches phrases in :data:`MEDICAL_PHRASES`.
    """
    lower = utterance.lower()
    return [phrase for phrase in MEDICAL_PHRASES if phrase in lower]


def tokenize_conversation(transcript: str) -> list[dict]:
    """Tokenize a full conversation transcript.

    Parameters
    ----------
    transcript:
        Multi-line string where each line may begin with a speaker label
        such as ``"Doctor:"`` or ``"Senior:"``.

    Returns
    -------
    list of dicts, one per non-empty line, each containing:
      - ``speaker``          – identified speaker or ``None``
      - ``utterance``        – the spoken text
      - ``sentences``        – list of sentences in the utterance
      - ``tokens``           – word-level tokens
      - ``medical``          – dict mapping each token to ``True`` if it is a
                               recognised single-word medical term
      - ``medical_phrases``  – list of recognised multi-word medical phrases
    """
    results = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker, utterance = detect_speaker(line)
        sentences = tokenize_sentences(utterance)
        tokens = tokenize_words(utterance)
        medical = flag_medical_terms(tokens)
        phrases = detect_medical_phrases(utterance)
        results.append(
            {
                "speaker": speaker,
                "utterance": utterance,
                "sentences": sentences,
                "tokens": tokens,
                "medical": medical,
                "medical_phrases": phrases,
            }
        )
    return results


if __name__ == "__main__":
    sample = """
Doctor: Good morning! How are you feeling today?
Senior: Good morning, Doctor. I have been having a headache and some dizziness.
Doctor: I see. Any chest pain or shortness of breath?
Senior: No chest pain, but I feel quite fatigued in the afternoons.
Doctor: Let's check your blood pressure and run a quick blood test to be sure.
Senior: Should I bring my current medication list?
Doctor: Yes, please bring your prescription bottles to the next appointment.
"""
    for turn in tokenize_conversation(sample):
        speaker_label = turn["speaker"] or "Unknown"
        flagged = [t for t, is_med in turn["medical"].items() if is_med]
        print(f"[{speaker_label}] {turn['utterance']}")
        if flagged:
            print(f"  Medical terms:   {', '.join(flagged)}")
        if turn["medical_phrases"]:
            print(f"  Medical phrases: {', '.join(turn['medical_phrases'])}")
        print()
