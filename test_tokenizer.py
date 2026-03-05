"""Tests for the doctor–senior conversation tokenizer."""

import pytest
from tokenizer import (
    tokenize_words,
    tokenize_sentences,
    detect_speaker,
    flag_medical_terms,
    detect_medical_phrases,
    tokenize_conversation,
)


class TestTokenizeWords:
    def test_simple_sentence(self):
        assert tokenize_words("Hello world") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenize_words("Pain, fever, and nausea.") == [
            "pain",
            "fever",
            "and",
            "nausea",
        ]

    def test_empty_string(self):
        assert tokenize_words("") == []

    def test_numbers_kept(self):
        tokens = tokenize_words("Take 2 tablets daily.")
        assert "2" in tokens
        assert "tablets" in tokens


class TestTokenizeSentences:
    def test_single_sentence(self):
        assert tokenize_sentences("I have a headache.") == ["I have a headache."]

    def test_multiple_sentences(self):
        result = tokenize_sentences("I feel dizzy. My head hurts.")
        assert result == ["I feel dizzy.", "My head hurts."]

    def test_question(self):
        result = tokenize_sentences("Are you in pain? Please describe it.")
        assert result == ["Are you in pain?", "Please describe it."]

    def test_empty_string(self):
        assert tokenize_sentences("") == []


class TestDetectSpeaker:
    def test_doctor_label(self):
        speaker, utterance = detect_speaker("Doctor: How are you feeling?")
        assert speaker == "Doctor"
        assert utterance == "How are you feeling?"

    def test_dr_abbreviation(self):
        speaker, utterance = detect_speaker("Dr: Please take this medication.")
        assert speaker == "Dr"
        assert "medication" in utterance

    def test_senior_label(self):
        speaker, utterance = detect_speaker("Senior: I have a headache.")
        assert speaker == "Senior"
        assert utterance == "I have a headache."

    def test_patient_label(self):
        speaker, utterance = detect_speaker("Patient: My knee hurts.")
        assert speaker == "Patient"

    def test_no_label(self):
        speaker, utterance = detect_speaker("No label here.")
        assert speaker is None
        assert utterance == "No label here."

    def test_case_insensitive(self):
        speaker, _ = detect_speaker("DOCTOR: Good morning.")
        assert speaker == "Doctor"


class TestFlagMedicalTerms:
    def test_known_medical_term(self):
        result = flag_medical_terms(["headache", "pain", "hello"])
        assert result["headache"] is True
        assert result["pain"] is True
        assert result["hello"] is False

    def test_empty_tokens(self):
        assert flag_medical_terms([]) == {}

    def test_non_medical_tokens(self):
        result = flag_medical_terms(["morning", "coffee", "walk"])
        assert all(v is False for v in result.values())


class TestDetectMedicalPhrases:
    def test_known_phrase(self):
        assert "blood pressure" in detect_medical_phrases("Let's check your blood pressure.")

    def test_case_insensitive(self):
        assert "heart rate" in detect_medical_phrases("Your Heart Rate is normal.")

    def test_no_phrase(self):
        assert detect_medical_phrases("Good morning, how are you?") == []

    def test_multiple_phrases(self):
        result = detect_medical_phrases("Chest pain and shortness of breath are serious.")
        assert "chest pain" in result
        assert "shortness of breath" in result


class TestTokenizeConversation:
    SAMPLE = (
        "Doctor: Good morning. How are you feeling today?\n"
        "Senior: I have a headache and some dizziness.\n"
        "Doctor: Any chest pain?\n"
        "Senior: No, but I feel fatigued.\n"
    )

    def test_returns_list_of_dicts(self):
        result = tokenize_conversation(self.SAMPLE)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_speaker_detection(self):
        result = tokenize_conversation(self.SAMPLE)
        assert result[0]["speaker"] == "Doctor"
        assert result[1]["speaker"] == "Senior"

    def test_utterance_content(self):
        result = tokenize_conversation(self.SAMPLE)
        assert "headache" in result[1]["utterance"]

    def test_medical_terms_flagged(self):
        result = tokenize_conversation(self.SAMPLE)
        senior_turn = result[1]
        assert senior_turn["medical"].get("headache") is True
        assert senior_turn["medical"].get("dizziness") is True

    def test_empty_transcript(self):
        assert tokenize_conversation("") == []

    def test_blank_lines_skipped(self):
        transcript = "\nDoctor: Hello.\n\nSenior: Hi.\n"
        result = tokenize_conversation(transcript)
        assert len(result) == 2

    def test_medical_phrases_in_result(self):
        transcript = "Doctor: Let's check your blood pressure today.\n"
        result = tokenize_conversation(transcript)
        assert "blood pressure" in result[0]["medical_phrases"]

    def test_tokens_are_lowercase(self):
        result = tokenize_conversation("Doctor: Blood Pressure is high.\n")
        tokens = result[0]["tokens"]
        assert all(t == t.lower() for t in tokens)
