#!/usr/bin/env python3
"""Unit tests for the Perception and Social module.

Covers the parts of the pipeline that turn a raw signal into a social
decision: the mapping from a facial expression to a mood, the estimation of
urgency from mood and text, the proxemic geometry, the speaking pace, and the
handling of the menu inside the dialogue agent.

No simulator, no microphone and no network are needed: every function under
test is pure. Run with

    python -m unittest test_perception_social -v
"""
import math
import unittest

import waiter
from core import emotion_module as em
from core import dialogue_agent as da


class MoodFromFacialSignal(unittest.TestCase):
    """emotion_module: expression -> mood dictionary."""

    def test_smile_gives_positive_mood(self):
        mood = em._fallback_mood({"emotion": "happy", "expression": "smiling"}, "")
        self.assertEqual(mood["emotion"], "happy")
        self.assertEqual(mood["sentiment"], "positive")

    def test_negative_words_override_a_smile(self):
        mood = em._fallback_mood({"emotion": "happy"}, "this is terrible, I hate waiting")
        self.assertEqual(mood["sentiment"], "negative")
        self.assertEqual(mood["urgency"], "high")

    def test_sad_face_gives_negative_mood_without_urgency(self):
        mood = em._fallback_mood({"emotion": "sad"}, "")
        self.assertEqual(mood["emotion"], "sad")
        self.assertEqual(mood["urgency"], "medium")

    def test_no_face_falls_back_to_neutral(self):
        mood = em._fallback_mood({}, "")
        self.assertEqual(mood["emotion"], "neutral")
        self.assertEqual(mood["sentiment"], "neutral")

    def test_emotion_aliases_are_normalised(self):
        self.assertEqual(em._mood_from_emotion("happiness", "t")["emotion"], "happy")
        self.assertEqual(em._mood_from_emotion("ANGER", "t")["emotion"], "angry")

    def test_unknown_emotion_becomes_neutral(self):
        self.assertEqual(em._mood_from_emotion("perplexed", "t")["emotion"], "neutral")

    def test_mood_json_survives_trailing_commas(self):
        raw = '{"emotion": "sad", "sentiment": "negative",}'
        self.assertEqual(em._normalise_mood(raw, "test")["emotion"], "sad")

    def test_malformed_mood_json_raises(self):
        with self.assertRaises(ValueError):
            em._normalise_mood("no json here", "test")


class UrgencyEstimation(unittest.TestCase):
    """waiter: how urgently a customer should be served."""

    def test_absent_mood_defaults_to_medium(self):
        self.assertEqual(waiter.serve_urgency(None), "medium")

    def test_negative_sentiment_raises_urgency(self):
        self.assertEqual(
            waiter.serve_urgency({"sentiment": "negative", "urgency": "low"}), "high")

    def test_explicit_high_urgency_is_kept(self):
        self.assertEqual(waiter.serve_urgency({"urgency": "high"}), "high")

    def test_invalid_urgency_value_falls_back(self):
        self.assertEqual(waiter.serve_urgency({"urgency": "extreme"}), "medium")


class SpeakingPace(unittest.TestCase):
    """waiter: the voice speeds up only for genuine hurry."""

    def test_hurried_neutral_customer_speeds_the_voice_up(self):
        self.assertEqual(waiter.speech_urgency({"urgency": "high"}), "high")

    def test_angry_customer_keeps_a_calm_voice(self):
        self.assertEqual(
            waiter.speech_urgency({"urgency": "high", "emotion": "angry"}), "medium")

    def test_happy_customer_is_not_rushed(self):
        self.assertEqual(
            waiter.speech_urgency({"urgency": "high", "emotion": "happy"}), "medium")

    def test_pace_is_mapped_to_words_per_minute(self):
        waiter.set_speech_pace("high")
        self.assertEqual(waiter._SPEECH_WPM, 220)
        waiter.set_speech_pace("low")
        self.assertEqual(waiter._SPEECH_WPM, 165)
        waiter.set_speech_pace("medium")
        self.assertIsNone(waiter._SPEECH_WPM)


class HurrySignalInText(unittest.TestCase):
    """waiter: a textual cue is required before trusting the model's urgency."""

    def test_english_cues_are_detected(self):
        for phrase in ("I'm in a hurry", "make it quick", "I am running late"):
            self.assertTrue(waiter._text_suggests_hurry(phrase), phrase)

    def test_italian_cues_are_detected(self):
        for phrase in ("ho fretta", "sbrigati per favore"):
            self.assertTrue(waiter._text_suggests_hurry(phrase), phrase)

    def test_a_farewell_is_not_a_hurry_cue(self):
        self.assertFalse(waiter._text_suggests_hurry("thank you, goodbye"))


class ProxemicGeometry(unittest.TestCase):
    """waiter: mood-driven adjustment of the approach."""

    def test_angry_customer_gets_more_space(self):
        self.assertLess(waiter._proxemics_delta({"emotion": "angry"}), 0.0)

    def test_happy_customer_is_approached_more_closely(self):
        self.assertGreater(waiter._proxemics_delta({"emotion": "happy"}), 0.0)

    def test_neutral_customer_keeps_the_nominal_dock(self):
        self.assertEqual(waiter._proxemics_delta({"emotion": "neutral"}), 0.0)
        self.assertEqual(waiter._proxemics_angle({"emotion": "neutral"}), 0.0)

    def test_missing_mood_produces_no_adjustment(self):
        self.assertEqual(waiter._proxemics_delta(None), 0.0)
        self.assertEqual(waiter._proxemics_angle(None), 0.0)

    def test_distressed_customer_is_approached_off_axis(self):
        angle = waiter._proxemics_angle({"emotion": "angry"})
        self.assertAlmostEqual(math.degrees(angle), 20.0, places=3)

    def test_hurried_customer_is_also_approached_off_axis(self):
        self.assertGreater(waiter._proxemics_angle({"urgency": "high"}), 0.0)

    def test_adjustments_stay_within_a_few_centimetres(self):
        for mood in ({"emotion": "angry"}, {"emotion": "happy"}):
            self.assertLessEqual(abs(waiter._proxemics_delta(mood)), 0.20)


class MenuNormalisation(unittest.TestCase):
    """dialogue_agent: from the word the customer used to the word the bridge needs."""

    def test_synonyms_resolve_to_the_serve_word(self):
        for word in ("coke", "cola", "coca", "Coca-Cola"):
            self.assertEqual(da.canon_item(word), "coca cola", word)

    def test_food_synonyms_resolve(self):
        self.assertEqual(da.canon_item("crisps"), "pringles")

    def test_unknown_word_is_rejected(self):
        self.assertEqual(da.canon_item("mojito"), "")

    def test_empty_input_is_rejected(self):
        self.assertEqual(da.canon_item(""), "")
        self.assertEqual(da.canon_item(None), "")

    def test_every_menu_item_canonicalises_to_itself(self):
        for item in da._ITEMS:
            serve = str(item["serve"]).lower()
            self.assertEqual(da.canon_item(serve), serve)


class OffMenuGuard(unittest.TestCase):
    """dialogue_agent: the robot may echo an off-menu word, not offer one."""

    def test_an_invented_item_is_caught(self):
        self.assertEqual(da._invents_offmenu("Would you like a coffee?", ""), "coffee")

    def test_echoing_the_customer_is_allowed(self):
        self.assertIsNone(
            da._invents_offmenu("Sorry, we have no coffee.", "can I have a coffee?"))

    def test_menu_items_are_never_flagged(self):
        self.assertIsNone(da._invents_offmenu("I can bring you a Sprite.", ""))


class SpokenTextSanitising(unittest.TestCase):
    """dialogue_agent: what reaches the speech synthesiser."""

    def test_action_markers_are_removed(self):
        self.assertNotIn("[serve", da._sanitize_say("[serve] Here is your drink."))

    def test_emoji_are_removed(self):
        self.assertNotIn("\U0001F600", da._sanitize_say("Enjoy \U0001F600"))

    def test_reply_is_truncated_to_two_sentences(self):
        out = da._sanitize_say("One. Two. Three. Four.")
        self.assertEqual(out, "One. Two.")

    def test_surrounding_quotes_are_stripped(self):
        self.assertEqual(da._sanitize_say('"Hello there"'), "Hello there")


class CustomerReplyHelpers(unittest.TestCase):
    """waiter: small interpreters used around the dialogue."""

    def test_affirmative_answers_are_recognised(self):
        for reply in ("yes please", "sure", "okay", "why not", "go ahead"):
            self.assertTrue(waiter.is_yes(reply), reply)

    def test_negative_answer_is_not_affirmative(self):
        self.assertFalse(waiter.is_yes("no thanks"))

    def test_serve_words_map_to_display_names(self):
        self.assertEqual(waiter.nice_name("coca cola"), "Coca-Cola")
        self.assertEqual(waiter.nice_name("juice"), "orange juice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
