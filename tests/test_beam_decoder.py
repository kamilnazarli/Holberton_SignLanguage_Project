#!/usr/bin/env python3
"""
Unit Tests for Word-Level Beam Search Decoder and Lexicon Trie.
"""

import unittest
import numpy as np

from scripts.beam_decoder import (
    AZ_ALPHABET,
    AZ_LEXICON,
    LETTER_TO_IDX,
    BigramLanguageModel,
    BeamSearchDecoder,
    LexiconTrie,
    create_simulated_sequence,
)


class TestLexiconTrie(unittest.TestCase):
    def setUp(self):
        self.trie = LexiconTrie(["BAKI", "BALIQ", "SALAM", "GÜL"])

    def test_exact_word_matches(self):
        self.assertTrue(self.trie.is_valid_word("BAKI"))
        self.assertTrue(self.trie.is_valid_word("BALIQ"))
        self.assertTrue(self.trie.is_valid_word("SALAM"))
        self.assertTrue(self.trie.is_valid_word("GÜL"))

        self.assertFalse(self.trie.is_valid_word("BAK"))
        self.assertFalse(self.trie.is_valid_word("RAKI"))
        self.assertFalse(self.trie.is_valid_word("GÜLL"))

    def test_prefix_matches(self):
        self.assertTrue(self.trie.is_valid_prefix("B"))
        self.assertTrue(self.trie.is_valid_prefix("BA"))
        self.assertTrue(self.trie.is_valid_prefix("BAK"))
        self.assertTrue(self.trie.is_valid_prefix("BAKI"))
        self.assertTrue(self.trie.is_valid_prefix("BAL"))
        self.assertTrue(self.trie.is_valid_prefix("SAL"))
        self.assertTrue(self.trie.is_valid_prefix("GÜ"))

        self.assertFalse(self.trie.is_valid_prefix("R"))
        self.assertFalse(self.trie.is_valid_prefix("RA"))
        self.assertFalse(self.trie.is_valid_prefix("GÜK"))

    def test_continuations(self):
        conts = set(self.trie.get_valid_continuations("BA"))
        self.assertEqual(conts, {"K", "L"})


class TestBigramLanguageModel(unittest.TestCase):
    def setUp(self):
        self.lm = BigramLanguageModel(AZ_LEXICON, alpha=0.3)

    def test_bigram_probabilities(self):
        # In Azerbaijani, after 'B', 'A' should have high probability (BAKI, BACI, BALIQ)
        lp_ba = self.lm.log_prob("A", "B")
        lp_bx = self.lm.log_prob("X", "B")
        self.assertGreater(lp_ba, lp_bx)

    def test_unigram_probabilities(self):
        lp_a = self.lm.log_prob("A", None)
        lp_f = self.lm.log_prob("F", None)
        self.assertIsInstance(lp_a, float)
        self.assertIsInstance(lp_f, float)


class TestBeamSearchDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = BeamSearchDecoder(beam_width=5, lm_weight=0.6, lexicon_word_bonus=3.0)

    def test_greedy_decoding_clean(self):
        seq = create_simulated_sequence("SALAM")
        pred, score = self.decoder.greedy_decode(seq)
        self.assertEqual(pred, "SALAM")
        self.assertLess(score, 0.0)

    def test_beam_width_1_equals_greedy_without_lm(self):
        seq = create_simulated_sequence("BAKI")
        greedy_pred, _ = self.decoder.greedy_decode(seq)
        beam_hyps = self.decoder.decode(seq, beam_width=1, use_lm=False, use_lexicon=False)
        self.assertEqual(beam_hyps[0].sequence, greedy_pred)

    def test_error_correction_on_ambiguous_letter(self):
        # Simulate acoustic error at position 0: 'R' (46%) > 'B' (38%)
        seq = create_simulated_sequence("BAKI", ambiguities={0: ("R", 0.46, 0.38)})

        # Greedy fails and gives RAKI
        greedy_pred, _ = self.decoder.greedy_decode(seq)
        self.assertEqual(greedy_pred, "RAKI")

        # Config D (+Bigram +Lexicon) corrects to BAKI
        hyps_lexicon = self.decoder.decode(seq, beam_width=5, use_lm=True, use_lexicon=True)
        self.assertEqual(hyps_lexicon[0].sequence, "BAKI")
        self.assertTrue(hyps_lexicon[0].is_valid_word)

    def test_ghj_confusion_correction(self):
        # Simulate static G vs Ş confusion in 'GÜL': 'Ş' (49%) > 'G' (39%)
        seq = create_simulated_sequence("GÜL", ambiguities={0: ("Ş", 0.49, 0.39)})
        greedy_pred, _ = self.decoder.greedy_decode(seq)
        self.assertEqual(greedy_pred, "ŞÜL")

        hyps = self.decoder.decode(seq, beam_width=5, use_lm=True, use_lexicon=True)
        self.assertEqual(hyps[0].sequence, "GÜL")


if __name__ == "__main__":
    unittest.main()

