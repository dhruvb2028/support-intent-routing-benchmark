"""Independent acceptance checks for evaluation semantics and invalid inputs."""
import unittest
import numpy as np
import pandas as pd
from kaggle_yc.evaluate import (checked_probabilities, choose_threshold,
    classification_metrics, challenge_metrics, selective_metrics)

class IndependentMetricsTests(unittest.TestCase):
    def test_zero_coverage_is_undefined_accuracy(self):
        result = selective_metrics(['a', 'b'], ['a', 'a'], [False, False])
        self.assertEqual(result['coverage'], 0)
        self.assertIsNone(result['selective_accuracy'])
        self.assertIsNone(result['selective_error'])
        self.assertEqual(result['correctly_routed_fraction'], 0)

    def test_wrong_abstention_is_not_counted_as_correct_route(self):
        result = selective_metrics(['a', 'b', 'a'], ['a', 'a', 'a'], [True, False, False])
        self.assertAlmostEqual(result['coverage'], 1/3)
        self.assertAlmostEqual(result['correctly_routed_fraction'], 1/3)
        self.assertEqual(result['wrongly_routed_fraction'], 0)
        self.assertEqual(result['selective_accuracy'], 1)

    def test_empty_data_does_not_claim_perfect_accuracy(self):
        result = selective_metrics([], [], [])
        self.assertIsNone(result['coverage'])
        self.assertIsNone(result['selective_accuracy'])

    def test_macro_f1_includes_absent_ontology_classes(self):
        result = classification_metrics(['a', 'b'], [[1,0,0], [0,1,0]], ['a','b','c'], 0)
        self.assertAlmostEqual(result['macro_f1'], 2/3)
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(np.shape(result['confusion_matrix']), (3,3))

    def test_invalid_probability_values_rejected(self):
        for bad in ([[np.nan, 0]], [[np.inf, 0]], [[-0.1, 1.1]], [[.2,.2]], [[1,0,0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checked_probabilities(bad, ['a','b'])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            selective_metrics(['a'], ['a','b'], [True])

    def test_threshold_selection_uses_declared_cost(self):
        best, rows = choose_threshold(['a','b'], [[.9,.1],[.8,.2]], ['a','b'], [[.55,.45]])
        # A threshold above .8 and <=.9 routes exactly one correct request and defers the unsafe request.
        self.assertGreater(best['threshold'], .8)
        self.assertLessEqual(best['threshold'], .9)
        self.assertAlmostEqual(best['utility'], .75)
        self.assertEqual(best['accepted'], 1)
        self.assertEqual(best['unsafe_defer_rate'], 1)

    def test_unsafe_development_scores_also_validated(self):
        for bad in ([[np.nan,0]], [[.1,.1]], [[-.1,1.1]], [[1,0,0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                choose_threshold(['a'], [[.9,.1]], ['a','b'], bad)

    def test_challenge_metrics_separate_supported_and_unsafe(self):
        frame = pd.DataFrame({'kind':['supported','ambiguous','out_of_domain'], 'label':['a','ABSTAIN','ABSTAIN']})
        result = challenge_metrics(frame, [[.9,.1],[.6,.4],[.9,.1]], ['a','b'], .8)
        self.assertEqual(result['supported']['selective_accuracy'], 1)
        self.assertEqual(result['supported']['correctly_routed_fraction'], 1)
        self.assertEqual(result['ambiguous_defer_rate'], 1)
        self.assertEqual(result['out_of_domain_defer_rate'], 0)
        self.assertEqual(result['unsafe_defer_rate'], .5)

    def test_unknown_challenge_kind_rejected(self):
        frame = pd.DataFrame({'kind':['suported'], 'label':['a']})
        with self.assertRaises(ValueError):
            challenge_metrics(frame, [[.9,.1]], ['a','b'], .8)

    def test_unknown_supported_label_rejected(self):
        frame = pd.DataFrame({'kind':['supported'], 'label':['typo_label']})
        with self.assertRaises(ValueError):
            challenge_metrics(frame, [[.9,.1]], ['a','b'], .8)

if __name__ == '__main__':
    unittest.main()
