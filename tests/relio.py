import unittest

import numpy as np
import pandas as pd

from relio import (
    Concept,
    CostFunction,
    DriftStatus,
    OTMetric,
    Relio,
)


class TestRelio(unittest.TestCase):

    def setUp(self):
        # Two well-separated concepts: rows 0-299 ~ N(0, 1), rows 300-599 ~ N(6, 1).
        rng = np.random.default_rng(0)
        self.features = np.vstack(
            [rng.normal(0, 1, (300, 4)), rng.normal(6, 1, (300, 4))]
        )
        self.df = pd.DataFrame(self.features[:300])

    def create_detector(
        self,
        window_size=30,
        warning_threshold=20,
        detection_threshold=40,
    ):
        return Relio(
            window_size=window_size,
            warning_threshold=warning_threshold,
            detection_threshold=detection_threshold,
            ot_metric=OTMetric.WASSERSTEIN2,
            cost_function=CostFunction.SEUCLIDEAN,
            stability_threshold=4,
            df=self.df,
        )

    def test_compare_distributions(self):
        relio = self.create_detector()

        # Same concept: no drift.
        result, distance = relio.compare_distributions(
            self.features[:50],
            self.features[50:100],
        )
        self.assertLess(distance, relio.get_warning_thold())
        self.assertEqual(result, DriftStatus.NO_DRIFT)

        # Different concepts: drift.
        result, distance = relio.compare_distributions(
            self.features[250:300],
            self.features[300:350],
        )
        self.assertGreater(distance, relio.get_detect_thold())
        self.assertEqual(result, DriftStatus.DRIFT)

    def test_detect_gradual_drift(self):
        # is_drift_gradual() reads the concepts that started inside the
        # transition, so we build them directly (private attributes).
        def relio_with_concept_lengths(lengths):
            relio = self.create_detector()
            for start_time, length in enumerate(lengths, start=1):
                concept = Concept(start_time, None)
                for _ in range(length - 1):
                    concept.increment_length()
                relio.add_concept(concept)
            relio._Relio__transition_start = 1
            relio._Relio__transition_end = len(lengths)
            return relio

        # Source 1 shrinks (5, 5, 2, 1) while source 2 grows (1, 2, 4, 6).
        self.assertTrue(
            relio_with_concept_lengths([5, 1, 5, 2, 2, 4, 1, 6]).is_drift_gradual()
        )
        self.assertFalse(
            relio_with_concept_lengths([5, 1, 3, 2, 4, 2, 7]).is_drift_gradual()
        )

    def test_detect_drift(self):
        relio = self.create_detector(window_size=50)

        # .copy(): detect_drift updates the reference array in place.
        concept = Concept(1, self.features[:50].copy())
        relio.add_concept(concept)
        relio.set_curr_concept(concept)

        # No drift.
        relio.set_curr_win(self.features[50:100])
        relio.detect_drift()

        self.assertFalse(relio.get_adapt_model())
        self.assertFalse(relio.get_adjust_model())
        self.assertEqual(relio.get_action(), DriftStatus.NO_DRIFT)

        # Drift detected.
        relio.set_curr_win(self.features[300:350])
        relio.detect_drift()

        self.assertFalse(relio.get_adjust_model())
        self.assertTrue(relio.get_adapt_model())
        self.assertEqual(relio.get_action(), DriftStatus.DRIFT)


if __name__ == "__main__":
    unittest.main()
