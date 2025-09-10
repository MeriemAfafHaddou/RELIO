import unittest

import numpy as np
import pandas as pd

import relio_api as relio


class TestRelio(unittest.TestCase):
    def test_reset_retrain(self):
        df = pd.read_csv("data/iris_sudden.csv")
        api = relio.RelioApi(
            30,
            10,
            20,
            relio.OTMetric.WASSERSTEIN2,
            relio.CostFunction.SEUCLIDEAN,
            4,
            df,
        )
        api.reset_retrain_model()
        self.assertEqual(api.get_retrain_model(), False)

    def test_add_concept(self):
        df = pd.read_csv("data/iris_sudden.csv")
        api = relio.RelioApi(
            30,
            10,
            20,
            relio.OTMetric.WASSERSTEIN2,
            relio.CostFunction.SEUCLIDEAN,
            4,
            df,
        )
        df = np.array(df)[:, :-1]
        concept = relio.Concept(1, df[:30])
        api.add_concept(concept)
        api.add_concept(concept)
        self.assertEqual(len(api.get_concepts()), 2)
        self.assertEqual(api.get_concepts(), [concept, concept])

    def test_compare_dist(self):
        df = pd.read_csv("data/iris_sudden.csv")
        api = relio.RelioApi(
            30,
            20,
            40,
            relio.OTMetric.WASSERSTEIN2,
            relio.CostFunction.SEUCLIDEAN,
            4,
            df,
        )
        df = np.array(df)[:, :-1]
        result, dist = api.compare_distributions(df[:50], df[50:100])
        self.assertLess(dist, api.get_alert_thold())
        self.assertEqual(result, 0)
        result, dist = api.compare_distributions(df[150:200], df[200:250])
        self.assertGreater(dist, api.get_detect_thold())
        self.assertEqual(result, 2)

    def test_is_gradual(self):
        df = pd.read_csv("data/iris_sudden.csv")
        api = relio.RelioApi(
            30,
            20,
            40,
            relio.OTMetric.WASSERSTEIN2,
            relio.CostFunction.SEUCLIDEAN,
            4,
            df,
        )
        lengths_true = [5, 1, 5, 2, 2, 4, 1, 6]
        self.assertTrue(api.is_gradual(lengths_true))
        lengths_false = [5, 1, 3, 2, 4, 2, 7]
        self.assertFalse(api.is_gradual(lengths_false))

    def test_monitor_drift(self):
        df = pd.read_csv("data/iris_sudden.csv")
        relio_api = relio.RelioApi(
            50,
            20,
            40,
            relio.OTMetric.WASSERSTEIN2,
            relio.CostFunction.SEUCLIDEAN,
            4,
            df,
        )
        # no drift
        df = np.array(df)[:, :-1]
        concept = relio.Concept(1, df[:50])
        relio_api.set_curr_concept(concept)
        relio_api.set_curr_win(df[50:100])
        relio_api.monitor_drift()
        self.assertFalse(relio_api.get_retrain_model())
        self.assertFalse(relio_api.get_partial_fit())

        relio_api.set_curr_win(df[300:350])

        relio_api.monitor_drift()
        self.assertFalse(relio_api.get_partial_fit())
        self.assertTrue(relio_api.get_retrain_model())


if __name__ == "__main__":
    unittest.main()
