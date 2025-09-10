import unittest

import relio_api as relio


class ConceptDriftTest(unittest.TestCase):
    def test_set_drift_type(self):
        drift = relio.ConceptDrift(5)
        drift.set_drift_type(relio.DriftType.SUDDEN)
        self.assertEqual(drift.get_drift_type(), relio.DriftType.SUDDEN)


if __name__ == "__main__":
    unittest.main()
