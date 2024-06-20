import unittest
import RELIO_API as relio

class ConceptDriftTest(unittest.TestCase):
    def testSetDriftType(self):
        drift=relio.ConceptDrift(5)
        drift.set_drift_type(relio.DriftType.SUDDEN)
        self.assertEqual(drift.get_drift_type(), relio.DriftType.SUDDEN)