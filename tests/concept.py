import unittest
import RELIO_API as relio

class ConceptTest(unittest.TestCase):
    
    def testIncrementLength(self):
        ref=[1,2,3]
        concept=relio.Concept(0, ref)
        concept.increment_length()
        self.assertEqual(concept.get_length(),2)
