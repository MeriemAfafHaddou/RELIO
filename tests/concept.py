import unittest

import relio_api as relio


class ConceptTest(unittest.TestCase):

    def test_increment_length(self):
        ref = [1, 2, 3]
        concept = relio.Concept(0, ref)
        concept.increment_length()
        self.assertEqual(concept.get_length(), 2)


if __name__ == "__main__":
    unittest.main()
