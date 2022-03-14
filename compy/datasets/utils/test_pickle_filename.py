import unittest
from compy.datasets.utils import pickle_filename


class TestPickleFilename(unittest.TestCase):

    def test_parse_filename(self):
        file_idx, num_samples = pickle_filename.parse_pickle_filename("17_samples42.pickle")
        self.assertEqual(file_idx, 17)
        self.assertEqual(num_samples, 42)


if __name__ == '__main__':
    unittest.main()
