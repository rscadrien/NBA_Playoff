import unittest
from unittest.mock import patch

from Inference.valid_input import (
    get_valid_input_str,
    get_valid_input_seed,
    get_valid_input_record
)

class TestUserInput(unittest.TestCase):

    @patch('builtins.input', side_effect=['maybe', 'yes'])
    def test_get_valid_input_str(self, mock_input):
        result = get_valid_input_str("Enter yes or no: ", ['yes', 'no'])
        self.assertEqual(result, 'yes')

    @patch('builtins.input', side_effect=['0', '31', '15'])
    def test_get_valid_input_seed(self, mock_input):
        result = get_valid_input_seed("Enter seed (1-30): ", 30)
        self.assertEqual(result, 15)

    @patch('builtins.input', side_effect=['-0.1', '1.5', '0.75'])
    def test_get_valid_input_record(self, mock_input):
        result = get_valid_input_record("Enter record (0-1): ")
        self.assertAlmostEqual(result, 0.75)

if __name__ == '__main__':
    unittest.main()