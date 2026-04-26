import unittest
import pandas as pd
from src.preprocessing import run_full_preprocessing

class TestTitanicPipeline(unittest.TestCase):
    def test_preprocessing_columns(self):
        # Create dummy data
        df = pd.DataFrame({
            'Name': ['Gaines, Mr. Gary'],
            'Sex': ['male'],
            'Age': [28],
            'Survived': [1]
        })
        processed = run_full_preprocessing(df)
        # Check if Survived stayed and Name was dropped/processed
        self.assertIn('survived', processed.columns)
        self.assertNotIn('name', processed.columns)

if __name__ == '__main__':
    unittest.main()
