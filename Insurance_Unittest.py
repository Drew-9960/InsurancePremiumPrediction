import unittest
import pandas as pd
from Insurance_preprocessing import load_data, preprocess_data
from Insurance_Model_Training import train_and_evaluate_model
from sklearn.model_selection import train_test_split

class TestInsurancePrediction(unittest.TestCase):
    
    def setUp(self):
        self.df = load_data('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
        self.preprocessor = preprocess_data(self.df)
        
    def test_preprocessing(self):
        X, y = self.df.drop('charges', axis=1), self.df['charges']
        X_transformed = self.preprocessor.fit_transform(X)
        
        self.assertEqual(X_transformed.shape[1], 11)  # Adjust based on final feature count
        
    def test_model_training(self):
        X, y = self.df.drop('charges', axis=1), self.df['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = train_and_evaluate_model(self.preprocessor, X_train, y_train, X_test, y_test)
        
        for model_name, metrics in results.items():
            self.assertGreater(metrics['R²'], 0.5)

if __name__ == '__main__':
    unittest.main()
