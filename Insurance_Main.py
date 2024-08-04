import pandas as pd
from sklearn.model_selection import train_test_split
from Insurance_preprocessing import load_data, preprocess_data
from Insurance_Model_Training import train_and_evaluate_model

def main():
    # Load data
    url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
    df = load_data(url)
    
    # Preprocess data
    preprocessor = preprocess_data(df)
    X, y = df.drop('charges', axis=1), df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    results = train_and_evaluate_model(preprocessor, X_train, y_train, X_test, y_test)
    
    # Print results
    for model_name, metrics in results.items():
        print(f'\n{model_name} Results:')
        print(f'Best Parameters: {metrics["Best Params"]}')
        print(f'R²: {metrics["R²"]:.2f}')
        print(f'RMSE: {metrics["RMSE"]:.2f}')
        print(f'MAE: {metrics["MAE"]:.2f}')

if __name__ == '__main__':
    main()
