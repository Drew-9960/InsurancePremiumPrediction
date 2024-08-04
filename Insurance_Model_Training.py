from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np 
from sklearn.pipeline import Pipeline

def train_and_evaluate_model(preprocessor, X_train, y_train, X_test, y_test):
    param_grids = {
        'Random Forest Regressor': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10],
            'model__min_samples_split': [2, 5]
        },
        'Gradient Boosting Regressor': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5]
        },
        'Decision Tree Regressor': {
            'model__max_depth': [None, 10],
            'model__min_samples_split': [2, 5]
        }
    }
    
    results = {}
    
    for model_name, model in {
        'Random Forest Regressor': RandomForestRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(),
        'Decision Tree Regressor': DecisionTreeRegressor()
    }.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        test_predictions = best_model.predict(X_test)
        
        r2 = r2_score(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mae = mean_absolute_error(y_test, test_predictions)
        
        results[model_name] = {
            'Best Params': grid_search.best_params_,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        
    return results
