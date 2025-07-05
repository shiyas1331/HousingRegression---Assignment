from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data

def evaluate_with_grid(model, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return mse, r2, grid.best_params_

def main():
    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge Regression": (
            Ridge(),
            {
                "alpha": [0.01, 0.1, 1, 10],
                "solver": ['auto', 'saga', 'lsqr'],
                "fit_intercept": [True, False]
            }
        ),
        "Decision Tree": (
            DecisionTreeRegressor(),
            {
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        ),
        "Random Forest": (
            RandomForestRegressor(),
            {
                "n_estimators": [50, 100],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5]
            }
        )
    }

    print("Model Performance with Hyperparameter Tuning:\n")
    for name, (model, params) in models.items():
        mse, r2, best_params = evaluate_with_grid(model, params, X_train, y_train, X_test, y_test)
        print(f"{name} -> MSE: {mse:.2f}, R²: {r2:.2f}")
        print(f"Best Params: {best_params}\n")

if __name__ == "__main__":
    main()
