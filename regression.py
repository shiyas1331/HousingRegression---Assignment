from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils import load_data

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2

def main():
    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    print("Model Performance:\n")
    for name, model in models.items():
        mse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"{name} -> MSE: {mse:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    main()

    
# This script evaluates multiple regression models on the Boston housing dataset.
# It loads the data, splits it, trains models, and prints their MSE and R² scores.
# Models used: Linear Regression, Decision Tree, and Random Forest (scikit-learn).
# Data loading is handled by the load_data function from utils.py.
# Results are printed for easy model comparison.