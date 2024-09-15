# Import mlflow
from ClassifierModel import DTCModel

# Ejemplo de uso
if __name__ == "__main__":
    filepath_model = "model.pkl"
    filepath_preprocessor = "preprocessor.pkl"

    # Instanciamos la clase del modelo DecisionTreeClassifier
    model = DTCModel(filepath_model, filepath_preprocessor)

    # Get data
    data_path = "test_set.csv"
    X = model.get_data(data_path)

    # Preprocess data
    X = model.transform(X)

    # Predictions
    y_pred = model.predict(X)

    print("Predictions:\n", y_pred)