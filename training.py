# Import mlflow
from ClassifierModel import DTCTrainer

# Ejemplo de uso
if __name__ == "__main__":
    dataset_path = "orders_challengue_sep2023_PE_CO.csv"
    test_size = 0.3 # Size to made the splitting in train-test
    
    # Instanciamos la clase de entrenamiento
    trainer = DTCTrainer(test_size=test_size)

    # Get data
    X_train, X_test, y_train, y_test = trainer.get_data(dataset_path)

    # Transform data
    X_train = trainer.fit_transform(X_train)
    X_test = trainer.transform(X_test)

    # Parameters for DecisionTreeClassifier
    criterion = "gini"
    max_depth = None
    class_weight = "balanced"
    random_state = 0

    # Training model
    trainer = trainer.train(
        X_train,
        X_test,
        y_train,
        y_test,
        criterion=criterion,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )

    # Save model and preprocessor
    trainer.save_model_scaler_enc(
        filepath_model="model.pkl", filepath_preprocessor="preprocessor.pkl"
    )
