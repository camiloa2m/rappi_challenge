from pickle import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# DecisionTreeClassifierTrainer
class DTCTrainer(TransformerMixin, BaseEstimator):
    def __init__(self, test_size=0.3):
        """
        Inicializa el entrenador del DecisionTreeClassifier.

        Args:
        - test_size: Proporción del conjunto de datos destinado a test.
        """
        self.model = tree.DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        )
        self.test_size = test_size
        self.numeric_features = [
            "TO_USER_DISTANCE",
            "TOTAL_EARNINGS",
            "DISTANCE_TO_STORE",
            "TIP",
        ]
        self.categorical_features = ["COUNTRY", "CITY"]
        self.preprocessor = None

    def get_data(self, dataset_path: str):
        """
        Carga el conjunto de datos desde un archivo CSV, feature selection, feature engineering inicial,  y divide en train y test.

        Args:
        - dataset_path: Ruta al archivo CSV.

        Returns:
        - X_train, X_val, y_train, y_val: Datos de entrenamiento y validación divididos.
        """
        print("Leyendo datos y creando nuevas features...")
        df = pd.read_csv(dataset_path, sep=",", engine="python")
        
        X = df.drop(columns=["ORDER_ID", "CREATED_AT", "SATURATION", "TAKEN"])
        y = df[["TAKEN"]]
        
        X["RATIO_DISTANCE"] = X["DISTANCE_TO_STORE"]/X["TO_USER_DISTANCE"]
        self.numeric_features = self.numeric_features + ["RATIO_DISTANCE"]
        
        # X = X.drop(columns=["DISTANCE_TO_STORE", "TO_USER_DISTANCE"])
        # self.numeric_features.remove("DISTANCE_TO_STORE")
        # self.numeric_features.remove("TO_USER_DISTANCE")
        
        print(f"Dividiendo en conjunto de train y test (test_size={self.test_size})...")
        # Dividimos en conjunto de entrenamiento y validación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=42
        )

        print("Datos cargados!")
        
        return X_train, X_test, y_train, y_test
    
    
    def fit(self, X_train: pd.DataFrame):
        """
        Ajustando transformaciones para preprocesamiento de los datos.

        Args:
        - X_train: Datos de entrada (features).

        Returns:
        - X_train: Datos procesados.
        """
        print("Ajustando preprocessor...")

        # # Impute nan values
        # if any(X_train.isna().sum()):
        #     nan_count = X_train.isna().sum()
        #     nans = dict(nan_count[nan_count != 0])
        #     for k, v in nans.items():
        #         if k in self.numeric_features:
        #             mask = X_train[k].isna()
        #             X_train.loc[mask, k] = X_train.loc[:, k].mean()
        #         elif k in self.categorical_features:
        #             mask = X_train[self.categorical_features].isna()
        #             X_train.loc[:, self.categorical_features] = (
        #                 X_train[self.categorical_features].mode().to_numpy()
        #             )

        # # Impute outliers

        # Q1 = X_train[self.numeric_features].quantile(0.25)
        # Q3 = X_train[self.numeric_features].quantile(0.75)

        # IQR = Q3 - Q1  # Interquartile range

        # upper_limit_to_impute = Q3 + 1.5 * IQR

        # for col_str in self.numeric_features:
        #     X_train.loc[
        #         X_train[col_str] > (Q3[col_str] + 1.5 * IQR[col_str]), col_str
        #     ] = upper_limit_to_impute[col_str]
        
        
        numeric_transformer = Pipeline(
            steps=[("imputer_median", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
        )

        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_transformer = Pipeline(
            steps=[("imputer_mode", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )
        
        self.preprocessor.fit(X_train)
        
        print("Peprocessor ajustando!")
        
        return self

    def transform(self, X: pd.DataFrame):
        """
        Aplica transformaciones de preprocesamiento a los datos.

        Args:
        - X: Datos de entrada (features).

        Returns:
        - X: Datos transformados.
        """
        print("Transformando datos...")
        
        X = self.preprocessor.transform(X)

        print("Datos transformados!")
        
        return X


    def train(self, X_train, y_train):
        """
        Ajustando el modelo en el conjunto de entrenamiento.

        Args:
        - X_train: Características de entrenamiento.
        - y_train: Etiquetas de entrenamiento.
        """
        print("Ajustando el modelo de clasificación...")

        self.model.fit(X_train, y_train)
        
        print("Modelo ajustado!")
        
        return self

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en el conjunto de test.

        Args:
        - X_val: Características de validación.
        - y_val: Etiquetas de validación.

        Returns:
        - accuracy, precision, recall, f1: accuracy_Score, precision_score, recall_score, f1_score
        """
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred,)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")

        return accuracy, precision, recall, f1

    def save_model_scaler_enc(
        self,
        filepath_model="model.pkl",
        filepath_preprocessor="preprocessor.pkl",
    ):
        """
        Guarda el modelo entrenado en un archivo.

        Args:
        - filepath_model: Ruta del archivo donde se guardará el modelo.
        - filepath_preprocesso: Ruta del archivo donde se guardará el preprocessoe.
        """
        # save the model
        dump(self.model, open(filepath_model, "wb"))
        # save the preprocessor
        dump(self.preprocessor, open(filepath_preprocessor, "wb"))

        print("*" * 100)
        print(f"Modelo guardado en: {filepath_model}")
        print(f"Preprocessor guardado en: {filepath_preprocessor}")
        print("*" * 100)


# DecisionTreeClassifierModel
class DTCModel(BaseEstimator):
    def __init__(self, filepath_model, filepath_preprocessor):
        """
        Inicializa el entrenador de modelos de sklearn.

        Args:
        - filepath_model: Ruta del archivo del modelo.
        - filepath_preprocessor: Ruta del archivo del preprocessor.
        """
        self.numeric_features = [
            "TO_USER_DISTANCE",
            "TOTAL_EARNINGS",
            "DISTANCE_TO_STORE",
            "TIP",
        ]
        self.categorical_features = ["COUNTRY", "CITY"]
        self.model = load(open(filepath_model, "rb"))
        self.preprocessor = load(open(filepath_preprocessor, "rb"))

    def get_data(self, dataset_path: str):
        """
        Carga el conjunto de datos desde un archivo CSV, y feature selection.

        Args:
        - dataset_path: Ruta al archivo CSV.

        Returns:
        - X_train, X_val, y_train, y_val: Datos de entrenamiento y validación divididos.
        """
        df = pd.read_csv(dataset_path, sep=",", engine="python")
        X = df[self.numeric_features + self.categorical_features]
        
        X["RATIO_DISTANCE"] = X["DISTANCE_TO_STORE"]/X["TO_USER_DISTANCE"]
        self.numeric_features = self.numeric_features + ["RATIO_DISTANCE"]
        
        # X = X.drop(columns=["DISTANCE_TO_STORE", "TO_USER_DISTANCE"])
        # self.numeric_features.remove("DISTANCE_TO_STORE")
        # self.numeric_features.remove("TO_USER_DISTANCE")
        
        return X

    def transform(self, X: pd.DataFrame):
        """
        Aplica transformaciones de preprocesamiento a los datos.

        Args:
        - X: Datos de entrada (features).

        Returns:
        - X: Datos procesados.
        """
        print("Transformando datos...")
        
        X = self.preprocessor.transform(X)

        print("Datos transformados!")
        
        return X

    def predict(self, X):
        """
        Hace predicciones respecto a los datos ingresados.

        Args:
        - X: Features.

        Returns:
        - y_pred: Prediccion respecto a los datos ingresados.
        """
        y_pred = self.model.predict(X)

        return y_pred