from sensor.utils.main_utils import load_numpy_array_data
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
import os,sys


from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object,load_object



class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):

        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

            
        except Exception as e:
            raise SensorException(e,sys)

    def perform_hyper_paramter_tuning(self, x_train, y_train):...
        # try:
        #     # Define the parameter grid for XGBoost
        #     param_grid = {
        #         'n_estimators': [100, 200, 300],
        #         'max_depth': [3, 5, 7],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'subsample': [0.8, 0.9, 1.0],
        #         'colsample_bytree': [0.8, 0.9, 1.0],
        #         'min_child_weight': [1, 3, 5],
        #         'gamma': [0, 0.1, 0.2]
        #     }

        #     # Initialize the base model
        #     xgb_clf = XGBClassifier(
        #         objective='binary:logistic',
        #         random_state=42,
        #         n_jobs=-1
        #     )

        #     # Initialize GridSearchCV
        #     from sklearn.model_selection import GridSearchCV
        #     grid_search = GridSearchCV(
        #         estimator=xgb_clf,
        #         param_grid=param_grid,
        #         scoring='f1',
        #         cv=5,
        #         n_jobs=-1,
        #         verbose=1
        #     )

        #     # Perform grid search
        #     logging.info("Starting hyperparameter tuning...")
        #     grid_search.fit(x_train, y_train)

        #     # Get the best parameters
        #     best_params = grid_search.best_params_
        #     logging.info(f"Best parameters found: {best_params}")
        #     logging.info(f"Best F1 score: {grid_search.best_score_}")

        #     # Create a new model with the best parameters
        #     best_model = XGBClassifier(
        #         **best_params,
        #         objective='binary:logistic',
        #         random_state=42,
        #         n_jobs=-1
        #     )

        #     return best_model

        # except Exception as e:
        #     logging.error(f"Error in hyperparameter tuning: {str(e)}")
        #     raise SensorException(e, sys)
    

    def train_model(self,x_train,y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train,y_train)
            return xgb_clf
        except Exception as e:
            raise e
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            model = self.train_model(x_train, y_train)

            # # Use hyperparameter tuning to get the best model
            # logging.info("Starting model training with hyperparameter tuning...")
            # model = self.perform_hyper_paramter_tuning(x_train, y_train)
            
            # # Fit the best model on the training data
            # model.fit(x_train, y_train)
            # logging.info("Model training completed with best hyperparameters")

            y_train_pred = model.predict(x_train)

            classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")
            
            y_test_pred = model.predict(x_test)

            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)


            #Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            sensor_model = SensorModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)
            

            #model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)