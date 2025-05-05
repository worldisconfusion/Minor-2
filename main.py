from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os , sys
from sensor.logger import logging


from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import load_object
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
import os
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR


from  fastapi import FastAPI
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, File, UploadFile, Response
import pandas as pd
import numpy as np


app = FastAPI()



origins = ["*"]
#Cross-Origin Resource Sharing (CORS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/",tags=["authentication"])
async def  index():
    return RedirectResponse(url="/docs")





@app.get("/train")
async def train():
    try:

        training_pipeline = TrainPipeline()

        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        
        training_pipeline.run_pipeline()
        return Response("Training successfully completed!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
        




@app.get("/predict")
async def predict():
    try:
        import os
        from datetime import datetime
        import numpy as np
        
        # Create results directory structure
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Create timestamp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_dir = os.path.join(results_dir, timestamp)
        if not os.path.exists(timestamp_dir):
            os.makedirs(timestamp_dir)
            
        # Check if model exists
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return {"status": "error", "message": "Model is not available"}
        
        # Load the model
        best_model_path = model_resolver.get_best_model_path()
        print(f"Loading model from: {best_model_path}")
        sensor_model = load_object(file_path=best_model_path)
        
        # Read data from example.csv
        try:
            print("Attempting to read example.csv...")
            df = pd.read_csv("example.csv")
            print(f"Successfully read {len(df)} rows from example.csv")
        except FileNotFoundError:
            return {"status": "error", "message": "example.csv file not found. Please make sure the file exists in the same directory as main.py"}
        except Exception as e:
            return {"status": "error", "message": f"Error reading example.csv: {str(e)}"}
            
        # Store actual values if they exist
        actual_values = None
        if 'class' in df.columns:
            actual_values = df['class']
            df = df.drop('class', axis=1)
            
        # Get the preprocessor from the model
        preprocessor = sensor_model.preprocessor
        
        # Get the feature names that the preprocessor was trained on
        try:
            # Try to get feature names from the Imputer step
            feature_names = preprocessor.named_steps['Imputer'].feature_names_in_
            print(f"Found {len(feature_names)} features in preprocessor")
        except (AttributeError, KeyError):
            # If we can't get feature names, use all numeric columns
            feature_names = df.select_dtypes(include=['number']).columns.tolist()
            print(f"Using {len(feature_names)} numeric columns as features")
        
        # Print the feature names for debugging
        print("Feature names from preprocessor:", feature_names)
        print("Columns in DataFrame:", df.columns.tolist())
        
        # Ensure the DataFrame has the same columns as the training data
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return {"status": "error", "message": f"Missing required features: {missing_features}"}
            
        # Reorder columns to match the training data
        df = df[feature_names]
        
        # Make predictions using the sensor model
        print("Making predictions...")
        y_pred = sensor_model.predict(df)
        print(f"Generated {len(y_pred)} predictions")
        
        # Create response DataFrame
        result_df = df.copy()
        result_df['prediction'] = y_pred
        result_df['prediction'] = result_df['prediction'].replace(TargetValueMapping().reverse_mapping())
        
        # Convert predictions to a more readable format
        predictions = []
        for idx, row in result_df.iterrows():
            predictions.append({
                "row": int(idx + 1),
                "prediction": str(row['prediction'])
            })
        
        # Calculate evaluation metrics if actual values are available
        evaluation_metrics = {}
        if actual_values is not None:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
            
            # Create mapping for label conversion
            label_mapping = {'pos': 1, 'neg': 0}
            
            # Convert actual values to numeric
            actual_values_numeric = actual_values.map(label_mapping)
            
            # Convert predictions to numeric if they're strings
            if isinstance(y_pred[0], str):
                y_pred_numeric = [1 if pred == 'pos' else 0 for pred in y_pred]
            else:
                y_pred_numeric = y_pred
            
            # Calculate metrics and convert to Python native types
            accuracy = float(accuracy_score(actual_values_numeric, y_pred_numeric))
            f1 = float(f1_score(actual_values_numeric, y_pred_numeric, average='weighted'))
            precision = float(precision_score(actual_values_numeric, y_pred_numeric, average='weighted'))
            recall = float(recall_score(actual_values_numeric, y_pred_numeric, average='weighted'))
            
            # Calculate confusion matrix and convert to list
            cm = confusion_matrix(actual_values_numeric, y_pred_numeric)
            cm_list = cm.tolist()
            
            # Handle confusion matrix values safely
            try:
                tn, fp, fn, tp = cm.ravel()
                specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            except ValueError:
                # If we can't unpack 4 values, it means we have only one class
                if len(cm.ravel()) == 1:
                    # All predictions are the same class
                    if y_pred_numeric[0] == 1:  # All positive
                        tp = int(cm[0][0])
                        tn, fp, fn = 0, 0, 0
                        prediction_type = "All predictions are positive (1)"
                    else:  # All negative
                        tn = int(cm[0][0])
                        tp, fp, fn = 0, 0, 0
                        prediction_type = "All predictions are negative (0)"
                    specificity = 1.0 if tn > 0 else 0.0
                    sensitivity = 1.0 if tp > 0 else 0.0
                else:
                    # Handle other cases
                    specificity = 0.0
                    sensitivity = 0.0
                    prediction_type = "Unknown prediction pattern"
            
            # Create evaluation metrics with Python native types
            evaluation_metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "sensitivity": sensitivity,
                "confusion_matrix": cm_list,
                "prediction_pattern": prediction_type if 'prediction_type' in locals() else "Normal binary classification",
                "total_samples": int(len(y_pred_numeric)),
                "positive_predictions": int(sum(y_pred_numeric)),
                "negative_predictions": int(len(y_pred_numeric) - sum(y_pred_numeric))
            }
            
            # Save results to file
            result_file = os.path.join(timestamp_dir, "result.txt")
            with open(result_file, "w") as f:
                f.write("=== Prediction Results ===\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("=== Predictions ===\n")
                for pred in predictions:
                    f.write(f"Row {pred['row']}: {pred['prediction']}\n")
                f.write("\n")
                
                f.write("=== Evaluation Metrics ===\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"Specificity: {specificity:.4f}\n")
                f.write(f"Sensitivity: {sensitivity:.4f}\n\n")
                
                f.write("=== Confusion Matrix ===\n")
                f.write(f"{cm}\n\n")
                
                f.write("=== Prediction Details ===\n")
                f.write(f"Prediction Pattern: {evaluation_metrics['prediction_pattern']}\n")
                f.write(f"Total Samples: {evaluation_metrics['total_samples']}\n")
                f.write(f"Positive Predictions: {evaluation_metrics['positive_predictions']}\n")
                f.write(f"Negative Predictions: {evaluation_metrics['negative_predictions']}\n")
            
            print(f"\nResults saved to: {result_file}")
        
        # Create response with Python native types
        response = {
            "status": "success",
            "message": f"Successfully generated {len(predictions)} predictions",
            "predictions": predictions,
            "evaluation_metrics": evaluation_metrics if evaluation_metrics else "No actual values provided for evaluation",
            "results_file": str(result_file) if 'result_file' in locals() else None
        }
        
        return response
        
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return {"status": "error", "message": f"An error occurred: {str(e)}"}





def main():
    try:
            
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)



if __name__ == "__main__":
    # file_path="/Users/myhome/Downloads/sensorlive/aps_failure_training_set1.csv"
    # database_name="smartride"
    # collection_name ="sensor"
    # dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)
    print(f"Starting server at http://localhost:{APP_PORT}")
    app_run(app, host="localhost", port=APP_PORT)







  












    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)