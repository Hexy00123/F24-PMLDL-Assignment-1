mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_ARTIFACT & 
airflow standalone
