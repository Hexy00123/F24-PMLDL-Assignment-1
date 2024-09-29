# PMLDL Assignment 1
## Ruslan Izmailov B22-DS-01


### Project description

The goal of this project was to develop a model for predicting the number of stars based on feedback for places on maps. This project aims to provide an accurate and efficient way to evaluate the quality of locations based on user reviews.

To achieve this, I utilized a dataset from Yandex. The project involved extracting and preparing data stored in S3 object storage, training various models using different approaches, and logging the results to MLFlow for tracking and comparison.

Dataset link: https://github.com/yandex/geo-reviews-dataset-2023

### Project structure

#### Airflow
Airflow implements the main pipeline of extracting and preparing data (that is stored in S3 object storage), training models, and logging them to MLFlow. This pipeline ensures that the data is processed efficiently and that the models are trained and tracked accurately.

![alt text](content/pipeline.png)
 
#### MLFlow 
MLFlow is used to keep track of all models that are trained by the Airflow pipeline. It provides a centralized platform for managing and comparing different models, making it easier to select the best performing ones.

#### Model architectures 
Fro this project there were used 2 ways of text processing: classic one via tf-idf and the modern using transformers. The following table shows the models trained for each type of embedding:
 | Embedding Type | Models Trained                                 |
 |----------------|-------------------------------------------------|
 | TF-IDF         | CatBoost, Random Forest, XGBoost               |
 | Transformer    | CatBoost, Random Forest, XGBoost, SVM          |

### How to run
#### 1. Setup the environment: 
Export environment variables. 
```sh 
source path/to/your/env/bin/activate

export AIRFLOW_HOME=$PWD/services/airflow
export PROJECT_ROOT=$PWD

export MLFLOW_TRACKING_URI=postgresql://user:password@localhost:5432/db
export MLFLOW_ARTIFACT=s3://your-bucket-name/mlflow
export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
```

Install libraries
```sh
pip install -r requirements.txt
```

#### 2. Run services: 
```sh
bash scripts/run_services.sh
```

It will run Airflow and MLFlow servers and UIs. 


#### 3. Setup configs: 
All model data is stored in configs directory. 

Create config files for models/embedding types that you want and add them to ___main.yaml___ config file. These configs will be parsed by Airflow service and added to the ___train pipeline___.  

Config example: 
```yaml
model_name: CatBoost_classifier

module_name: "catboost"
class_name: "CatBoostClassifier"
 
params:
  classifier__iterations: 1000
  classifier__depth: 5
  classifier__logging_level: 'Silent'
  classifier__task_type: 'GPU'

  embedding__max_features: 5000
```
#### 4. Run train pipeline: 
To train model specified in configs directory you need to run the main pipeline via Airflow UI: 
![alt text](content/run_airflow_pipeline.png)

After the pipeline is run you can check your models and their metrics in MLFlow UI: 
![alt text](content/mlflow_ui.png)

#### 5. Download the best model: 
Before deploying you need to download the model.

```src
python src/download_model.py --filename=filename.pkl --model_url=model_url
```

Model url looks like this and can be found in MLFlow UI:  
```
runs:/e71da5424fc34ac0b0c2d51a79580f7f/XGBoost_transformer_experiment
```

Script will automatically put the model file to ```deployment/model/filename.pkl```. 

#### 6. Deploy: 
Deployment done via Docker and Docker-compose tools, you need to only edit ```services:model:environment:MODEL_FILENAME``` field to be the filename of the model you want to deploy. 
```yaml
services:
  frontend:
    build:
      context: ./
      dockerfile: ./deployment/frontend/Dockerfile
    environment:
      - BACKEND_URL=model
      - BACKEND_PORT=8000
    ports:
      - 8501:8501
    networks:
      - net

  model:
    build:
      context: ./
      dockerfile: ./deployment/backend/Dockerfile
    volumes:
      - ./deployment/model:/deployment/model
    environment:
      - MODEL_FILENAME=filename.pkl
    ports:
      - 8001:8000
    networks: 
      - net 

networks:
  net:
    driver: bridge
```
After you put the filename in docker-compose.yaml file just execute: 
```sh 
docker compose up --build
```

#### 7. Enjoy: 
![alt text](content/prediction_positive.png)

![alt text](content/prediction_negative.png)
