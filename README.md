
# Titanic ML Challenge Solution

**Author:** Aquino Velasco Osorio  
**Date:** 2025-07-13

---

## Overview

This project provides a full pipeline and production-ready API for the Titanic survival prediction challenge.  
It includes:
- Modular Design
- Modular ML training pipeline (Random Forest & SVC) with profiling and logging.
- FastAPI service for real-time predictions (with A/B model support).
- API Key security for all endpoints (loaded from `.env`).
- Unit testing (pytest).
- Dockerized workflow for both pipeline and API.
- CI/CD deployment

## Model Design & Results

This solution follows a **fully modular and reproducible pipeline** for the Titanic challenge ([Kaggle link](https://www.kaggle.com/c/titanic/data)).  
Key steps include:
- Exploratory data analysis, feature engineering, and visualization.
- Data cleaning, one-hot encoding of categorical features, and feature scaling (`Age`, `Fare`).
- Hyperparameter optimization with cross-validation for Random Forest and SVC.
- Detailed evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
- Feature importance analysis and A/B model comparison.

**Key Results:**
- **SVC:** Accuracy 0.82, F1 0.73, ROC-AUC 0.83
- **Random Forest:** Accuracy 0.81, F1 0.73, ROC-AUC 0.85
- Most important features: Sex (female), Fare, and Pclass.

**For full details see:**
- [rappi-ml-challenge.ipynb](./rappi-ml-challenge.ipynb)  
- [rappi-ml-challenge.pdf](./rappi-ml-challenge.pdf)



## Project Structure

```

titanic-ml/
│
├── .github/workflows/
│   └── ci-cd.yml
├── titanic/
│   └── train.csv
├── models/
├── pipeline/
│   ├── **init**.py
│   ├── data.py
│   ├── model.py
│   ├── profiling.py
│   └── logging_config.py
├── .env
├── .gitignore
├── api.py
├── Dockerfile.api
├── Dockerfile.pipeline
├── main.py
├── rappi-ml-challenge.ipynb
├── rappi-ml-challenge.pdf
├── README.md
├── requirements.txt
└── test_api.py
````
**Note:**
* You must create both titanic/ (with the train.csv file downloaded from Kaggle Titanic Dataset) and models/ folders manually before running the pipeline or API.

## Requirements

Before running or deploying this project, ensure you have installed:

- **Python 3.11**  
  [Download & install Python 3.11](https://www.python.org/downloads/release/python-3110/)
- **Docker**  
  [Install Docker](https://docs.docker.com/get-docker/)

All other dependencies are handled automatically via `pip install -r requirements.txt`.

## Setup

1. **Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install requirements**
   
   ```bash
   pip install -r requirements.txt
   ````

3. **Create your `.env` file**

   ```
   TITANIC_API_KEY=your_secret_api_key
   ```

4. **Train the models locally**

   ```bash
   python main.py
   ```

   Models and metrics will be saved in `/models`.

    **Or Build and run training pipeline with Docker:**

    ```bash
    docker build -f Dockerfile.pipeline -t titanic-pipeline .
    docker run --rm -v $PWD/models:/app/models titanic-pipeline
    ```


## API Usage

1. **Run the API locally**

   ```bash
   uvicorn api:app --reload
   ```

2. **Or with Docker**

   ```bash
   docker build -f Dockerfile.api -t titanic-api .
   docker run --rm -p 8000:8000 -v $PWD/models:/app/models --env-file .env titanic-api
   ```

3. **Endpoints**

   * `POST /predict`
     Predicts survival for a list of passengers.

     * Query param: `model=rf` or `model=svc`
     * Header: `X-API-Key: your_super_secret_api_key`
     * Example request body:

       ```json
       [
         {
           "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25,
           "C": false, "Q": false, "S": true,
           "female": false, "male": true,
           "Class_1": false, "Class_2": false, "Class_3": true
         }
       ]
       ```

   * `GET /feature_importance`
     Returns feature importances for Random Forest.
     Requires API key.

   * `GET /health`
     Simple health check and request counter.

   * **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)


## Testing

Run the API unit tests with:

```bash
pytest test_api.py
```

Tests require a valid `.env` with the same API key as the running API.


## Profiling, Logging & Monitoring

* Profiling and logging of CPU, RAM and runtime for key operations.
* Logs stored in `models/pipeline.log`.
* All endpoints are logged, including request profiling.


## Security

* All endpoints require an API key, set via `.env` and sent as `X-API-Key` header.
* Unauthorized or missing keys return HTTP 401.


## Dockerized Pipeline

**Build and run training pipeline:**

```bash
docker build -f Dockerfile.pipeline -t titanic-pipeline .
docker run --rm -v $PWD/models:/app/models titanic-pipeline
```

**Build and run API:**

```bash
docker build -f Dockerfile.api -t titanic-api .
docker run --rm -p 8000:8000 -v $PWD/models:/app/models --env-file .env titanic-api
```

## A/B Model Testing

* Both Random Forest and SVC models are available.
* Choose which model to use via `model=rf` or `model=svc` in `/predict` endpoint.
* Model metrics summary in `models/ab_testing_metrics.csv`.

## CI/CD Workflow and Deploy (Staging Environment)

This project uses **GitHub Actions** for continuous integration and deployment.  
**The workflow is only triggered when a pull request is opened from `develop` to `staging`**, allowing safe automated deploys to a staging (testing) environment.

**CI/CD steps:**
1. **Runs on PR from `develop` to `staging`.**
2. Installs all dependencies
3. Trains and exports the models using the latest data.
4. Runs the entire test suite (`pytest`).
5. Builds and pushes the Docker API image to Docker Hub.
6. Uploads the trained model files to the AWS EC2 server (staging instance).
7. Deploys the updated API via Docker and Nginx reverse proxy on EC2 (replacing the previous container).

> **Note:** This setup ensures that **only validated code on `staging` is deployed to the staging environment**, minimizing risk.

**API is available (for staging/testing) at:**  
[http://ec2-3-20-164-182.us-east-2.compute.amazonaws.com/docs](http://ec2-3-20-164-182.us-east-2.compute.amazonaws.com/docs)

**(You must use the correct API key for all requests.)**


### How to Deploy to Production?

To deploy to production, open a pull request from `staging` to `master` and configure a similar workflow targeting the production EC2 server or other infrastructure.

## Nginx Reverse Proxy

The AWS EC2 instance hosting the API uses **Nginx as a reverse proxy** to route external HTTP traffic from port 80 to the internal FastAPI application running on port 8000 in Docker.

**Why Nginx?**
- Exposes the API on standard web ports (80 for HTTP, optionally 443 for HTTPS).
- Enables better security, logging, and scalability.
- Makes it easy to add SSL certificates (Let's Encrypt) in the future(production enviroment).