# End-to-End Student Performance Prediction Pipeline

A production-ready, modular Machine Learning project implementing clean software engineering principles. This repository builds a complete predictive pipeline—from data ingestion and custom preprocessing to hyperparameter-tuned multi-model benchmarking—to predict student **math scores** based on demographic and academic performance metrics.

---

## 🚀 Key Features
* **Modular Codebase:** Clean separation of concerns between data ingestion, transformation pipeline configs, and model training components.
* **13-Model Benchmarking Suite:** Automated optimization across Linear Models, Tree Ensembles, and advanced boosting frameworks (`XGBoost`, `LightGBM`).
* **Automated Hyperparameter Tuning:** Integrates robust, parallelized `RandomizedSearchCV` tuning with custom fallback safeguards for baseline models.
* **Custom Logging & Exception Framework:** Detailed tracking of training checkpoints and error line tracing.
* **Production Deployment Ready:** A Flask-based interactive web interface that reads real-time user inputs and streams immediate model inferences using local preprocessor and model artifacts.

---

## 📁 Project Directory Layout

```text
├── artifacts/               # Serialized data split boundaries and pickle binaries
│   ├── raw.csv              # Initial source dataset record
│   ├── train.csv            # Splitted training data partition
│   ├── test.csv             # Splitted validation data partition
│   ├── preprocessor.pkl    # Serialized feature processing pipeline object
│   └── model.pkl           # The absolute highest-scoring trained model binary
├── data/                    # Source data storage location
├── notebooks/               # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                     # Core backend source package directory
│   ├── components/          # Modular processing segments
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Partitions dataset boundaries
│   │   ├── data_transformation.py # Handles scaling, imputation, and encoding
│   │   └── model_trainer.py       # Benchmarks, tunes, and serializes models
│   ├── pipeline/            # Execution layer orchestrating application blocks
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py # Maps frontend payloads for inference
│   ├── logger.py            # Automated milestone tracking configuration
│   ├── exception.py         # Custom tracer error handler with line numbers
│   └── utils.py             # Reusable shared utilities (save_object, load_object)
├── templates/               # Frontend template layout views
│   ├── home.html            # Core project portal landing dashboard
│   └── index.html           # Interactive calculator input form view
├── app.py                   # Flask server endpoint router
├── requirements.txt         # Project package dependencies
└── setup.py                 # Installs 'src' directory as a local package module