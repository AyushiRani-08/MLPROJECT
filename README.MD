# End-to-End Machine Learning Pipeline Project

A production-ready, modular Machine Learning project implementing clean software engineering principles. This repository handles the complete lifecycle of an ML model—from data ingestion and preprocessing to model training, evaluation, and deployment via a REST API.

---

## 🚀 Project Overview & Core Features
* **Modular Pipeline Architecture:** Separates data ingestion, transformation, and model training components into isolated, reusable blocks.
* **Custom Logging & Exception Framework:** Detailed tracking of script executions and precise error line tracing.
* **Reusable Artifacts Pipeline:** Serializes and manages preprocessors (`scaler.pkl`) and trained model configurations (`model.pkl`) cleanly.
* **Production Deployment:** Built-in web server endpoint to serve real-time predictions.

---

## 📁 Repository Structure

```text
├── artifacts/               # Serialized models and preprocessing objects (.pkl files)
├── data/                    # Local storage for data partitions
│   ├── raw/                 # Unprocessed source datasets
│   └── processed/           # Train/test splits after ingestion
├── notebooks/               # Jupyter notebooks for exploratory data analysis (EDA)
├── src/                     # Core source code package
│   ├── components/          # Pipeline data processing blocks
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/            # Execution layer orchestrating components
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── logger.py            # Automated checkpoint tracking
│   ├── exception.py         # Custom traceback error handling
│   └── utils.py             # Shared utility functions (e.g., saving/loading objects)
├── app.py                   # REST API production deployment script
├── requirements.txt         # Project package dependencies
└── setup.py                 # Script to package 'src' as a local editable module