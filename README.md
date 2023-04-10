# MLOps Framework

Description: Building a MLOps framework



To install requirements
```Bash
pip install -r requirements.txt
```

To set up a SQLite Database to store model artifacts
```Bash
sqlite mlflow.db
```

To launch a MLFlow Server
```Bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db # temp
mlflow server 
--backend-store-uri sqlite:///mlflow.db
--default-artifact-root ./mlruns 
--host 0.0.0.0 -p 1234
```

To train a model leveraging DVC pipeline
```Bash
dvc repro
```

To launch a Web App
```Bash
python src/api/app.py
```

To launch the frontend streamlit app
```Bash
streamlit run src/frontend/streamlit_main.py
```

## Localhost
MLFlow: http://0.0.0.0:1234 \
Flask: http://127.0.0.1:5000 \
Streamlit: http://localhost:8501


