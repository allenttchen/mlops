# MLOps Framework

Building a MLOps framework

To install requirements
```Bash
pip install -r requirements.txt
```

To launch a MLFlow Server
```Bash
mlflow server 
--backend-store-uri ./mlruns 
--default-artifact-root ./mlruns 
--host 0.0.0.0 -p 1234
```

To launch a Web App
```Bash
python src/api/app.py
```
