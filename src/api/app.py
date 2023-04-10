from typing import List
import os
import json

from flask import Flask, render_template, request, jsonify, url_for
from markupsafe import escape
import numpy as np
from pydantic import BaseModel
import mlflow
from mlflow.pyfunc import load_model
from src.constants import ROOT_DIR
import torch.nn.functional as F

static_dir = os.path.join(ROOT_DIR, "static")
template_dir = os.path.join(ROOT_DIR, "templates")
app = Flask(
    __name__,
    static_folder=static_dir,
    template_folder=template_dir
)


class PredictData(BaseModel):
    input_image: List[List]


@app.route("/predict", methods=["POST"])
def predict():
    content = PredictData(**json.loads(request.json))
    img = content.input_image
    # load model
    model_name = "MNISTModel"
    stage = "Production"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = load_model(model_uri=f"models:/{model_name}/{stage}")
    print(np.max(img))
    print(np.min(img))
    img = np.array(img, dtype=np.float32)[np.newaxis, np.newaxis, ...] / 255
    preds = model.predict(img)[0]
    res = int(np.argmax(preds))
    return jsonify({"result": res})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
