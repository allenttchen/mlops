from flask import Flask, render_template, request, jsonify
import os
from src.constants import ROOT_DIR

static_dir = os.path.join(ROOT_DIR, "static")
template_dir = os.path.join(ROOT_DIR, "templates")
app = Flask(
    __name__,
    static_folder=static_dir,
    template_folder=template_dir
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                data_request = dict(request.form)
                response = api.form_response(data_request)
                return render_template("index.html", response=response)
            elif request.json:
                response = api.api_response(request.json)
                return jsonify(response)

        except Exception as e:
            print(e)
            return render_template("404.html", error=e)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
