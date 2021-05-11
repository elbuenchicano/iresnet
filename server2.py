from engine import *

from flask import Flask
from flask import jsonify
from flask import request
import json

################################################################################
################################################################################
@app.route("/")
def hello():
    return "<h1 style='color:blue'>Faces server</h1>"

if __name__ == "__main__":
    app.run(host='0.0.0.0')

