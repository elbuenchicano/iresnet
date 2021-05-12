from flask import Flask
from flask import Flask
from flask import jsonify
from engine import *
from flask import request

import json

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1 style='color:blue'> Server Facer Raji!</h1>"
################################################################################
################################################################################
################################################################################
@app.route("/get")
def get():
    return jsonify(queryStudents())

################################################################################
################################################################################
################################################################################
@app.route("/getpost", methods=['GET','POST'])
def post():
    data = {"process": 'ok_update'}
    if request.method == 'GET':
        return jsonify(data)
    if request.method == 'POST':
        data = updateStudents(json.loads(request.data))
        return jsonify(data)

################################################################################
################################################################################
################################################################################
@app.route("/getpost_image", methods=['GET','POST'])
def postImage():
    data = {"process": 'ok_image'}
    if request.method == 'GET':
        print(__name__)
        queryImage(json.loads(request.data))
        return jsonify(data)
    if request.method == 'POST':
        print(__name__)
        data = queryImage(json.loads(request.data))
        print (data)
        return jsonify(data)

################################################################################
################################################################################
################################################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0')
