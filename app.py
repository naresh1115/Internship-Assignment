from flask import Flask, request, jsonify
from functions import classifier
app = Flask(__name__)
cl = classifier()

@app.route("/")
def welcome():
    return "Hello Testers"

@app.route("/predict/<sentence>")
def predict(sentence):
    print(sentence)
    pred = cl.classify(sentence)
    return jsonify({"prediction" : pred[0]})

if __name__ == '__main__':
    app.run(debug=True)