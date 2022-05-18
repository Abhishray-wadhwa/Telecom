from flask import Flask,request,jsonify, render_template
import numpy as np
import pickle
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/predict',methods=['post'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The customer will {}".format(prediction))
if __name__ == '__main__':
    app.run(debug=True)


