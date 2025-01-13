from flask import Flask, render_template, request
# flask is framework for web development
# Flask - for creating web app

# render_template - class
# will run app via server that is backend/server, so if we need to re-direct from backend to frontend (index)
# then we need this

# request - frontend updated data is forst requested with backend

import pickle # saved scaler loaded from pickle library
import numpy as np
from tensorflow.keras.models import load_model

# creating flask app
app = Flask(__name__)  # built in parameters passed like this

# in between, we code the communication bet frontend and backend

# load the saved model and scalar
model = load_model("models/model.h5")
scaler = pickle.load(open("models/scaler.pkl", "rb")) # rb - read binary - reading file


# define the make_prediction function
def make_prediction(input_data):
    # preprocess input data, apply scaling
    input_data_scaled = scaler.transform(input_data) # use transform instead of fit

    # use the trained model to predict the class
    predictions = model.predict(input_data_scaled)

    # convert predictions to binary (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int)
    return predicted_classes

# routes - for communication
@app.route("/") # path/url of file for frontend index.html
def index():
    return render_template("index.html")


# route - that acts as an API End point
@app.route("/predict", methods=["GET", "POST"]) # there'r 2 methods to send/receive data
def predict():
    if request.method == "POST":
        # get input field data
        VWTI = float(request.form["vwti"]) # stored data and converted to float
        SWTI = float(request.form["swti"])
        CWTI = float(request.form['cwti'])
        EI = float(request.form["ei"])

# we get all data and converting it into numpy array
# Prepare input data for prediction
        input_data = np.array([[VWTI, SWTI, CWTI, EI]])

# get predictions
        result = make_prediction(input_data)  # 0 or 1
        print(result)
        if result[0] == 1:
            output = "Real"
        else:
            output = "Fake"
        print(output)

        # pass the result to the template
        return render_template("index.html", prediction = output)
    return render_template("index.html", prediction = None) 
# now return entire html but there"ll be no prediction

# creating python main function
if __name__ == "__main__":
    app.run(debug=True)  # app running and param passed is 'debug' for debugging

# basic syntax
