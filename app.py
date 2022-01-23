"""
Flask server
"""

import os
import random

from flask import Flask, request, jsonify, render_template
from playsound import playsound


from genre_rec_service import Genre_Recognition_Service

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    
    # get audio file
    audio_file = request.files["UploadedAudio"]

    # random string of digits for file name
    file_name = str(random.randint(0, 100000))

    # save the file locally
    audio_file.save(file_name)

    # invoke the genre recognition service
    grs = Genre_Recognition_Service()

    # make prediction
    prediction = grs.predict(file_name)

    # remove the .wav file
    os.remove(file_name)



    # message to be displayed on the html webpage
    prediction_message = f"""
    The song is predicted to be in the {prediction} genre.
    """
    
    return render_template("index.html", prediction_text=prediction_message)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False)
