import numpy as np
from flask import Flask, request, jsonify, render_template
# import pickle
from pyspark import SparkFiles
import flask
import pandas as pd
from rec_file import userID_to_int, findksimilarusers, predict_userbased, predict_items, give_upayment, give_activity
# import tensorflow as tf
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
# import keras
# from keras.models import load_model
# import h5py
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# keras.backend.clear_session() 
app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # request.form.values()
    inputs = [(x) for x in request.form.values()]
    userID=int(inputs[0][-3:])

    if(userID>138 or userID<1 or int(inputs[1])>8 or int(inputs[1])<1):
        return render_template('index.html', prediction_text="You have enter invalid content: TRY AGAIN!!!")

    if(inputs[2]=="cosine"):
        metric="cosine"
    elif(inputs[2]=='correlation'):
        metric="correlation"
    else:
        return render_template('index.html', prediction_text="Explicitly write 'correlation' or 'cosine': your text was invalid")

    seed_text=predict_items(userID_to_int(inputs[0]), metric = metric, count=int(inputs[1]))

    return render_template('prediction.html', userID=inputs[0], activity=give_activity(str(inputs[0])), upayment=give_upayment(str(inputs[0])), cuisines=seed_text)



if __name__ == "__main__":
    app.run(debug=True)
