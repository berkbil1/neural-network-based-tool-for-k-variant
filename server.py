import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import models
from keras import layers
from keras.models import load_model
from flask import Flask, request
from datetime import datetime
import numpy as np
import sys
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import os
deprecation._PRINT_DEPRECATION_WARNINGS = False

app = Flask(__name__)
#loading model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#new added
global graph
graph = tf.get_default_graph()
#------------------------------

model1 = load_model('uniform.h5')
model2 = load_model('normal.h5')
model3 = load_model('binary.hdf5')
model4 = load_model('deltastep.hdf5')

#these inputs are genereted in attack.py after training the model

#uniform attack
mean1 =  np.array( [1368.03994495,125.16154019,25.25810582,2.99654323]).reshape(1,4)
std1 =  np.array( [655.36577029,72.012915,14.32451882,1.41273746]).reshape(1,4)

#normal distribution attack
mean2 =  np.array( [1359.36952026,125.56079027 ,25.60425219,3.000047]).reshape(1,4)
std2 =  np.array( [657.35571739,71.91749184,14.44257004, 1.4190522]).reshape(1,4)

#binary attack 
mean3 =  np.array( [1357.59725589, 125.58345101, 25.56920353, 2.99833498]).reshape(1,4)
std3 =  np.array( [659.53625882,72.18142518,14.44750587,1.42175579]).reshape(1,4)

#deltastep attack
mean4 =  np.array( [1363.45653278, 124.99257611, 26.01680652, 3.00154765]).reshape(1,4)
std4 =  np.array( [659.05766738, 72.34098871, 14.15333235, 1.41279276]).reshape(1,4)

@app.route("/predict", methods=["GET","POST"])
def predict():
  N = int(request.args['N'])
  n = int(request.args['n'])
  M = int(request.args['M'])
  K = int(request.args['K'])
  A = int(request.args['A'])
  
  inputA =  np.array( [N,n,M,K]).reshape(1,4)

  with graph.as_default():
   if(A == 1):  
    inputA = inputA - mean1
    inputA = inputA / std1    
    q = model1.predict([inputA])
   elif(A == 2): 
    inputA = inputA - mean2
    inputA = inputA / std2   
    q = model2.predict([inputA])
   elif(A == 3): 
    inputA = inputA - mean3
    inputA = inputA / std3 
    q = model3.predict([inputA])
   elif(A == 4): 
    inputA = inputA - mean4
    inputA = inputA / std4    
    q = model4.predict([inputA]) 
  return str( q[0][0])

app.run(host="localhost", debug=False, port=5000)