from nn import NeuralNetwork
from flask import Flask, json, request
from flask_cors import CORS, cross_origin
import numpy as np
from run_nn import RunNN



api = Flask(__name__)
cors = CORS(api)

@api.route('/api/getResult', methods=['POST'])
def get_result():
  run_nn_obj = RunNN()
  data = request.json['pixels']
  result = run_nn_obj.train_and_test(data) 
  return result
  
if __name__ == '__main__':
    api.run()

