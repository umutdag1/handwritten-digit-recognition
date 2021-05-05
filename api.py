from nn import NeuralNetwork
from flask import Flask, json, request
from flask_cors import CORS, cross_origin
import numpy as np

def prepare_data(handwritten_digit_array):
    return ((handwritten_digit_array / 255.0 * 0.99) + 0.0001).flatten()
    
def create_target(digit_target):
    target = np.zeros(10) + 0.01
    target[digit_target] = target[digit_target] + 0.98
    return target

def get_index_of_max(array):
    array = array[0]
    index = 0
    m = max(array)
    for n in array:
        if n == m:
            return index
        index = index + 1

api = Flask(__name__)
cors = CORS(api)

@api.route('/api/getResult', methods=['POST'])
def get_result():
  # Training
  neural_network = NeuralNetwork()
  training_data_file = open('mnist_train.csv', 'r')
  training_data = training_data_file.readlines()
  training_data_file.close()
  
  for data in training_data:
    handwritten_digit_raw = data.split(',')
    handwritten_digit_array = np.asfarray(handwritten_digit_raw[1:]).reshape((28, 28))
    handwritten_digit_target = int(handwritten_digit_raw[0])
    neural_network.train(prepare_data(handwritten_digit_array), create_target(handwritten_digit_target))

  for data in training_data:
    handwritten_digit_raw = data.split(',')
    handwritten_digit_array = np.asfarray(handwritten_digit_raw[1:]).reshape((28, 28))
    handwritten_digit_target = int(handwritten_digit_raw[0])
    neural_network.train(prepare_data(handwritten_digit_array), create_target(handwritten_digit_target))   
    
    data = request.json['pixels']
    output = neural_network.query(data)
    #print(get_index_of_max(output))

  return json.dumps(get_index_of_max(output))

if __name__ == '__main__':
    api.run()

