from nn import NeuralNetwork
import numpy as np

class RunNN():
    def __init__(self):
        self.neural_network = NeuralNetwork()
        
    def train_and_test(self, test_data):
        training_data_file = open('mnist_train_editted.csv', 'r')
        training_data = training_data_file.readlines()
        training_data_file.close()   
        
        for data in training_data:
            handwritten_digit_raw = data.split(',')
            handwritten_digit_array = np.asfarray(handwritten_digit_raw[1:]).reshape((28, 28))
            handwritten_digit_target = int(handwritten_digit_raw[0])
            self.neural_network.train(prepare_data(handwritten_digit_array), create_target(handwritten_digit_target))
        
        for data in training_data:
            handwritten_digit_raw = data.split(',')
            handwritten_digit_array = np.asfarray(handwritten_digit_raw[1:]).reshape((28, 28))
            handwritten_digit_target = int(handwritten_digit_raw[0])
            self.neural_network.train(prepare_data(handwritten_digit_array), create_target(handwritten_digit_target)) 
            
            output = self.neural_network.query(test_data) 
        
        return get_index_of_max(output)
        
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