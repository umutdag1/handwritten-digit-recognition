from flask import Flask, json, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
from image_processing_and_nn import IP_and_NN
import cv2
from run_nn import RunNN
import time

start_time = time.time()
run_nn_obj = RunNN()
print("Training begining.")
run_nn_obj.train()
print("Training done.")
print("--- %s seconds ---" % (time.time() - start_time))

api = Flask(__name__)
cors = CORS(api)

@api.route('/api/getResult', methods=['POST'])
def get_result():
  will_be_used_obj = IP_and_NN()
  data = request.json['pixels']
  resultt = run_nn_obj.test(data)
  print(resultt)
  data_new = np.array(data).reshape((28,28))
  new_image = Image.fromarray(data_new * 255)
  new_image.save('tmp.png')
  read_image = cv2.imread('tmp.png')
  result_arr = will_be_used_obj.get_results(run_nn_obj,read_image)
  
  ##new_image.save('tmp.png')
  ##result = run_nn_obj.train_and_test(data) 
  return json.JSONEncoder().encode(result_arr)
  
if __name__ == '__main__':
    api.run()

