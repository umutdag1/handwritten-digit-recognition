from flask import Flask, json, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
from image_processing_and_nn import IP_and_NN
import cv2



api = Flask(__name__)
cors = CORS(api)

@api.route('/api/getResult', methods=['POST'])
def get_result():
  will_be_used_obj = IP_and_NN()
  #run_nn_obj = RunNN()
  data = request.json['pixels']
  data_new = np.array(data).reshape((28,28))
  new_image = Image.fromarray(data_new * 255)
  new_image.save('tmp.png')
  read_image = cv2.imread('tmp.png')
  result_arr = will_be_used_obj.get_results(read_image)
  
  #new_image.save('tmp.png')
  #result = run_nn_obj.train_and_test(data) 
  return json.dumps(result_arr)
  
if __name__ == '__main__':
    api.run()

