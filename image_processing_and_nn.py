import numpy as np
import cv2

class IP_and_NN():
    def __init__(self):
        pass
    
    def find_corners_in_img(self, image_data):
        self.original = image_data.copy()
        self.gray = rgb2gray(self.original)
        self.blurred = cv2.GaussianBlur(self.gray, (3, 3), 3)
        self.canny = cv2.Canny(self.blurred, 120, 255, 1)
        kernel = np.ones((5,5),np.uint8)
        self.dilate = cv2.dilate(self.canny, kernel, iterations=1)
    
    def find_contours(self):
        # Find contours
        self.cnts = cv2.findContours(self.dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = self.cnts[0] if len(self.cnts) == 2 else self.cnts[1]
        self.cnts = [cv2.boundingRect(i) for i in self.cnts]
        self.cnts=sorted(self.cnts, key=lambda x: x[0])
        
    def get_results(self,run_nn_obj,image_data):
        results = []
        #run_nn_obj = RunNN()
        self.find_corners_in_img(image_data)
        self.find_contours()
        self.check_image_contours()
        for c in self.cnts:
            x,y,w,h = c
            ROI = self.gray[y:y+h, x:x+w]
            resized_image = cv2.resize(ROI,(28,28))
            blackAndWhiteImage = thresholdingImage(resized_image,0.66)
            blackAndWhiteImage = btowwtob(blackAndWhiteImage)
            result = run_nn_obj.test(blackAndWhiteImage.flatten())
            results.append(result)
        return results
    def check_image_contours(self):
        cv2.imshow("RGB",self.original)
        cv2.waitKey(0)
        cv2.imshow("Blurred",self.blurred)
        cv2.waitKey(0)
        for c in self.cnts:
            x,y,w,h = c
            ROI = self.gray[y:y+h, x:x+w]
            resized_image = cv2.resize(ROI,(28,28))
            cv2.imshow("Test",resized_image)
            cv2.waitKey(0)
        
        
        
        

def btowwtob(inputs):
    for i in range(0,inputs[:,0].size):
        for j in range(0,inputs[0,:].size):
            inputs[i,j] = 0 if inputs[i,j] == 255 else 255
    return inputs
def rgb2gray(image):
    gray_image = np.empty((image[:,0,0].size,image[0,:,0].size),dtype="uint8");
    for i in range(0,image[:,0,0].size):
        for j in range(0,image[0,:,0].size):
            gray_image[i,j] = ((0.3*image[i,j,0]) + (0.59 * image[i,j,1]) + (0.11 * image[i,j,2]) )
    return gray_image;
def thresholdingImage(grayImage,threshold):
    for i in range(0,grayImage[:,0].size):
        for j in range(0,grayImage[0,:].size):
            grayImage[i,j] = 0 if grayImage[i,j]/255.0 > threshold else 255
    return grayImage
