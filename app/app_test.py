# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
from imutils.video import FPS
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2
import os

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tflite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def main(image_filename):
    global image
    COUNT=0
    fps = FPS().start()
    cap = cv2.VideoCapture('/home/metis/test.avi')#'/home/test2.mp4'

        # Read frame from camera
    ret, image = cap.read()
    cv2.imwrite('contour1.jpg',image)
    filename = './contour1.jpg'
    
    #cap = cv2.VideoCapture(0)#'/home/test2.mp4'

    # Check if camera is opened correctly
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Set initial frame number
    frame_num = 0
    
    while True:
        try:
            ret, image = cap.read()
            cv2.imwrite('contour1.jpg',image)
            image_filename = './contour1.jpg'
            COUNT=COUNT+1
            
      

            # Load labels
            with open(LABELS_FILENAME, 'r') as f:
                labels = [label.strip() for label in f.readlines()]

            od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

            image = Image.open(image_filename)
            #print("this")
            image_np = np.array(image)
            # cv2.imshow('frame', image_np)
        

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            predictions = od_model.predict_image(image)
            
            for prediction in predictions:
                if prediction['probability']>0.5:
                    #print(prediction['boundingBox']['left'])
                    left = int(prediction['boundingBox']['left'] * image_np.shape[1])
                    
                    top = int(prediction['boundingBox']['top'] * image_np.shape[0])
                    width = int(prediction['boundingBox']['width'] * image_np.shape[1])
                    height = int(prediction['boundingBox']['height'] * image_np.shape[0])
                    #print(left,top,width,height)
                    # Draw the rectangle on the image
                    cv2.rectangle(image_np, (left, top), (left+width, top+height), (0, 255, 0), 2)
                    cv2.imshow('frame', image_np)
                

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    print(prediction)
            fps.update()
            # Press 'q' to quit
            if COUNT ==30:
                fps.stop()
                print("FPS: {:.2f}".format(fps.fps()))
                fps = FPS().start()
                COUNT=0
            #print(predictions)
        except:
            print("continue")
            cap = cv2.VideoCapture('/home/metis/test.avi')#'/home/test2.mp4'

        # Read frame from camera
            #ret, image = cap.read()
            continue

if __name__ == '__main__':


    # Open default camera (index 0)
    
    # Check if camera is opened correctly
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    # # Create output directory if it doesn't exist
    # if not os.path.exists('output'):
    #     os.makedirs('output')

    # Set initial frame number
    frame_num = 0
    
    COUNT=0
    global image
    

        # # Check if frame is read correctly
        # if not ret:
        #     print("Cannot read frame")
        #     break

        # # Increment frame number
        # frame_num += 1

        # Save image as 'output/frame_00001.jpg', 'output/frame_00002.jpg', etc.

    #cv2.imwrite('contour1.jpg',image)
    filename = './contour1.jpg'
    #

    # Display the captured frame


    main('./contour1.jpg')
