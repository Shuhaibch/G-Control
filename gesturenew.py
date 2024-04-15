from subprocess import call
import cv2, pickle
import numpy as np
import os
import webbrowser

from keras.models import load_model

prediction = None
model = load_model('cnn_model_keras2.h5')
def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text(pred_class):
	if pred_class==14:
	 value="Function 1"
	elif pred_class==11:
	 value="Function 4"
	elif pred_class==36:
	 value="Function 8"
	elif pred_class==24:
	 value="Function 5"
	else:
	 value=" "
	return value

def recognize():
	global prediction
	cam = cv2.VideoCapture(0)
	first_frame = cv2.flip(cam.read()[1],1)
	first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
	first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

	x, y, w, h = 300, 100, 300, 300
	while True:
		text = ""
		img = cv2.flip(img, 1)
		gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
		difference = cv2.absdiff(first_gray, gray_frame)
	
		 thresh1 = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
                 thresh1=thresh1[y:y+h,x:x+w]
		 contours = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
		 if len(contours) > 0:
	         contour = max(contours, key = cv2.contourArea)
		 if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				save_img = thresh[y1:y1+h1, x1:x1+w1]
				
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				
				pred_probab, pred_class = keras_predict(model, save_img)
				
				if pred_probab*100 > 95:
					text = get_pred_text(pred_class)
					if text=="Function 8":
					 os.popen('firefox www.google.com')
					 cv2.destroyWindow('Recognizing gesture')
					 cv2.destroyWindow('thresh')
					 cam.release()
					 #call("python3 shi.py", shell=True)
					elif text=="Function 1":
					 os.popen('libreoffice --writter')
					 cv2.destroyWindow('Recognizing gesture')
					 cv2.destroyWindow('thresh')
					 cam.release()
					 #call("python3 shi.py", shell=True)
					elif text=="Function 4":
					 call(["amixer", "-D", "pulse", "sset", "Master", "100%"])
					 #cv2.destroyWindow('Recognizing gesture')
					 #cv2.destroyWindow('thresh')
					 #cam.release()
					 #call("python3 shi.py", shell=True)
					elif text=="Function 5":
					 call(["amixer", "-D", "pulse", "sset", "Master", "0%"])
					 call(["shutdown", "-f", "-s", "-t", "60"])
					 #cv2.destroyWindow('Recognizing gesture')
					 #cv2.destroyWindow('thresh')
					 #cam.release()
					 #call("python3 shi.py", shell=True)
					else:
					 cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.imshow("Recognizing gesture", img)
		
		cv2.imshow("thresh", thresh1)
		
		if cv2.waitKey(2)==ord('r'):
			print('Background reset')
			first_frame = cv2.flip(cam.read()[1],1)
			first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
			first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
		elif cv2.waitKey(1)==ord('e'):
			break

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
recognize()
