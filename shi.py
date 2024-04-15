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
	if pred_class==0:
	 value="Function 1"
	elif pred_class==2:
	 value="Function 2"
	elif pred_class==11:
	 value="Function 11"
	elif pred_class==24:
	 value="Function 24"
	elif pred_class==8:
	 value="Function 8"
	elif pred_class==30:
	 value="Function 30"
	elif pred_class==31:
	 value="Function 31"
	elif pred_class==36:
	 value="Function 36"
	 #webbrowser.open("/home/nishad/Desktop/Interface Design/index.html",new=2)
	elif pred_class==5:
	 value="Function 3"
	elif pred_class==6:
	 value="Function 4"
	 #call(["amixer", "-D", "pulse", "sset", "Master", "100%"])
	elif pred_class==9:
	 value="Function 5"
	 #call(["amixer", "-D", "pulse", "sset", "Master", "0%"])
	else:
	 value=" "
	return value

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def recognize():
	global prediction
	cam = cv2.VideoCapture(2)
	if cam.read()[0] == False:
		cam = cv2.VideoCapture(0)
	hist = get_hand_hist()

	x, y, w, h = 300, 100, 300, 300
	while True:
		text = ""
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		thresh = cv2.merge((thresh,thresh,thresh))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh = thresh[y:y+h, x:x+w]
		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
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
					if text=="Function 2":
					 cv2.destroyWindow('Recognizing gesture')
					 cv2.destroyWindow('thresh')
					 cam.release()
					 call("python3 recognize_gesture.py", shell=True)
					 #cv2.destroyWindow('Recognizing gesture') 
					cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.imshow("Recognizing gesture", img)
		
		cv2.imshow("thresh", thresh)
		if cv2.waitKey(1) == ord('q'):
			break

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
recognize()
