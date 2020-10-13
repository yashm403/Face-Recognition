# USAGE

#python recognize_video.py --detector face_detection_model --embedding-model openface.nn4.small2.v1.t7 --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from os.path import basename
from csv import writer
# For geting the data in excel.
import time
import datetime
from datetime import datetime
from datetime import date
import pandas as pd
#

list_user=[]
l=[]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")



#######################################################
def path_cutter():
    l=(current_directory).split("\\")
    l.remove(basename(current_directory))
    s=""
    for i in range(0,len(l)):
        if(i==len(l)-1):
            s=s+l[i]
        else:
            s=s+l[i]+"\\"
            
    return s


def directory_create():
    dir=os.path.join(path_cutter(),basename(current_directory),"Output_excel")
    os.mkdir(dir)
    
def export(l,d2,address):
    data=pd.DataFrame(l,columns=['Person Name',"Date","Time"])
    address_=address+"\\"+d2+".csv"
    data.to_csv(address_,index=False,header=True)


def add_to_existing(filename, data):
	with open(filename,'a+',newline='') as write_obj:
		csv_writer=writer(write_obj)
		for i in range(0,len(data)):
			csv_writer.writerow(data[i])



########################################################

#vs = VideoStream(src='http://harsh:harsh@192.168.43.207:7777/video').start()
vs = VideoStream(src=0).start()
time.sleep(2.0)
rows = 480
cols = 640

# start the FPS throughput estimator
fps = FPS().start()



# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	frame=cv2.flip(frame,1)
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=640)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			#Data Frame work
			#now=datetime.datetime.now()
			#t = time.localtime()

			#current_time = time.strftime('%I;%M;%S-%p', t)
			#day=now.strftime("%A")

			#d=now.strftime("%Y-%m-%d %H:%M:%S")
			#d1=d.split(" ")
			#print(name,d1)
			
			#current=d1[0][0:7]+".csv"
		

			#if(len(l)!=0):
			#	if(l[-1]!=name):
			#		list_user.append([name,d1[0],d1[1]])
			#		l.append(name)
			#else:
			#	list_user.append([name,d1[0],d1[1]])
			#	l.append(name)


			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			rtr=text.split(":")
			
			rtr1=rtr[1].split("%")


			# Data Ready.
			
			if(float(rtr1[0]) >= 70.00 ):
				#Data Frame work
				print(rtr[0]," ",rtr[1])
				now=datetime.now()
				t = time.localtime()

				current_time = time.strftime('%I;%M;%S-%p', t)
				day=now.strftime("%A")

				d=now.strftime("%Y-%m-%d %H:%M:%S")
				d1=d.split(" ")
				#print(name,d1)
				
				current=d1[0][0:7]+".csv"
			

				if(len(l)!=0):
					if(l[-1]!=name):
						list_user.append([name,d1[0],d1[1]])
						l.append(name)

						###########################################
						now=datetime.now()
						t = time.localtime()

						current_time = time.strftime('%I;%M;%S%p', t)
						day=now.strftime("%A")

						d=now.strftime("%Y-%m-%d %H:%M:%S")
						d1=d.split(" ")

									
						today = str(datetime.today())
						month=today.split(" ")
						d2=(month[0][0:7])


						current_directory=os.getcwd()
						data=pd.DataFrame(list_user,columns=['Person Name',"Date","Time"])


						if not os.path.exists(current_directory+"\\"+"Output_excel"):
							directory_create()
							addre=current_directory+"\\"+"Output_excel"
							if os.path.exists(addre+'\\'+current):
								print("Yes")
								add_to_existing(addre+"\\"+current,list_user)
							else:
								print("No")
								export(list_user,d2,addre)
														
						else:
							addre=current_directory+"\\"+"Output_excel"
							if os.path.exists(addre+"\\"+current):
								print("yes")
								add_to_existing(addre+"\\"+current,list_user)
							else:
								print("No")
								export(list_user,d2,addre)
						#######################################################################

				else:
					list_user.append([name,d1[0],d1[1]])
					l.append(name)

					##########################################################################
					now=datetime.now()
					t = time.localtime()

					current_time = time.strftime('%I;%M;%S%p', t)
					day=now.strftime("%A")

					d=now.strftime("%Y-%m-%d %H:%M:%S")
					d1=d.split(" ")

								
					today = str(datetime.today())
					month=today.split(" ")
					d2=(month[0][0:7])


					current_directory=os.getcwd()
					data=pd.DataFrame(list_user,columns=['Person Name',"Date","Time"])


					if not os.path.exists(current_directory+"\\"+"Output_excel"):
						directory_create()
						addre=current_directory+"\\"+"Output_excel"
						if os.path.exists(addre+'\\'+current):
							print("Yes")
							add_to_existing(addre+"\\"+current,list_user)
						else:
							print("No")
							export(list_user,d2,addre)
													
					else:
						addre=current_directory+"\\"+"Output_excel"
						if os.path.exists(addre+"\\"+current):
							print("yes")
							add_to_existing(addre+"\\"+current,list_user)
						else:
							print("No")
							export(list_user,d2,addre)

					###############################################################################



				print(list_user)
				print(l)
				##########################################################

				

				#screen-part
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

				


	# update the FPS counter
	fps.update()
	# show the output frame
	
	cv2.imshow("Frame", frame)
    
	key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
#######################################################################################################
# Imp part

###########################################################################################################

print(list_user)
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup

vs.stop()
cv2.destroyAllWindows()
