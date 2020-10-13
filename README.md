# Face-Recognition
This is small introduction project towards implementation of OpenCV using python. This project will feature Dataset creation, train model and use webcam to recognize.

Some Libraries and their uses:
1. OpenCV library - 
OpenCV is a library of programming functions mainly aimed at real-time computer vision. Originally developed by Intel, it was later supported by Willow Garage.
to install this library use command : pip install opencv-python, or
			                                pip install opencv-contrib-python
2. Numpy library - 
It is one of the many libraries available for python that adds support for large, multidimensional array and matrices, having a large collection of high-level mathematical        funtions for performing computtions.
to install this library use command : pip install numpy

3. Pillows library - 
Pillow or PIL is a python imaging library that adds image processing capabilities to python browser.
To install: pip install pillow

4. Pickle Library - 
Pickling is a process in which an object hierarchy to byte streams and unpickling is reverse process.
Pickle library implements binary protocols for serializing and de-serializing a Python object structure.
   
TO run the following code: make sure to have installed all the libraries mentioned.

1. python main.py*
2. python extract_embeddings.py --dataset dataset --embeddings output/PyPower_embed.pickle --detector face_detection_model --embedding-model openface.nn4.small2.v1.t7
3. python train_model.py --embeddings output/PyPower_embed.pickle --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle
4. python recognize_video.py --detector face_detection_model --embedding-model openface.nn4.small2.v1.t7 --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle


*While running main.py file, enter your name and take atleast 22 captures with diferrent face angles and lighting condition, repeat the main.py process atleast two times to have created two different datasets for two different persons.
