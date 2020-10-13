import os
from os.path import basename
current_directory=os.getcwd()

def capture(path):
    os.chdir(path)
    import cv2
    import numpy as np
    from time import sleep

    cap=cv2.VideoCapture(0)
    count=0
    try:
        while True:
            ret,frame=cap.read()
            #frame=cv2.flip(frame,1)
            frame=cv2.resize(frame,(1200,600))
            key = cv2.waitKey(1)

            photos="Photos : "+str(count)
            x=500
            y=180
            w=250
            h=250

            cv2.rectangle(frame,(x,y),(x+w+20,y+h+20),(0,255,255),3)
            cv2.putText(frame,"Capture Module: Press 's' to Capture & 'q' to quit",(0,40),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0, color=(0,128,0),thickness=2)
            cv2.putText(frame,photos,(0,120),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.4, color=(0,128,0),thickness=3)
            if(key==ord('s')):
                filename='image'+str(count)+'.jpg'
                cv2.imwrite(filename, img=frame[y:y+h,x:x+h])   
                print("Clicked...")
                count=count+1
                cv2.putText(frame,"Captured",(230,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.4, color=(0,128,0),thickness=3)
                sleep(0.5)


            if(count==21):
                cv2.putText(frame," ,Sufficient Photos is captured.",(250,120),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.2, color=(0,0,255),thickness=3)


            cv2.imshow("Frame",frame)
            if (key==ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()
    except AssertionError:
        pass   
    except:
        pass
    
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
    dir=os.path.join(path_cutter(),basename(current_directory),"Dataset")
    os.mkdir(dir)
    
def user_create(n):
    dir=os.path.join(path_cutter()+"\\"+basename(current_directory),"Dataset",n)
    os.mkdir(dir)
    
    
n=str(input("Enter your Name:"))

if not os.path.exists(current_directory+"\\Dataset"):
    directory_create()
    user_create(n)
    capture(current_directory+"\\Dataset"+"\\"+n)
else:
    if os.path.exists(current_directory+"\\Dataset"+"\\"+n):
        print("Okay, So your name is aleardy in the Database :)")
        print("After, that you can add your photos.")
        capture(current_directory+"\\Dataset"+"\\"+n)
    else:
        user_create(n)
        capture(current_directory+"\\Dataset"+"\\"+n)
    
print("Your name is successfully entered into the DataBase, Now Add Your photos")