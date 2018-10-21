import numpy as np
import cv2
# import h5py
# from keras.models import load_model
# model = h5py.File(best_trained_weights, -r)
# from keras.models import load_model

# model = load_model('model.h5')
def opencamera():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while 1:
        ret, img = cap.read()
        sum_faces=0
        # print(model.predict(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        sum_faces=np.sum(faces)

        # boxes = model.predict(img)
        # print(boxes)

        for (x,y,w,h) in faces:
        	cv2.rectangle(img, (x,y), (x+w ,y+h), (255,0,0), 2)
        	cv2.putText(img,'Person without Helmet',(x,y), font, .75, (0,0,255), 2, cv2.LINE_AA)

        if sum_faces==0:
        	cv2.putText(img,'Person with Helmet',(x,y), font, .75, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows() 
    return
opencamera()
