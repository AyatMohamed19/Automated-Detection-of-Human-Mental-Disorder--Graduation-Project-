import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('D:/Ayat_/-fci/مشروع التخرج/model/Emotion_detection_with_CNN-main/weights/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("D:/Ayat_/-fci/مشروع التخرج/model/Emotion_detection_with_CNN-main/weights/model.h5")
print("Loaded model from disk")

# pass here your video path
cap = cv2.VideoCapture("D:/Ayat_/-fci/مشروع التخرج/model/Emotion_detection_with_CNN-main/an.mp4")
cap.set(cv2.CAP_PROP_FPS, 1)

# get frames from videos
def getFrame(start_sec, End_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    hasFrames, image = cap.read()
    while hasFrames:
        if start_sec <= End_sec:
            cv2.imwrite("image/frame " + str(start_sec) + " sec.jpg", image)  # save frame as JPG file
            return image
        else:
            break


#variables to detect first and second predictor
first_predictor = [0,0]
final_prediction = [0, 0, 0]
section_of_first_predictor = ['Normal' ,'Abnormal']
sections_of_data = ['anxious', 'Depressed', 'other disorder']

start_sec =0
End_sec = 59
frameRate = 1/14
Nemotion0=0
Nemotion1=0
Nemotion2 =0
Nemotion3=0
i = 0
while True:
    if i > 826:
        break
    i += 1
    success = getFrame(start_sec, End_sec)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(success, cv2.COLOR_BGR2GRAY)
    start_sec = start_sec + frameRate
    start_sec = round(start_sec, 2)
    
    #detect faces available on frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    # take each face available on the frame and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(success, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        cropped_img=cropped_img/255
        
        #predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        if emotion_dict[maxindex] == 'Fearful' or emotion_dict[maxindex] == 'Disgusted' or emotion_dict[maxindex] == 'Sad' or emotion_dict[maxindex] == 'Angry':
            first_predictor[1] += 1
            if emotion_dict[maxindex] == 'Fearful':
                Nemotion0 += 1
            elif emotion_dict[maxindex] == 'Disgusted':
                Nemotion1 += 1
            elif emotion_dict[maxindex] == 'Sad':
                Nemotion2 += 1
            elif emotion_dict[maxindex] == 'Angry':
                Nemotion3 += 1
        elif emotion_dict[maxindex] == 'Happy' or emotion_dict[maxindex] == 'Neutral' or emotion_dict[maxindex] == 'Surprised':
            first_predictor[0] += 1

    cv2.imshow('Emotion Detection', success)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

x = first_predictor.index(max(first_predictor))
# check if user Normal or Abnormal
print(section_of_first_predictor[x])

# if user Abnormal
if section_of_first_predictor[x]=='Abnormal':
    final=0
    if (Nemotion0 + Nemotion1 )>(Nemotion2+Nemotion3) and Nemotion0 > 0 and Nemotion1 >0:
        final=0
    elif (Nemotion0 + Nemotion1 )<(Nemotion2+Nemotion3) and Nemotion1 > 0 and Nemotion2 >0:
        final=1
    else :
        final =2
    print(sections_of_data[final])
