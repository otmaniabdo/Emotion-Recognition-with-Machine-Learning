import cv2
import glob
import random
import numpy as np

emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier

data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80 - 20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] # 80% of file list
    prediction = files[-int(len(files)*0.2):] # 20% of file list 
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    
    print ("training fisher face classifier")
    print ("size of training set is:", len(training_labels), "images")
    tr_labels = np.array(training_labels)
    fishface.train(training_data, tr_labels)
    print ("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            #print(pred)
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
    
#Now run it
metascore = []
for i in range(0,10):
    print("Set number : %s" %i)
    correct = run_recognizer()
    print ("got", correct, "percent correct!")
    metascore.append(correct)
# test using live video capture
    
cap = cv2.VideoCapture(0)
while True:
    
        ret, frame = cap.read()
        faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(face) == 1:
                    facefeatures = face
        elif len(face_two) == 1:
                    facefeatures = face_two
        elif len(face_three) == 1:
                    facefeatures = face_three
        elif len(face_four) == 1:
                    facefeatures = face_four
        else:
                    facefeatures = ""
                
                #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
                    gray = gray[y:y+h, x:x+w] #Cut the frame to size
                    out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                    pred, conf = fishface.predict(out)
                    print(emotions[pred])
                    cv2.putText(frame, "Subject is %s" %emotions[pred], (100,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
                    cv2.imshow('Facial Expression', frame )
                    cv2.imshow('Face Gray', out)

        if cv2.waitKey(1) == 13:
            break
cap.release()
cv2.destroyAllWindows()
