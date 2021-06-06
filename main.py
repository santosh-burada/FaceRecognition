from flask import Flask, render_template, request, Response, send_file
import json
import cv2
import face_recognition
import os
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

data = {'people': []}

attendance = []


def attendence(name, depart, Yr, id, OPEN):
    with open(OPEN, "r+", newline="\n") as file:
        filedata = file.readlines()
        Ids_lst = []
        for data in filedata:
            candidate_details = data.split(",")
            Ids_lst.append(candidate_details[0])
        if id not in Ids_lst:
            t = datetime.now()
            d = t.strftime("%d/%m/%Y")
            dstr = t.strftime("%H:%M:%S")
            file.writelines(f"\n{id},{name},{depart},{Yr},{dstr},{d},present")


@app.route("/stu", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def stureg():
    msg = ''
    if request.method == "POST":
        print("In post")
        username = request.form["name"]
        roll = request.form["roll"]
        email = request.form["email"]
        phone = request.form["phone"]
        address = request.form["address"]
        mentor = request.form["mentor"]
        dep = request.form["dep"]
        sem = request.form["sem"]
        gender = request.form["gender"]
        batch = request.form["batch"]
        if request.form.get("upload"):
            # do something when upload button clicked
            directory = str(roll)
            parent_dir = r"static/images"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
            directory = r"static/images/" + str(roll)
            face_classifer = cv2.CascadeClassifier(r"static/haarcascade_frontalface_default.xml")
            os.chdir(directory)

            def Crop(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifer.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    facecrop = img[y:y + h, x:x + w]
                    return facecrop

            cap = cv2.VideoCapture(0)
            counter = 0

            while True:

                ret, frame = cap.read()
                cv2.imshow("frame", frame)
                k = cv2.waitKey(1)
                if Crop(frame) is not None:
                    counter += 1
                    face = cv2.resize(Crop(frame), (450, 450))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                    if k % 256 == 27:
                        break
                    elif k % 256 == 32:

                        filepath = str(roll) + "_" + str(counter) + ".jpeg"
                        cv2.imwrite(filepath, face)
                        cv2.imshow("Cropped Face", face)
            cap.release()
            cv2.destroyAllWindows()
        msg = "Data Recorded!!!"
        os.chdir(r"FaceRecognition")
        with open('static/data/data.json', 'r+') as outfile:
            flag = 0
            data = json.load(outfile)
            for p in data['people']:
                if p['Id'] == roll:
                    flag = 1

            if not flag:
                data['people'].append({
                    'Name': username,
                    'Id': roll,
                    'Email': email,
                    'Phone': phone,
                    'Address': address,
                    'Mentor': mentor,
                    'Department': dep,
                    'Semester': sem,
                    'Gender': gender,
                    'Batch': batch,
                })
                json.dump(data, outfile)

    else:
        msg = "noting recorded"

    return render_template('Home.html', title="Student Registration", msg=msg)


@app.route("/train", methods=["GET", "POST"])
def train():
    msg = ''
    if request.method == "POST":
        if request.form.get("train"):
            msg = "started"
            os.chdir(r"static/images")
            images = []
            imgNames = []
            # Now we have to go through every photo in multiple folders
            for root, dirs, files in os.walk(".", topdown=False):

                for name in files:
                    imgNames.append(name.split("_")[0])
                    images.append(cv2.imread(os.path.join(root, name)))
            print(imgNames, "Imagnmaes")

            # This function is used to collect the features and face location of particular image.
            def features(images):
                featuresOfImages = []
                c = 0
                print("Face Locations of Loaded Images")
                for img in images:
                    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if len(face_recognition.face_locations(imgs)) > 0:
                        print(face_recognition.face_locations(imgs)[0])
                    try:
                        featuresOfImg = face_recognition.face_encodings(imgs, model='cnn')[0]
                    except IndexError as e:
                        print("Some Faces are not detected by dlib")
                        # sys.exit(1)
                    featuresOfImages.append(featuresOfImg)

                return featuresOfImages

            featuresOfTrainingImages = features(
                images)  # In this featuresOfTrainingImages list we will have the features of all
            # loaded images
            print(type(featuresOfTrainingImages))
            print("Features are collected...", len(featuresOfTrainingImages))
            os.chdir(r"FaceRecognition")
            with open("featuresOfTrainingImages.txt", "wb") as fp:  # Pickling
                pickle.dump(featuresOfTrainingImages, fp)
            with open("images.txt", "wb") as fp:  # Pickling
                pickle.dump(images, fp)
            with open("imgNames.txt", "wb") as fp:  # Pickling
                pickle.dump(imgNames, fp)
            msg = "completed"

    return render_template("train.html", msg=msg)


@app.route("/recog", methods=["GET", "POST"])
def recog():
    if request.method == "POST":
        if request.form.get("stop"):
            # this statement helps to stop the camera flash after the camera.release() statement
            os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
            cv2.VideoCapture(0).release()
    return render_template('attendance.html', attendance=attendance)


@app.route("/download", methods=["GET", "POST"])
def download_file():
    path = "Attendance.csv"
    return send_file(path, as_attachment=True)


# this statement helps to stop the camera flash after the camera.release() statement
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


def gen():
    camera = cv2.VideoCapture(0)
    msg = ''
    face_classifer = cv2.CascadeClassifier(r"static/haarcascade_frontalface_default.xml")
    c = 0
    while True:

        status, img = camera.read()
        c += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifer.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        c += 1
        imgInput = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)

        try:

            faceLocation = face_recognition.face_locations(imgInput)
            if len(faceLocation) > 0:

                inputFeatures = face_recognition.face_encodings(imgInput, faceLocation, model='cnn')
                # Here we wil have the input face features
                with open("featuresOfTrainingImages.txt", "rb") as fp:  # Unpickling
                    featuresOfTrainingImages = pickle.load(fp)
                with open("imgNames.txt", "rb") as fp:
                    imgNames = pickle.load(fp)
                with open("images.txt", "rb") as fp:
                    images = pickle.load(fp)
                # matching the input feature with the loaded images features.

                for encodeInput, (x, y, w, h) in zip(inputFeatures, faceLocation):

                    faceDistance = face_recognition.face_distance(featuresOfTrainingImages, encodeInput)

                    index = np.argmin(faceDistance)
                    os.chdir(r"FaceRecognition")
                    with open('static/data/data.json', 'r+') as outfile:

                        data = json.load(outfile)
                        for p in data['people']:

                            if p['Id'] == imgNames[index]:
                                msgk = f"{p['Id']}+{p['Name']}"
                                attendence(p['Name'], p['Department'], p['Batch'], p['Id'], "Attendance.csv")


        except Exception as e:
            print(f" {e}")
        ret, jpeg = cv2.imencode('.jpg', img)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(jpeg) + b'\r\n\r\n')


@app.route('/video', methods=["GET", "POST"])
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/showdata")
def showdata():
    return render_template("showdata.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
