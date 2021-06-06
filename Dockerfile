FROM ubuntu:18.04
EXPOSE 8000
RUN mkdir -p /app/FaceRecognition
COPY . /app/FaceRecognition
ENV OPENCV_VERSION="3.4.2"
WORKDIR /app/FaceRecognition
RUN apt-get clean
RUN apt-get update -y && apt-get install -y build-essential libboost-all-dev cmake libsm6 libxext6 libxrender-dev python3 python3-pip python3-dev pkg-config git curl wget libjpeg-dev software-properties-common
RUN pip3 install -r requirements.txt
# Install DLIB
RUN git clone -b 'v19.7' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install

# Install Face-Recognition Python Library
RUN git clone https://github.com/ageitgey/face_recognition.git face_recognition/ && \
    cd face_recognition/ && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

CMD cd /app/FaceRecognition && \
    python3 main.py
