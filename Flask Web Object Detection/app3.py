from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO
import tensorflow as tf
import cv2
from skimage.feature import hog
import time
import numpy as np
import pandas as pd
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric

model = YOLO("yolov8s(1).pt")



def extract_patch(image, bbox, target_size):
    """
    Extracts a patch from an image given a bounding box in (x, y, width, height) format,
    ensuring a uniform shape for every patch.

    :param image: Input image (numpy array).
    :param bbox: Bounding box in (x, y, width, height) format.
    :param target_size: Desired output size (width, height) as a tuple.
    :return: Extracted and resized patch as a numpy array.
    """
    x, y, w, h = bbox
    patch = image[y:y+h, x:x+w]

    # Resize to target size while maintaining uniform shape
    patch_resized = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
    patch_vector = patch_resized.flatten()

    return patch_resized
def extract_features(patch):
    """
    Extracts features from the given image patch using Histogram of Oriented Gradients (HOG).

    :param patch: Input image patch (numpy array)
    :return: Feature vector as a 1D numpy array
    """
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features = hog(patch_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return features


class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

app= Flask(__name__, template_folder="Template")

@app.route('/') 
def index():
    return render_template('index4.html')

# Create an upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

count=0
@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        nn=NearestNeighborDistanceMetric("cosine", 0.4)
        tracker=Tracker(nn)
        count=0
        cap=cv2.VideoCapture(filepath)
        down = {}
        up = {}
        counter_down = []
        counter_up = []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height
        red_line_y =  int(frame_height * 0.45)
        blue_line_y = int(frame_height * 0.6)
        offset = 50

        output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use MP4 format
        out = cv2.VideoWriter(output_path, fourcc, 20.0,(frame_width,frame_height) )

        while True:
            ret, frame = cap.read()
            if not ret:
                break


            # # Draw bounding boxes on the frame
            # for r in results:
            #     for box in r.boxes.xyxy:
            #         x1, y1, x2, y2 = map(int, box[:4])
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frame = cv2.resize(frame, (frame_width,frame_height))

            results = model.predict(frame)
            a = results[0].boxes.data
            a = a.detach().cpu().numpy()
            px = pd.DataFrame(a).astype("float")
            list1 = []

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                w, h = x2 - x1, y2 - y1
                if 'car' in c:
                    patch=extract_patch(frame,[x1,y1,x2,y2],(128,128))

                    f_v=extract_features(patch)
                    det=Detection([x1,y1,w,h], int(row[4]), f_v)
                    list1.append(det)

            tracker.predict()
            tracker.update(list1)

            bbox_id=[]
            # DeepSORT -> Plotting the tracks.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

            # DeepSORT -> Changing track bbox to top left, bottom right coordinates.
                bbox = list(track.to_tlbr())
                bbox = list(map(int, bbox))
                bbox.append(track.track_id)
                bbox_id.append(bbox)
            # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
            # txt = 'id:' + str(track.track_id)


            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2

                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame,str(id),(x3-2,y3-2),cv2.FONT_HERSHEY_COMPLEX,0.9,(0, 0, 255),2)
                #cv2_imshow( frame)


                if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                    down[id]=time.time()   # current time when vehichle touch the first line
                if id in down:

                    if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                        elapsed_time=time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)

                        if counter_down.count(id)==0:
                            counter_down.append(id)
                            distance = 10 # meters
                            a_speed_ms = distance / elapsed_time
                            a_speed_kh = a_speed_ms * 3.6  # this will give kilometers per hour for each vehicle. This is the condition for going downside
                            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                            # cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                            cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                            count+=1
                            img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{count}.jpg")
                            cv2.imwrite(img_path,frame)

                #####going UP blue line#####
                if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                    up[id]=time.time()
                
                if id in up:
                    if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                        elapsed1_time=time.time() - up[id]
                        # formula of speed= distance/time

                        if counter_up.count(id)==0:
                            counter_up.append(id)
                            distance1 = 10 # meters  (Distance between the 2 lines is 10 meters )
                            a_speed_ms1 = distance1 / elapsed1_time
                            a_speed_kh1 = a_speed_ms1 * 3.6
                            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                            # cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                            cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                            count+=1
                            img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{count}.jpg")
                            cv2.imwrite(img_path,frame)

            text_color = (0, 0, 0)  # Black color for text
            yellow_color = (0, 255, 255)  # Yellow color for background
            red_color = (0, 0, 255)  # Red color for lines
            blue_color = (255, 0, 0)  # Blue color for lines

            cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

            #cv2.line(frame, (172, 198), (774, 198), red_color, 2)
            cv2.line(frame, (0, red_line_y), (frame_width, red_line_y), red_color, 4)
            cv2.putText(frame, ('Red Line'), (0, red_line_y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2, cv2.LINE_AA)

            #cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
            cv2.line(frame, (0, blue_line_y), (frame_width, blue_line_y), blue_color, 4)
            cv2.putText(frame, ('Blue Line'), (0, blue_line_y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, blue_color, 2, cv2.LINE_AA)

            cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            #Save frame
            #frame_filename = f'/detected_frames/frame_{count}.jpg'


            out.write(frame)

        

        images = [img for img in os.listdir("static/uploads") if img.endswith(".jpg")]
        return render_template("index4.html", video_url=url_for("static", filename="uploads/processed.mp4"), 
                               file_extension=file.filename.rsplit('.', 1)[1].lower(), images=images)
        

    return "Upload failed"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
