from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO
import cv2

model = YOLO("yolov8s(1).pt")

app= Flask(__name__, template_folder="Template")

@app.route('/') 
def index():
    return render_template('index4.html')

# Create an upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

        cap = cv2.VideoCapture(filepath)
        width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use MP4 format

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            # # Draw bounding boxes on the frame
            # for r in results:
            #     for box in r.boxes.xyxy:
            #         x1, y1, x2, y2 = map(int, box[:4])
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            results = model(frame)  #working
            print(results)
            #cv2.waitKey(1)

            res_plotted = results[0].plot()

            out.write(res_plotted)  # Write processed frame to output video

        cap.release()
        out.release()

        
        return render_template("index4.html", video_url=url_for("static", filename="uploads/processed.mp4"), 
                               file_extension=file.filename.rsplit('.', 1)[1].lower())

    return "Upload failed"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
