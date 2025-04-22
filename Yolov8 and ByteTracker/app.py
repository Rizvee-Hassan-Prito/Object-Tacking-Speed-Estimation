import ultralytics
import supervision as sv
import numpy as np
from ultralytics import YOLO
import gradio as gr
import cv2
import os
import ffmpeg
import supervision as sv

model = YOLO("./best.pt")

CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['candy']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name
    in SELECTED_CLASS_NAMES
]

LINE_START = sv.Point(50, 50)
LINE_END = sv.Point(580 - 50, 326-50)


# Example video processing function
def process_video(input_video):
    # Define output path where the processed video will be saved

    # create BYTETracker instance
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3)

    byte_tracker.reset()

    # create VideoInfo instance
    video_info = sv.VideoInfo.from_video_path(input_video)

    # create frame generator
    generator = sv.get_video_frames_generator(input_video)

    # create LineZone instance, it is previously called LineCounter class
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END )

    # create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

    # create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5, text_color=sv.Color.WHITE, color=sv.Color.BLACK)

    # define call back function to be used in video processing
    with sv.VideoSink(target_path='output.mp4', video_info=video_info, codec='h264' ) as sink:
        for frame in generator:
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            # only consider class id from selected_classes define above
            detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
            # tracking detections
            detections = byte_tracker.update_with_detections(detections)
            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id
                in zip(detections.confidence, detections.class_id, detections.tracker_id)
            ]
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels)

            # update line counter
            line_zone.trigger(detections)
            # return frame with box and line annotated result
            sink.write_frame(frame=line_zone_annotator.annotate(annotated_frame, line_counter=line_zone))
    return "./output.mp4"

    

# Create a Gradio interface
interface = gr.Interface(process_video,
                    inputs=gr.Video(label="Upload Video"),
                    outputs=gr.Video(label="Processed Video")
                    
                    )

# Launch the interface
interface.launch(share=True,debug=True, server_port=80)
