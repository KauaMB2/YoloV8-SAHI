import cv2
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

def detect_objects(video_path='data/vehicles.mp4', weights='yolov8s.pt'):
    # Verify if the video path exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video path '{video_path}' does not exist.")

    # Path for the YOLOv8 model
    yolov8_model_path = f'models/{weights}'
    download_yolov8s_model(yolov8_model_path)
    
    # Initialize the SAHI detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device='cpu'  # Set to 'cuda' for GPU usage
    )

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if there are no frames
        frame = cv2.resize(frame, (640, 440))#Resize the frame
        # Make predictions on the current frame
        results = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # Draw bounding boxes and labels on the frame
        for obj in results.object_prediction_list:
            box = obj.bbox
            cls = obj.category.name
            confidence = obj.score
            
            try:
                confidence_value = confidence.value
            except AttributeError:
                confidence_value = float(confidence)

            x1, y1, x2, y2 = int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (56, 56, 255), 2)
            label = f"{cls} {confidence_value:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), (56, 56, 255), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Display the frame with detections
        cv2.imshow('Detected Objects', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_objects()
