import os
from ultralytics import YOLO
import cv2

video_path = os.path.join('data/images/test/testvideo1.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

model_path = os.path.join('D:/Users/Kevin/VSC Projects/ski-analyzer/runs/detect/train11/weights/last.pt')
model = YOLO(model_path)

threshold = 0.6

while True:
    ret, frame = cap.read()

    if not ret:
        break  # End the loop if no more frames are available

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Video', frame)  # Display the frame in a window

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()