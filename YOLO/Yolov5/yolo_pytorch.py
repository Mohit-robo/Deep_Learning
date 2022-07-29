import torch
import numpy as np
import cv2
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/silicon/Documents/Paper_review_notes/YOLO/Yolov5/exp3/weights/last.pt', force_reload=True)

cap = cv2.VideoCapture('/home/silicon/Documents/Paper_review_notes/YOLO/Yolov5/data/test_0.mp4')
writer = cv2.VideoWriter("output/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,(1280,720))

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    start_time = time.time()
    # Make detections 
    results = model(frame)

    fps = 1.0 / (time.time() - start_time)
    
    cv2.putText(frame, str("FPS: %.2f" % fps),(30,30),0, 0.75, (0,0,255),2)
    
    writer.write(np.squeeze(results.render()))
    cv2.imshow('YOLO', np.squeeze(results.render()))
    # cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()