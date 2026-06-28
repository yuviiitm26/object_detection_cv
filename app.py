import cv2
from ultralytics import YOLO

# 1. Load the YOLO model 
# (This will automatically download the lightweight 'nano' model the first time you run it)
model = YOLO('yolo26n.pt')

# 2. Open the webcam 
# (0 is usually the default built-in laptop camera. If you have an external USB cam, try 1)
cap = cv2.VideoCapture(0)

print("Starting the Desk Scanner... Press 'q' to quit.")

while True:
    # 3. Read a single frame from the webcam
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab a frame from the webcam.")
        break
        
    # 4. Run YOLO object detection on the frame
    # stream=True is recommended for live video to keep memory usage low
    results = model(frame, stream=True)
    
    # 5. Visualize the results on the frame
    for r in results:
        # The plot() function does all the heavy lifting: drawing boxes, labels, and scores
        annotated_frame = r.plot()
        
    # 6. Display the annotated frame in a pop-up window
    cv2.imshow("Real-Time Desk Scanner", annotated_frame)
    
    # 7. Listen for the 'q' key to break the loop and close the software
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up: Release the webcam hardware and destroy the window
cap.release()
cv2.destroyAllWindows()