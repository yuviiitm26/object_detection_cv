import cv2
from ultralytics import solutions

# 1. Load the custom model you generated in step 3
# YOLO automatically saves your trained model in this 'runs' directory
model_path = "runs/detect/train/weights/best.pt"

# 2. Define a Polygon Zone to catch weaving traffic
# Adjust these (X, Y) coordinates to match the road in your specific video
counting_polygon = [(50, 300), (1000, 300), (950, 450), (100, 450)] 

# 3. Initialize the Object Counter
counter = solutions.ObjectCounter(
    show=True, 
    region=counting_polygon,  
    model=model_path, 
    # Tracking Two-wheelers (10), Autos (6), Cars (0,1,2,3), and Buses (4)
    classes=[0, 1, 2, 3, 4, 6, 10] 
)

# 4. Open your sample video
# Ensure you have a video file named 'bengaluru_traffic.mp4' in your folder
cap = cv2.VideoCapture("bengaluru_traffic.mp4")

print("Starting Traffic Counter... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video stream.")
        break
        
    # Process the frame with the polygon tracking logic
    processed_frame = counter.count(frame)
    
    # Listen for the 'q' key to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()