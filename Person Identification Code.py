import threading
import cv2
from deepface import DeepFace

# Open the video capture
cap = cv2.VideoCapture(0)

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Try to load the reference image
reference_img = cv2.imread("Path of photo")

# Check if the reference image was loaded successfully
if reference_img is None:
    print("Error: Could not load reference image. Check the file path.")
    cap.release()
    cv2.destroyAllWindows()
    exit(1)

def check_face(frame):
    global face_match
    try:
        # Verify the face in the frame with the reference image
        if DeepFace.verify(frame, reference_img)['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 38 == 8:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow("video", frame)
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()