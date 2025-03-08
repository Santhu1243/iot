import cv2
import face_recognition
import os
import numpy as np

# Path to the folder containing known face images
KNOWN_FACES_DIR = "C:\\Users\\santh\\OneDrive\\Desktop\\chezzion-iot\\facelog\\known_faces"

# Lists to store known face encodings and their names
known_faces = []
known_names = []

# Load and encode all images from the folder
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_faces.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])  

# Open webcam
video_capture = cv2.VideoCapture(0)

print("Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame from BGR to RGB (required for face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Draw rectangle and label for each detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Authentication", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
