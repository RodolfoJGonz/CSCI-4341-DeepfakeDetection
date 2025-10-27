from retinaface import RetinaFace
import cv2
import os



def detect_and_crop_face(image_path, output_dir):
    # Read image
    img = cv2.imread(image_path)
    detections = RetinaFace.detect_faces(img)

    if not detections:
        print(f"No face detected in {image_path}")
        return

    # Loop through detected faces
    for i, (key, face) in enumerate(detections.items()):
        facial_area = face["facial_area"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = facial_area

        # Crop and save the face
        cropped_face = img[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"face_{i}.jpg")
        cv2.imwrite(output_path, cropped_face)



def extract_and_detect(video_path, output_dir, frame_skip=32):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_num % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
            cv2.imwrite(frame_path, frame)

            # Detect and crop face
            detect_and_crop_face(frame_path, output_dir)

        frame_num += 1

    cap.release()


video_path = "./data/manipulated_sequences/Deepfakes/c23/videos/033_097.mp4"
output_dir = "./data/manipulated_sequences/Deepfakes/c23/vid_frames/"
extract_and_detect(video_path, output_dir)
