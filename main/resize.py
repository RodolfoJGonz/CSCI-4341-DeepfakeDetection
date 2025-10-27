
import cv2, glob, os

def resize_faces(input_dir, output_dir, size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    for img_path in glob.glob(f"{input_dir}/face_*.jpg"):
        img = cv2.imread(img_path)
        resized = cv2.resize(img, size)
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, filename), resized)



input_dir = "./data/manipulated_sequences/Deepfakes/c23/vid_frames"
output_dir = "./data/manipulated_sequences/Deepfakes/c23/vid_frames_resized"

resize_faces(input_dir, output_dir)
