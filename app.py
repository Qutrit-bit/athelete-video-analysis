from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
import shutil
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path, output_path):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        out.write(frame)
    
    cap.release()
    out.release()

def upload_to_cloudinary(file_path):
    response = cloudinary.uploader.upload(
        file_path, 
        resource_type="video", 
        format="mp4"
    )
    return response["secure_url"]


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"processed_{file.filename}"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    process_video(input_path, output_path)
    cloudinary_url = upload_to_cloudinary(output_path)
    
    os.remove(input_path)
    os.remove(output_path)
    
    return {"cloudinary_url": cloudinary_url}
