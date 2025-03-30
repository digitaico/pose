# import dependencies
import cv2
import torch
import time

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# variables for labeling
font_family = cv2.FONT_HERSHEY_DUPLEX
font_scale = 1.0
font_thickness = 2

# test GPU
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

# test versions
print(f"PyTorch Version: {torch.__version__}")
#print(f"TorchVision Version: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version (reported by torch): {torch.version.cuda}")

# Force the use of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# configure the Pose estimation Model, using COCO / human body trained model
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
# Initialize predictor
predictor = DefaultPredictor(cfg)

# Initialize Video Capture
video_path = "sources/jea-1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output/jea-1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

start = time.time()
print(f"without amp: {time.time() - start:.3f} seconds") # Extract keypints from predictor output

    # Loop through video frames and apply predictor
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with torch.autocast(device_type="cuda"):
        outputs = predictor(frame) # list of keypoints for each frame
    print(f"with amp: {time.time() - start:.3f} seconds") # Extract keypints from predictor output
    instances = outputs["instances"].to("cpu") # constins all detected objects (humans). Move to cpu for easier processing

    if len(instances) > 0: # if any human body was detected
        keypoints = instances.pred_keypoints[0] # Get keypoints for first person. pred_keypoints is a tensor of shape 
        # (num_keypoints, 3) where each keypoint has (x,y, confidence) values. 

        # Draw keypoints
        for i in range(len(keypoints)):
            x, y, confidence = keypoints[i]
            if confidence > 0.5: # draw if confidence is high
                cv2.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)
        # Add text label
        cv2.putText(frame, "Pose Estimation", (10,30), font_family, font_scale, (0,255,0), font_thickness )

    out.write(frame)

cap.release()
out.realease()
cv2.destroyAllWindows()