import cv2
import numpy as np
import mediapipe as mp
import math
import argparse
from tqdm import tqdm
import os

class PoseAnalyzer:
    def __init__(self, use_gpu=True):
        # Enable GPU usage for MediaPipe
        self.use_gpu = use_gpu
        """
        if self.use_gpu:
            # Set environment variable to force MediaPipe to use GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            mp_gpu_status = "Enabled" if mp.solutions.pose._POSE_GPU_SUPPORTED else "Not supported"
            print(f"MediaPipe GPU acceleration: {mp_gpu_status}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
            print("Using CPU only (GPU disabled)")
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create Pose instance with appropriate settings for better performance
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use 1 for better performance, 2 for better accuracy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define colors
        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "purple": (255, 0, 255),
            "white": (255, 255, 255)
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        self.font_line_type = cv2.LINE_AA  # Anti-aliased line for smoother text
        
        # Store previous positions for stride calculation
        self.prev_positions = {}
        self.stride_distances = []
        self.step_heights = []
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (in degrees)"""
        if not all(p.visibility > 0.5 for p in [a, b, c]):
            return None
            
        a_coords = np.array([a.x, a.y])
        b_coords = np.array([b.x, b.y])
        c_coords = np.array([c.x, c.y])
        
        ba = a_coords - b_coords
        bc = c_coords - b_coords
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_distance(self, a, b):
        """Calculate distance between two points"""
        if not all(p.visibility > 0.5 for p in [a, b]):
            return None
            
        return math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)
    
    def check_horizontality(self, a, b):
        """Calculate horizontality (angle from horizontal in degrees)"""
        if not all(p.visibility > 0.5 for p in [a, b]):
            return None
            
        dx = b.x - a.x
        dy = b.y - a.y
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def calculate_vertical_lift(self, ankle, knee, hip, prev_ankle_y=None):
        """Calculate how much a leg is lifted"""
        if not all(p.visibility > 0.5 for p in [ankle, knee, hip]):
            return None, None
            
        # Current height relative to hip
        current_lift = hip.y - ankle.y
        
        # Change in height from previous frame
        lift_change = None
        if prev_ankle_y is not None:
            lift_change = prev_ankle_y - ankle.y
            
        return current_lift, lift_change
    
    def calculate_stride(self, left_ankle, right_ankle, frame_height):
        """Calculate stride distance between steps"""
        if not all(p.visibility > 0.5 for p in [left_ankle, right_ankle]):
            return None
            
        # Convert to pixel coordinates
        left_pos = (int(left_ankle.x * frame_height), int(left_ankle.y * frame_height))
        right_pos = (int(right_ankle.x * frame_height), int(right_ankle.y * frame_height))
        
        # Store current positions
        current_positions = {
            "left_ankle": left_pos,
            "right_ankle": right_pos
        }
        
        # Calculate stride if we have previous positions
        stride = None
        if "left_ankle" in self.prev_positions and "right_ankle" in self.prev_positions:
            # Check which foot moved (the one with larger displacement)
            left_disp = np.linalg.norm(np.array(left_pos) - np.array(self.prev_positions["left_ankle"]))
            right_disp = np.linalg.norm(np.array(right_pos) - np.array(self.prev_positions["right_ankle"]))
            
            if max(left_disp, right_disp) > 5:  # Threshold to detect actual step
                stride = max(left_disp, right_disp)
                if stride > 5:  # Reasonable stride detected
                    self.stride_distances.append(stride)
        
        # Update previous positions
        self.prev_positions = current_positions
        return stride
    
    def draw_text(self, frame, text, position, color, background=True):
        """Draw text with custom font settings and optional background"""
        # Get text size
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
        
        # Add background rectangle for better readability
        if background:
            # Draw semi-transparent background
            bg_rect = (
                position[0] - 5,
                position[1] - text_size[1] - 5,
                text_size[0] + 10,
                text_size[1] + 10
            )
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                         (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
                         (0, 0, 0), -1)
            # Apply semi-transparent background
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text (try to use JetBrainsMono if available)
        try:
            cv2.putText(
                frame, text, position, self.font, self.font_scale, color, 
                self.font_thickness, self.font_line_type
            )
        except Exception:
            # Fallback to default font if custom font fails
            cv2.putText(
                frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 
                self.font_thickness, self.font_line_type
            )
    
    def process_video(self, input_video, output_video=None):
        if output_video is None:
            # Create output filename by adding "_analyzed" before extension
            name_parts = input_video.rsplit('.', 1)
            output_video = f"{name_parts[0]}_analyzed.{name_parts[1]}"
        
        # Open the video file
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        prev_ankle_y = {"left": None, "right": None}
        
        # Process frames in batches for better performance
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Pose
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw the pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                landmarks = results.pose_landmarks.landmark
                
                # Extract key points
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
                left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                
                # 1. Calculate shoulder horizontality
                shoulder_angle = self.check_horizontality(left_shoulder, right_shoulder)
                if shoulder_angle is not None:
                    # Draw shoulder line
                    cv2.line(
                        frame, 
                        (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height)),
                        (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height)),
                        self.colors["blue"], 
                        2
                    )
                    # Add shoulder horizontality info
                    shoulder_text = f"Shoulder angle: {shoulder_angle:.1f}°"
                    self.draw_text(
                        frame, 
                        shoulder_text,
                        (30, 50), 
                        self.colors["white"]
                    )
                
                # 2. Calculate hip horizontality
                hip_angle = self.check_horizontality(left_hip, right_hip)
                if hip_angle is not None:
                    # Draw hip line
                    cv2.line(
                        frame, 
                        (int(left_hip.x * frame_width), int(left_hip.y * frame_height)),
                        (int(right_hip.x * frame_width), int(right_hip.y * frame_height)),
                        self.colors["green"], 
                        2
                    )
                    # Add hip horizontality info
                    hip_text = f"Hip angle: {hip_angle:.1f}°"
                    self.draw_text(
                        frame, 
                        hip_text,
                        (30, 100), 
                        self.colors["white"]
                    )
                
                # 3. Calculate leg lift for left leg
                left_lift, left_lift_change = self.calculate_vertical_lift(
                    left_ankle, left_knee, left_hip, prev_ankle_y["left"]
                )
                if left_lift is not None:
                    # Update previous ankle position
                    prev_ankle_y["left"] = left_ankle.y
                    # Normalize lift to frame height
                    normalized_lift = left_lift * frame_height
                    if normalized_lift > 0:
                        self.step_heights.append(normalized_lift)
                    # Add left leg lift info
                    left_lift_text = f"Left leg lift: {normalized_lift:.1f}px"
                    self.draw_text(
                        frame,
                        left_lift_text,
                        (30, 150),
                        self.colors["yellow"]
                    )
                
                # 4. Calculate leg lift for right leg
                right_lift, right_lift_change = self.calculate_vertical_lift(
                    right_ankle, right_knee, right_hip, prev_ankle_y["right"]
                )
                if right_lift is not None:
                    # Update previous ankle position
                    prev_ankle_y["right"] = right_ankle.y
                    # Normalize lift to frame height
                    normalized_lift = right_lift * frame_height
                    if normalized_lift > 0:
                        self.step_heights.append(normalized_lift)
                    # Add right leg lift info
                    right_lift_text = f"Right leg lift: {normalized_lift:.1f}px"
                    self.draw_text(
                        frame,
                        right_lift_text,
                        (30, 200),
                        self.colors["yellow"]
                    )
                
                # 5. Calculate stride
                stride = self.calculate_stride(left_ankle, right_ankle, frame_height)
                if stride is not None and stride > 5:  # Reasonable stride
                    # Add stride info
                    stride_text = f"Stride: {stride:.1f}px"
                    self.draw_text(
                        frame,
                        stride_text,
                        (30, 250),
                        self.colors["purple"]
                    )
                
                # 6. Display average values
                if len(self.stride_distances) > 0:
                    avg_stride = sum(self.stride_distances) / len(self.stride_distances)
                    avg_stride_text = f"Avg stride: {avg_stride:.1f}px"
                    self.draw_text(
                        frame,
                        avg_stride_text,
                        (frame_width - 300, 50),
                        self.colors["white"]
                    )
                
                if len(self.step_heights) > 0:
                    avg_height = sum(self.step_heights) / len(self.step_heights)
                    avg_height_text = f"Avg leg lift: {avg_height:.1f}px"
                    self.draw_text(
                        frame,
                        avg_height_text,
                        (frame_width - 300, 100),
                        self.colors["white"]
                    )
            
            # Write the frame to output video
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        print(f"Analysis complete. Output saved to {output_video}")
        
        # Return summary stats
        summary = {
            "avg_stride": sum(self.stride_distances) / len(self.stride_distances) if self.stride_distances else 0,
            "max_stride": max(self.stride_distances) if self.stride_distances else 0,
            "avg_leg_lift": sum(self.step_heights) / len(self.step_heights) if self.step_heights else 0,
            "max_leg_lift": max(self.step_heights) if self.step_heights else 0
        }
        return summary


def try_load_custom_font():
    """Try to load JetBrainsMono font if available"""
    try:
        # Try to find JetBrainsMono font
        font_paths = [
            "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Light.ttf",
            "C:\\Windows\\Fonts\\JetBrainsMono-Light.ttf",
            os.path.expanduser("~/Library/Fonts/JetBrainsMono-Light.ttf"),
            os.path.expanduser("~/.local/share/fonts/JetBrainsMono-Light.ttf")
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                print(f"Found JetBrainsMono font at: {font_path}")
                return True
                
        return False
    except Exception as e:
        print(f"Error checking for custom font: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze pose in a video')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file (optional)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage instead of GPU')
    
    args = parser.parse_args()
    
    # Check for custom font
    has_custom_font = try_load_custom_font()
    if has_custom_font:
        print("JetBrainsMono font is available")
    else:
        print("JetBrainsMono font not found, using default fonts")
    
    analyzer = PoseAnalyzer(use_gpu=not args.cpu)
    summary = analyzer.process_video(args.input_video, args.output)
    
    print("\nAnalysis Summary:")
    print(f"Average Stride Distance: {summary['avg_stride']:.2f} pixels")
    print(f"Maximum Stride Distance: {summary['max_stride']:.2f} pixels")
    print(f"Average Leg Lift: {summary['avg_leg_lift']:.2f} pixels")
    print(f"Maximum Leg Lift: {summary['max_leg_lift']:.2f} pixels")
