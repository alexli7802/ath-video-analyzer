import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import threading
import time
from PIL import Image, ImageTk
import os
import numpy as np

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")
        self.root.geometry("800x600")
        
        self.video_file = None
        self.cap = None
        self.playing = False
        self.paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.detection_enabled = True
        self.last_detections = []  # Cache last detections
        self.detection_frame_skip = 3  # Only detect every N frames
        
        # Tracking variables
        self.tracker = None
        self.tracking_active = False
        self.frames_since_detection = 0
        self.redetect_interval = 15  # Re-detect every 15 frames for fast-moving athletes
        
        # Initialize person detection
        self.init_detection()
        self.init_motion_detection()
        
        self.setup_ui()
    
    def init_detection(self):
        """Initialize person detection using OpenCV's built-in HOG descriptor"""
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception as e:
            print(f"Warning: Could not initialize person detection: {e}")
            self.hog = None
    
    def init_motion_detection(self):
        """Initialize background subtraction for motion detection"""
        try:
            # Background subtractor for motion detection
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=50,  # More sensitive to motion
                history=20        # Shorter history for faster adaptation
            )
            self.prev_frame = None
            print("Motion detection initialized")
        except Exception as e:
            print(f"Warning: Could not initialize motion detection: {e}")
            self.bg_subtractor = None
    
    def init_tracker(self, frame, bbox):
        """Initialize simple centroid-based tracker"""
        try:
            x, y, w, h = bbox
            self.tracker = {
                'bbox': (x, y, w, h),
                'centroid': (x + w//2, y + h//2),
                'prev_frame': frame.copy()
            }
            self.tracking_active = True
            self.frames_since_detection = 0
            print(f"Simple tracker initialized at frame {self.current_frame}")
            return True
        except Exception as e:
            print(f"Tracker initialization error: {e}")
            self.tracking_active = False
            return False
    
    def update_tracker(self, frame):
        """Update simple tracker using template matching"""
        if not self.tracker:
            return False, None
        
        try:
            x, y, w, h = self.tracker['bbox']
            prev_frame = self.tracker['prev_frame']
            
            # Extract template from previous frame with padding for robustness
            pad = 5
            template_y1 = max(0, y - pad)
            template_x1 = max(0, x - pad) 
            template_y2 = min(prev_frame.shape[0], y + h + pad)
            template_x2 = min(prev_frame.shape[1], x + w + pad)
            
            template = prev_frame[template_y1:template_y2, template_x1:template_x2]
            if template.size == 0:
                return False, None
            
            # Larger search margin for fast-moving athletes
            search_margin = 120
            x1 = max(0, x - search_margin)
            y1 = max(0, y - search_margin)
            x2 = min(frame.shape[1], x + w + search_margin)
            y2 = min(frame.shape[0], y + h + search_margin)
            
            search_region = frame[y1:y2, x1:x2]
            if search_region.size == 0:
                return False, None
            
            # Try multiple template matching approaches for robustness
            best_val = 0
            best_loc = None
            
            # Standard template matching
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
            
            # Try with slightly scaled template for size variations
            for scale in [0.9, 1.1]:
                try:
                    scaled_h, scaled_w = int(template.shape[0] * scale), int(template.shape[1] * scale)
                    if scaled_h > 10 and scaled_w > 10:
                        scaled_template = cv2.resize(template, (scaled_w, scaled_h))
                        result = cv2.matchTemplate(search_region, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        if max_val > best_val:
                            best_val = max_val
                            best_loc = max_loc
                except:
                    continue
            
            # Lower threshold for fast-moving athletes (they change appearance quickly)
            if best_val > 0.25 and best_loc is not None:  # More lenient threshold for athletes
                new_x = x1 + best_loc[0]
                new_y = y1 + best_loc[1]
                new_bbox = (new_x, new_y, w, h)
                
                self.tracker['bbox'] = new_bbox
                self.tracker['centroid'] = (new_x + w//2, new_y + h//2)
                self.tracker['prev_frame'] = frame.copy()
                
                return True, new_bbox
            else:
                return False, None
                
        except Exception as e:
            print(f"Tracker update error: {e}")
            return False, None
    
    def detect_persons(self, frame):
        """Detect persons in frame and return bounding boxes"""
        if not self.hog:
            return []
        
        try:
            # Convert to grayscale for better HOG performance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Skip expensive histogram equalization for performance
            # gray = cv2.equalizeHist(gray)
            
            # Use single, optimized detection pass
            boxes, weights = self.hog.detectMultiScale(
                gray, 
                winStride=(8,8),    # Balance between accuracy and speed
                padding=(16,16),    
                scale=1.05,         # Reasonable scale step
                hitThreshold=0.0    # Lower hit threshold for better detection
            )
            
            # Filter detections by confidence
            persons = []
            for (x, y, w, h), weight in zip(boxes, weights):
                if weight > 0.3:  # Balanced confidence threshold
                    persons.append((x, y, x+w, y+h))
            
            return persons
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_motion(self, frame):
        """Detect moving objects using background subtraction"""
        if not self.bg_subtractor:
            return []
        
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Remove shadows
            fg_mask[fg_mask == 127] = 0  # Remove shadow pixels
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter by area - human-sized objects
                if 500 < area < 20000:  # Adjust based on video resolution
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter by aspect ratio - roughly human proportions
                    aspect_ratio = h / w if w > 0 else 0
                    if 1.2 < aspect_ratio < 4.0:  # Tall objects like people
                        detections.append((x, y, x+w, y+h))
            
            return detections
        except Exception as e:
            print(f"Motion detection error: {e}")
            return []
    
    def detect_optical_flow(self, frame):
        """Detect moving objects using optical flow"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return []
        
        try:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_gray, None, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Create motion magnitude image
            flow_magnitude = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_gray, None, None
            )
            
            # Alternative: Use dense optical flow for motion detection
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current_gray, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate magnitude of flow vectors
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Threshold to find areas with significant motion
            motion_thresh = np.mean(magnitude) + 2 * np.std(magnitude)
            motion_mask = (magnitude > motion_thresh).astype(np.uint8) * 255
            
            # Find contours in motion areas
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 300 < area < 15000:  # Human-sized motion areas
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append((x, y, x+w, y+h))
            
            self.prev_frame = current_gray
            return detections
            
        except Exception as e:
            print(f"Optical flow detection error: {e}")
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return []
    
    def non_max_suppression(self, boxes, overlap_threshold):
        """Apply non-maximum suppression to remove duplicate detections"""
        if len(boxes) == 0:
            return []
        
        # Convert to format needed for NMS
        boxes_array = np.array(boxes, dtype=np.float32)
        
        # Calculate areas
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
        
        # Sort by bottom-right y-coordinate
        indices = np.argsort(boxes_array[:, 3])
        
        keep = []
        while len(indices) > 0:
            # Pick the last index
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)
            
            # Find the largest coordinates for the intersection rectangle
            xx1 = np.maximum(boxes_array[i, 0], boxes_array[indices[:last], 0])
            yy1 = np.maximum(boxes_array[i, 1], boxes_array[indices[:last], 1])
            xx2 = np.minimum(boxes_array[i, 2], boxes_array[indices[:last], 2])
            yy2 = np.minimum(boxes_array[i, 3], boxes_array[indices[:last], 3])
            
            # Compute the width and height of the intersection rectangle
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            # Compute the intersection over union
            intersection = w * h
            union = areas[i] + areas[indices[:last]] - intersection
            overlap = intersection / union
            
            # Delete indices with overlap greater than threshold
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
        
        return [boxes[i] for i in keep]
    
    def detect_athletes_multi_method(self, frame):
        """Combined detection using multiple methods for better athlete detection"""
        all_detections = []
        
        # Method 1: HOG person detection (works for upright poses)
        if self.hog:
            hog_detections = self.detect_persons(frame)
            all_detections.extend(hog_detections)
        
        # Method 2: Motion detection (good for moving athletes)
        motion_detections = self.detect_motion(frame)
        all_detections.extend(motion_detections)
        
        # Method 3: Optical flow detection (sensitive to movement)
        # Skip optical flow every few frames for performance
        if self.current_frame % 3 == 0:
            flow_detections = self.detect_optical_flow(frame)
            all_detections.extend(flow_detections)
        
        # Remove duplicates and overlapping detections
        if len(all_detections) > 1:
            all_detections = self.non_max_suppression(all_detections, 0.5)
        
        # Filter by confidence - prefer detections that multiple methods agree on
        final_detections = []
        if len(all_detections) > 0:
            # If we have multiple detections, prefer the largest (likely most complete)
            if len(all_detections) > 1:
                largest = max(all_detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
                final_detections = [largest]
            else:
                final_detections = all_detections
        
        return final_detections
    
    def draw_bounding_boxes(self, frame, detections):
        """Draw bounding boxes around detected persons"""
        for i, (x1, y1, x2, y2) in enumerate(detections):
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Athlete {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def toggle_detection(self):
        """Toggle athlete detection on/off"""
        self.detection_enabled = self.detection_var.get()
        if not self.detection_enabled:
            # Reset tracking when detection is disabled
            self.tracking_active = False
            self.tracker = None
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection frame
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Select Video File", command=self.select_video_file).pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Video display frame
        self.video_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = ttk.Label(self.video_frame, text="Select a video file to start playing")
        self.video_label.pack(expand=True)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)
        
        # Progress bar
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(progress_frame, from_=0, to=100, 
                                     variable=self.progress_var, 
                                     orient=tk.HORIZONTAL,
                                     command=self.seek_video)
        self.progress_bar.pack(fill=tk.X, padx=(0, 10), side=tk.LEFT)
        
        self.time_label = ttk.Label(progress_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT)
        
        # Video info frame
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.fps_label = ttk.Label(info_frame, text="")
        self.fps_label.pack(side=tk.LEFT)
        
        self.frames_label = ttk.Label(info_frame, text="")
        self.frames_label.pack(side=tk.RIGHT)
        
        # Detection controls
        detection_frame = ttk.Frame(control_frame)
        detection_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.detection_var = tk.BooleanVar(value=True)
        self.detection_checkbox = ttk.Checkbutton(detection_frame, 
                                                 text="Enable Athlete Detection", 
                                                 variable=self.detection_var,
                                                 command=self.toggle_detection)
        self.detection_checkbox.pack(side=tk.LEFT)
        
        # Initially disable progress bar
        self.progress_bar.config(state="disabled")
        
    def select_video_file(self):
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=file_types
        )
        
        if filename:
            self.video_file = filename
            self.file_label.config(text=os.path.basename(filename))
            self.load_video()
            
    def load_video(self):
        if self.cap:
            self.cap.release()
            
        try:
            self.cap = cv2.VideoCapture(self.video_file)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.current_frame = 0
            
            # Reset tracking when loading new video
            self.tracking_active = False
            self.tracker = None
            self.frames_since_detection = 0
            
            self.progress_bar.config(to=self.total_frames-1, state="normal")
            
            # Display first frame and start playback automatically
            self.display_frame()
            self.update_time_label()
            self.update_video_info()
            self.play_video()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
            
    def play_video(self):
        if not self.cap:
            return
            
        self.playing = True
        self.paused = False
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()
            
    def playback_loop(self):
        frame_time = 1.0 / self.fps if self.fps > 0 else 1.0/30
        
        while self.playing and self.cap:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # End of video
                self.playing = False
                break
                
            self.current_frame += 1
            
            # Resize frame immediately in background thread
            height, width = frame.shape[:2]
            
            # Use fixed size for better performance
            display_width = 640
            display_height = 480
            
            # Calculate aspect ratio preserving resize
            aspect_ratio = width / height
            if display_width / display_height > aspect_ratio:
                new_height = display_height
                new_width = int(display_height * aspect_ratio)
            else:
                new_width = display_width
                new_height = int(display_width / aspect_ratio)
                
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Detect/track athletes and draw bounding boxes
            detections = []
            if self.detection_enabled:
                self.frames_since_detection += 1
                
                # Use tracking if active and successful
                if self.tracking_active and self.tracker:
                    success, bbox = self.update_tracker(resized_frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        detections = [(x, y, x+w, y+h)]
                        print(f"person tracked, frame [{self.current_frame}]")
                        
                        # Re-detect periodically to refresh tracker
                        if self.frames_since_detection >= self.redetect_interval:
                            new_detections = self.detect_athletes_multi_method(resized_frame)
                            if new_detections:
                                # Reinitialize tracker with new detection
                                largest_detection = max(new_detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
                                x1, y1, x2, y2 = largest_detection
                                self.init_tracker(resized_frame, (x1, y1, x2-x1, y2-y1))
                                detections = [largest_detection]
                                print(f"tracker refreshed, frame [{self.current_frame}]")
                            else:
                                self.frames_since_detection = 0  # Reset counter
                    else:
                        # Tracking failed, fall back to detection
                        self.tracking_active = False
                        detections = self.detect_athletes_multi_method(resized_frame)
                        if detections:
                            # Initialize tracker with first detection
                            largest_detection = max(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
                            x1, y1, x2, y2 = largest_detection
                            self.init_tracker(resized_frame, (x1, y1, x2-x1, y2-y1))
                            print(f"tracking failed, re-detected, frame [{self.current_frame}]")
                else:
                    # No active tracking, use detection
                    detections = self.detect_athletes_multi_method(resized_frame)
                    if detections:
                        # Initialize tracker with first detection
                        largest_detection = max(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
                        x1, y1, x2, y2 = largest_detection
                        self.init_tracker(resized_frame, (x1, y1, x2-x1, y2-y1))
                        print(f"athlete detected and tracking started, frame [{self.current_frame}]")
                
                resized_frame = self.draw_bounding_boxes(resized_frame, detections)
            
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Update UI in main thread with processed frame
            self.root.after_idle(lambda f=frame_rgb: self.update_display(f))
            
            # Update progress less frequently for better performance
            if self.current_frame % 5 == 0:  # Update every 5 frames
                self.root.after_idle(self.update_progress)
                self.root.after_idle(self.update_time_label)
            
            # More accurate timing
            processing_time = time.time() - frame_start
            sleep_time = max(0, frame_time - processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    def display_frame(self, frame=None):
        if not self.cap:
            return
            
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return
                
        # Resize frame to fit display
        height, width = frame.shape[:2]
        display_width = self.video_frame.winfo_width() - 4
        display_height = self.video_frame.winfo_height() - 4
        
        # Ensure minimum display size
        if display_width < 100:
            display_width = 400
        if display_height < 100:
            display_height = 300
            
        # Calculate aspect ratio preserving resize
        aspect_ratio = width / height
        if display_width / display_height > aspect_ratio:
            new_height = display_height
            new_width = int(display_height * aspect_ratio)
        else:
            new_width = display_width
            new_height = int(display_width / aspect_ratio)
            
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Detect athletes and draw bounding boxes
        detections = []
        if self.detection_enabled:
            detections = self.detect_athletes_multi_method(frame)
            frame = self.draw_bounding_boxes(frame, detections)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        
        # Update label
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo  # Keep a reference
        
    def update_display(self, frame_rgb):
        """Update display with pre-processed RGB frame"""
        try:
            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
        except Exception as e:
            print(f"Display update error: {e}")
        
    def update_progress(self):
        if self.cap and self.total_frames > 0:
            self.progress_var.set(self.current_frame)
            
    def update_time_label(self):
        if self.cap and self.fps > 0:
            current_ms = int((self.current_frame / self.fps) * 1000)
            total_ms = int((self.total_frames / self.fps) * 1000)
            
            self.time_label.config(text=f"{current_ms}ms / {total_ms}ms")
    
    def update_video_info(self):
        if self.cap and self.fps > 0:
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.frames_label.config(text=f"Frames: {self.total_frames}")
            
    def seek_video(self, value):
        if not self.cap or self.playing:
            return
            
        frame_number = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        self.display_frame()
        self.update_time_label()
        
    def on_closing(self):
        self.playing = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()