import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import threading
import time
from PIL import Image, ImageTk
import os
import numpy as np

# Import DNNDetector from detectMultiScale_demo
from detectMultiScale_demo import DNNDetector

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
        
        # Initialize person detection
        self.init_detection()
        
        self.setup_ui()
    
    def init_detection(self):
        """Initialize person detection using DNN detector"""
        try:
            self.detector = DNNDetector()
            print(f"Initialized: {self.detector.get_name()}")
        except Exception as e:
            print(f"Warning: Could not initialize person detection: {e}")
            self.detector = None
    
    def detect_persons(self, frame):
        """Detect persons in frame and return bounding boxes"""
        if not self.detector:
            return []
        
        try:
            # Detect people in the frame using DNN detector
            boxes = self.detector.detect(frame)
            
            # Convert (x, y, w, h) format to (x1, y1, x2, y2) format
            persons = []
            for (x, y, w, h) in boxes:
                persons.append((x, y, x+w, y+h))
            
            return persons
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_bounding_boxes(self, frame, detections):
        """Draw circles at the center of detected persons"""
        for (x1, y1, x2, y2) in detections:
            # Calculate center of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw solid red circle (radius = 10 pixels)
            cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
        
        return frame
    
    def toggle_detection(self):
        """Toggle athlete detection on/off"""
        self.detection_enabled = self.detection_var.get()
        
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
        
        # Export button
        self.export_btn = ttk.Button(detection_frame, text="Export Video", 
                                    command=self.export_video)
        self.export_btn.pack(side=tk.RIGHT)
        
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
            
            # Detect athletes and draw bounding boxes
            detections = self.detect_persons(resized_frame)
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
        detections = self.detect_persons(frame)
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
        
    def export_video(self):
        """Export video with tracking circles"""
        if not self.cap or not self.video_file:
            messagebox.showerror("Error", "No video loaded")
            return
        
        # Get output filename
        output_file = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        try:
            # Create new VideoCapture for export (don't interfere with playback)
            export_cap = cv2.VideoCapture(self.video_file)
            
            # Get video properties
            fps = export_cap.get(cv2.CAP_PROP_FPS)
            width = int(export_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(export_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(export_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            if not out.isOpened():
                messagebox.showerror("Error", "Could not create output video file")
                export_cap.release()
                return
            
            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Exporting Video")
            progress_window.geometry("400x100")
            progress_window.resizable(False, False)
            
            progress_label = ttk.Label(progress_window, text="Exporting video with tracking circles...")
            progress_label.pack(pady=10)
            
            export_progress = ttk.Progressbar(progress_window, length=300, mode='determinate')
            export_progress.pack(pady=10)
            export_progress['maximum'] = total_frames
            
            self.root.update()
            
            frame_count = 0
            while True:
                ret, frame = export_cap.read()
                if not ret:
                    break
                
                # Detect persons and draw circles if detection is enabled
                if self.detection_enabled:
                    detections = self.detect_persons(frame)
                    frame = self.draw_bounding_boxes(frame, detections)
                
                # Write frame
                out.write(frame)
                frame_count += 1
                
                # Update progress every 10 frames
                if frame_count % 10 == 0:
                    export_progress['value'] = frame_count
                    progress_window.update()
            
            # Cleanup
            export_cap.release()
            out.release()
            progress_window.destroy()
            
            messagebox.showinfo("Success", f"Video exported successfully to:\n{output_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
            if 'export_cap' in locals():
                export_cap.release()
            if 'out' in locals():
                out.release()
            if 'progress_window' in locals():
                progress_window.destroy()

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