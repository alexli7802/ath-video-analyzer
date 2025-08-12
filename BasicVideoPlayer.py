import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import threading
import time
from PIL import Image, ImageTk
import os


class BasicVideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Basic Video Player")
        self.root.geometry("900x700")
        
        self.video_file = None
        self.cap = None
        self.playing = False
        self.paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        # Start and end frame markers
        self.start_frame = None
        self.end_frame = None
        
        # Speed calculation
        self.calculated_speed = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for file selection
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Select Video File", command=self.select_video_file).pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Video display frame
        self.video_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = ttk.Label(self.video_frame, text="Select a video file to start", 
                                    font=("Arial", 14), foreground="gray")
        self.video_label.pack(expand=True)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Center the control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()
        
        self.play_button = ttk.Button(button_frame, text="Play", command=self.toggle_play_pause, state="disabled")
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_video, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(button_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Start/End marking buttons
        self.start_button = ttk.Button(button_frame, text="Mark Start", command=self.mark_start_frame, state="disabled")
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.end_button = ttk.Button(button_frame, text="Mark End", command=self.mark_end_frame, state="disabled")
        self.end_button.pack(side=tk.LEFT, padx=5)
        
        # Distance and speed calculation frame
        calc_frame = ttk.Frame(main_frame)
        calc_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Distance input
        distance_frame = ttk.Frame(calc_frame)
        distance_frame.pack(side=tk.LEFT)
        
        ttk.Label(distance_frame, text="Distance (m):").pack(side=tk.LEFT)
        
        # Create StringVar for distance input with validation
        self.distance_var = tk.StringVar()
        self.distance_var.trace('w', self.validate_distance_input)
        
        self.distance_entry = ttk.Entry(distance_frame, textvariable=self.distance_var, width=10)
        self.distance_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Calculate button
        self.calculate_button = ttk.Button(calc_frame, text="Calculate Speed", 
                                         command=self.calculate_speed, state="disabled")
        self.calculate_button.pack(side=tk.LEFT, padx=(20, 0))
        
        # Info panel frame
        info_frame = ttk.LabelFrame(main_frame, text="Video Information", padding=10)
        info_frame.pack(fill=tk.X)
        
        # Create scrollable text widget for metadata
        info_container = ttk.Frame(info_frame)
        info_container.pack(fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(info_container, height=8, width=70, wrap=tk.WORD, 
                                font=("Courier", 9), state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(info_container, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def select_video_file(self):
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.3gp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=file_types
        )
        
        if filename:
            self.video_file = filename
            self.file_label.config(text=os.path.basename(filename), foreground="black")
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
            
            # Enable control buttons
            self.play_button.config(state="normal")
            self.stop_button.config(state="normal")
            self.start_button.config(state="normal")
            self.end_button.config(state="normal")
            
            # Reset markers and calculations
            self.start_frame = None
            self.end_frame = None
            self.calculated_speed = None
            self.calculate_button.config(state="normal")
            
            # Display first frame
            self.display_frame()
            
            # Show initial frame markers info
            self.show_frame_markers_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
    
    def show_video_info(self):
        if not self.cap:
            return
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        
        # Convert fourcc to string
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # File information
        file_size = os.path.getsize(self.video_file)
        file_size_mb = file_size / (1024 * 1024)
        
        # Format duration
        duration_min = int(duration // 60)
        duration_sec = int(duration % 60)
        
        # Create info text
        info_text = f"""File Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File Name:          {os.path.basename(self.video_file)}
File Path:          {self.video_file}
File Size:          {file_size_mb:.2f} MB ({file_size:,} bytes)

Video Properties:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resolution:         {width} × {height} pixels
Frame Rate:         {fps:.2f} FPS
Total Frames:       {frame_count:,}
Duration:           {duration_min:02d}:{duration_sec:02d} ({duration:.2f} seconds)
Video Codec:        {fourcc_str.strip()}
Aspect Ratio:       {width/height:.2f}:1

Technical Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bit Rate:           {(file_size * 8) / duration / 1000:.0f} kbps (estimated)
Color Format:       BGR (OpenCV default)
Pixel Format:       24-bit RGB
"""
        
        # Update info text widget
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state=tk.DISABLED)
    
    def toggle_play_pause(self):
        if not self.cap:
            return
        
        if not self.playing:
            # Start playing
            self.playing = True
            self.paused = False
            self.play_button.config(text="Pause")
            
            # Start playback thread
            self.playback_thread = threading.Thread(target=self.playback_loop)
            self.playback_thread.daemon = True
            self.playback_thread.start()
        else:
            # Toggle pause
            self.paused = not self.paused
            if self.paused:
                self.play_button.config(text="Resume")
            else:
                self.play_button.config(text="Pause")
    
    def stop_video(self):
        self.playing = False
        self.paused = False
        self.play_button.config(text="Play")
        
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.display_frame()
    
    def playback_loop(self):
        frame_time = 1.0 / self.fps if self.fps > 0 else 1.0/30
        
        while self.playing and self.cap:
            if self.paused:
                time.sleep(0.1)
                continue
            
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # End of video
                self.playing = False
                break
            
            self.current_frame += 1
            
            # Update display in main thread
            self.root.after_idle(lambda f=frame: self.update_display(f))
            
            # Frame timing
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
        
        self.update_display(frame)
    
    def update_display(self, frame):
        try:
            # Resize frame to fit display
            height, width = frame.shape[:2]
            display_width = self.video_frame.winfo_width() - 4
            display_height = self.video_frame.winfo_height() - 4
            
            # Ensure minimum display size
            if display_width < 100:
                display_width = 640
            if display_height < 100:
                display_height = 480
            
            # Calculate aspect ratio preserving resize
            aspect_ratio = width / height
            if display_width / display_height > aspect_ratio:
                new_height = display_height
                new_width = int(display_height * aspect_ratio)
            else:
                new_width = display_width
                new_height = int(display_width / aspect_ratio)
            
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Add time and frame overlay
            frame = self.add_time_overlay(frame)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Display update error: {e}")
    
    def add_time_overlay(self, frame):
        """Add frame number and time overlay to the frame"""
        try:
            # Calculate time in mm:ss.SSS format
            if self.fps > 0:
                total_seconds = self.current_frame / self.fps
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60
                time_str = f"{minutes:02d}:{seconds:06.3f}"
            else:
                time_str = "00:00.000"
            
            # Create overlay text
            frame_info = f"Frame: {self.current_frame:06d}\nTime: {time_str}"
            
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)  # White text
            thickness = 2
            line_type = cv2.LINE_AA
            
            # Background properties
            bg_color = (0, 0, 0)  # Black background
            padding = 5
            
            # Split text into lines
            lines = frame_info.split('\n')
            line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 5
            
            # Calculate background rectangle size
            max_width = 0
            for line in lines:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                max_width = max(max_width, text_size[0])
            
            bg_height = len(lines) * line_height + padding * 2
            bg_width = max_width + padding * 2
            
            # Draw background rectangle
            cv2.rectangle(frame, (10, 10), (10 + bg_width, 10 + bg_height), bg_color, -1)
            
            # Draw text lines
            y_offset = 10 + padding + line_height - 5
            for line in lines:
                cv2.putText(frame, line, (10 + padding, y_offset), 
                           font, font_scale, color, thickness, line_type)
                y_offset += line_height
            
            return frame
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame
    
    def mark_start_frame(self):
        """Mark current frame as start frame"""
        self.start_frame = self.current_frame
        self.show_frame_markers_info()
        print(f"Start frame marked: {self.start_frame}")
    
    def mark_end_frame(self):
        """Mark current frame as end frame"""
        self.end_frame = self.current_frame
        self.show_frame_markers_info()
        print(f"End frame marked: {self.end_frame}")
    
    def show_frame_markers_info(self):
        """Display start and end frame information in the info panel"""
        if not self.cap:
            info_text = "No video loaded"
        elif self.calculated_speed is not None:
            # Show only speed analysis after calculation
            info_text = f"""Speed Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distance: {self.calculated_speed['distance']:.2f} m
Time Interval: {self.calculated_speed['time']:.3f}s
Speed: {self.calculated_speed['speed_ms']:.3f} m/s"""
        else:
            # Calculate duration and segment information
            total_duration = self.total_frames / self.fps if self.fps > 0 else 0
            total_min = int(total_duration // 60)
            total_sec = int(total_duration % 60)
            total_ms = int((total_duration % 1) * 1000)
            
            info_text = f"""Frame Markers Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Video: {os.path.basename(self.video_file)}
Total Frames: {self.total_frames:,}
Total Duration: {total_min:02d}:{total_sec:02d}.{total_ms:03d}
Frame Rate: {self.fps:.2f} FPS

Current Frame: {self.current_frame:06d}"""
            
            if self.fps > 0:
                current_time = self.current_frame / self.fps
                current_min = int(current_time // 60)
                current_sec = current_time % 60
                info_text += f"\nCurrent Time: {current_min:02d}:{current_sec:06.3f}"
            
            info_text += "\n\nMarked Segments:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            if self.start_frame is not None:
                start_time = self.start_frame / self.fps if self.fps > 0 else 0
                start_min = int(start_time // 60)
                start_sec = start_time % 60
                info_text += f"\nStart Frame: {self.start_frame:06d}\nStart Time:  {start_min:02d}:{start_sec:06.3f}"
            else:
                info_text += "\nStart Frame: Not marked"
            
            if self.end_frame is not None:
                end_time = self.end_frame / self.fps if self.fps > 0 else 0
                end_min = int(end_time // 60)
                end_sec = end_time % 60
                info_text += f"\nEnd Frame:   {self.end_frame:06d}\nEnd Time:    {end_min:02d}:{end_sec:06.3f}"
            else:
                info_text += "\nEnd Frame:   Not marked"
            
            # Calculate segment duration if both markers are set
            if self.start_frame is not None and self.end_frame is not None:
                segment_frames = abs(self.end_frame - self.start_frame)
                segment_duration = segment_frames / self.fps if self.fps > 0 else 0
                segment_min = int(segment_duration // 60)
                segment_sec = segment_duration % 60
                info_text += f"\n\nSegment Information:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                info_text += f"\nSegment Frames: {segment_frames:,}\nSegment Duration: {segment_min:02d}:{segment_sec:06.3f}"
                
                # Show frame range
                if self.start_frame <= self.end_frame:
                    info_text += f"\nFrame Range: {self.start_frame:06d} → {self.end_frame:06d}"
                else:
                    info_text += f"\nFrame Range: {self.end_frame:06d} → {self.start_frame:06d} (reversed)"
        
        # Update info text widget
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state=tk.DISABLED)
    
    def validate_distance_input(self, *args):
        """Validate distance input to allow only numbers and decimal point"""
        value = self.distance_var.get()
        
        # Allow empty string
        if not value:
            return
        
        # Remove any non-numeric characters except decimal point
        filtered_value = ''
        decimal_count = 0
        
        for char in value:
            if char.isdigit():
                filtered_value += char
            elif char == '.' and decimal_count == 0:
                filtered_value += char
                decimal_count += 1
        
        # Update the variable if it was changed
        if filtered_value != value:
            self.distance_var.set(filtered_value)
    
    def calculate_speed(self):
        """Calculate speed based on distance and time between start/end frames"""
        try:
            # Validate inputs
            if self.start_frame is None or self.end_frame is None:
                messagebox.showerror("Error", "Please mark both start and end frames")
                return
            
            distance_str = self.distance_var.get().strip()
            if not distance_str:
                messagebox.showerror("Error", "Please enter a distance value")
                return
            
            try:
                distance = float(distance_str)
                if distance <= 0:
                    messagebox.showerror("Error", "Distance must be greater than 0")
                    return
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid distance value")
                return
            
            # Calculate time difference
            frame_diff = abs(self.end_frame - self.start_frame)
            if frame_diff == 0:
                messagebox.showerror("Error", "Start and end frames cannot be the same")
                return
            
            time_diff = frame_diff / self.fps if self.fps > 0 else 0
            if time_diff == 0:
                messagebox.showerror("Error", "Invalid frame rate or time difference")
                return
            
            # Calculate speed
            speed_ms = distance / time_diff  # meters per second
            speed_kmh = speed_ms * 3.6       # kilometers per hour
            
            # Store calculation results
            self.calculated_speed = {
                'speed_ms': speed_ms,
                'speed_kmh': speed_kmh,
                'distance': distance,
                'time': time_diff,
                'start_frame': self.start_frame,
                'end_frame': self.end_frame
            }
            
            # Update info display
            self.show_frame_markers_info()
            
            print(f"Speed calculated: {speed_ms:.3f} m/s ({speed_kmh:.2f} km/h)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating speed: {str(e)}")
            print(f"Speed calculation error: {e}")
    
    def on_closing(self):
        self.playing = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = BasicVideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()