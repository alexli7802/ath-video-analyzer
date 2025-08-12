import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from abc import ABC, abstractmethod

class PersonDetector(ABC):
    """Abstract base class for person detection models"""
    
    @abstractmethod
    def detect(self, frame):
        """Detect persons in frame and return bounding boxes"""
        pass
    
    @abstractmethod
    def get_name(self):
        """Return detector name"""
        pass

class HOGDetector(PersonDetector):
    """HOG-based person detector"""
    
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect(self, frame):
        boxes, _ = self.hog.detectMultiScale(frame, 
                                           winStride=(8, 8),
                                           padding=(8, 8),
                                           scale=1.05)
        return boxes
    
    def get_name(self):
        return "HOG Detector"

class YOLODetector(PersonDetector):
    """YOLO-based person detector using YOLOv4"""
    
    def __init__(self):
        self.net = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        try:
            self.net = cv2.dnn.readNet("models/yolov4.weights", "models/yolov4.cfg")
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            print("YOLO: Loaded YOLOv4 model")
        except Exception as e:
            print(f"YOLO: Failed to load model files: {e}")
            self.net = None
    
    def detect(self, frame):
        if self.net is None:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run forward pass
            outputs = self.net.forward(self.output_layers)
            
            boxes = []
            confidences = []
            
            # Process outputs
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Class ID 0 is 'person' in COCO dataset
                    if class_id == 0 and confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate top-left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
            
            # Apply non-maximum suppression
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
                final_boxes = []
                if len(indices) > 0:
                    for i in indices.flatten():
                        final_boxes.append(tuple(boxes[i]))
                return final_boxes
            
            return []
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def get_name(self):
        if self.net is None:
            return "YOLO Detector (Not Loaded)"
        return "YOLO Detector (YOLOv4)"

class DNNDetector(PersonDetector):
    """OpenCV DNN-based detector using pre-trained COCO models"""
    
    def __init__(self):
        self.net = None
        self.confidence_threshold = 0.5
        
        try:
            self.net = cv2.dnn.readNetFromTensorflow("models/ssd_mobilenet_v2_coco.pb", 
                                                   "models/ssd_mobilenet_v2_coco.pbtxt")
            print("DNN: Loaded SSD MobileNet model")
        except Exception as e:
            print(f"DNN: Failed to load model files: {e}")
            self.net = None
    
    def detect(self, frame):
        if self.net is None:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run forward pass
            detections = self.net.forward()
            
            boxes = []
            
            # Process detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                
                # Class ID 1 is 'person' in COCO dataset
                if class_id == 1 and confidence > self.confidence_threshold:
                    # Scale bounding box coordinates
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    # Convert to (x, y, w, h) format to match HOG output
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append((x1, y1, w, h))
                    print(boxes[-1])
            
            return boxes
            
        except Exception as e:
            print(f"DNN detection error: {e}")
            return []
    
    def get_name(self):
        if self.net is None:
            return "DNN Detector (Not Loaded)"
        return "DNN Detector (SSD MobileNet)"

def demo_person_detection():
    """Demo showing different person detection models"""
    
    # Create GUI for model and image selection
    root = tk.Tk()
    root.title("Person Detection Demo")
    root.geometry("500x400")
    root.configure(bg='#f0f0f0')
    
    # Available detectors
    detectors = {
        "HOG Detector": HOGDetector,
        "YOLO Detector": YOLODetector, 
        "DNN Detector": DNNDetector
    }
    
    selected_detector = tk.StringVar(value="HOG Detector")
    selected_file = tk.StringVar(value="No file selected")
    
    # Title
    title_label = tk.Label(root, text="Person Detection Demo", 
                          font=("Arial", 16, "bold"), bg='#f0f0f0')
    title_label.pack(pady=(20, 10))
    
    # Model selection frame
    model_frame = tk.LabelFrame(root, text="Detection Model", 
                               font=("Arial", 12), bg='#f0f0f0', padx=10, pady=10)
    model_frame.pack(pady=10, padx=20, fill='x')
    
    for name in detectors.keys():
        tk.Radiobutton(model_frame, text=name, variable=selected_detector, 
                      value=name, bg='#f0f0f0', font=("Arial", 10)).pack(anchor=tk.W, pady=2)
    
    # File selection frame
    file_frame = tk.LabelFrame(root, text="Image Selection", 
                              font=("Arial", 12), bg='#f0f0f0', padx=10, pady=10)
    file_frame.pack(pady=10, padx=20, fill='x')
    
    # Selected file display
    file_label = tk.Label(file_frame, textvariable=selected_file, 
                         bg='#f0f0f0', font=("Arial", 9), wraplength=400)
    file_label.pack(pady=5)
    
    def select_file():
        file_path = filedialog.askopenfilename(
            title="Select an image with people",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            # Show just filename if path is long
            if len(file_path) > 50:
                display_name = "..." + file_path[-47:]
            else:
                display_name = file_path
            selected_file.set(display_name)
            return file_path
        return None
    
    def run_detection():
        file_path = None
        
        # Get the actual file path
        if selected_file.get() == "No file selected":
            file_path = select_file()
            if not file_path:
                return
        else:
            # Try to get full path from display name
            display_name = selected_file.get()
            if display_name.startswith("..."):
                # Need to re-select file
                file_path = select_file()
                if not file_path:
                    return
            else:
                file_path = display_name
        
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not load image")
            return
        
        # Initialize selected detector
        detector_class = detectors[selected_detector.get()]
        detector = detector_class()
        
        print(f"Using: {detector.get_name()}")
        print(f"Image: {file_path} ({img.shape})")
        
        # Run detection
        boxes = detector.detect(img)
        print(f"Detected {len(boxes)} people")
        
        # Display results
        if len(boxes) > 0:
            img_display = img.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add confidence text if available
                cv2.putText(img_display, 'Person', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Resize image for display if too large
            height, width = img_display.shape[:2]
            max_display_size = 800
            if width > max_display_size or height > max_display_size:
                if width > height:
                    new_width = max_display_size
                    new_height = int(height * max_display_size / width)
                else:
                    new_height = max_display_size
                    new_width = int(width * max_display_size / height)
                img_display = cv2.resize(img_display, (new_width, new_height))
            
            cv2.imshow(f"Results - {detector.get_name()}", img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showinfo("Results", "No people detected in the image")
    
    # Buttons frame
    button_frame = tk.Frame(root, bg='#f0f0f0')
    button_frame.pack(pady=20)
    
    # File picker button
    select_btn = tk.Button(button_frame, text="Browse Image", command=select_file,
                          font=("Arial", 11), bg='#e0e0e0', padx=15, pady=5)
    select_btn.pack(side=tk.LEFT, padx=10)
    
    # Run detection button
    run_btn = tk.Button(button_frame, text="Run Detection", command=run_detection,
                       font=("Arial", 11, "bold"), bg='#4CAF50', fg='white', padx=15, pady=5)
    run_btn.pack(side=tk.LEFT, padx=10)
    
    # Info label
    info_label = tk.Label(root, text="Select a detection model, choose an image, then run detection",
                         font=("Arial", 9), fg='#666', bg='#f0f0f0')
    info_label.pack(pady=(10, 20))
    
    root.mainloop()

if __name__ == "__main__":
    demo_person_detection()