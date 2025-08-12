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
    """YOLO-based person detector (placeholder - requires model files)"""
    
    def __init__(self):
        # Placeholder - would load YOLO model here
        self.net = None
        print("YOLO: Model files required (yolo.weights, yolo.cfg)")
    
    def detect(self, frame):
        if self.net is None:
            print("YOLO model not loaded")
            return []
        # YOLO detection logic would go here
        return []
    
    def get_name(self):
        return "YOLO Detector"

class DNNDetector(PersonDetector):
    """OpenCV DNN-based detector using pre-trained COCO models"""
    
    def __init__(self):
        self.net = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Try to load a pre-trained model from OpenCV's repository
        try:
            # Download and load SSD MobileNet from OpenCV model zoo
            config_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
            model_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
            
            print("DNN: Attempting to load pre-trained SSD MobileNet...")
            
            # For demo purposes, try to load from local files if available
            try:
                self.net = cv2.dnn.readNetFromTensorflow("ssd_mobilenet_v2_coco.pb", 
                                                       "ssd_mobilenet_v2_coco.pbtxt")
                print("DNN: Loaded local SSD MobileNet model")
            except:
                print("DNN: Local model files not found")
                print("DNN: Download model files:")
                print(f"Config: {config_url}")
                print(f"Model: {model_url}")
                self.net = None
                
        except Exception as e:
            print(f"DNN: Failed to initialize model: {e}")
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
    root.geometry("400x200")
    
    # Available detectors
    detectors = {
        "HOG Detector": HOGDetector,
        "YOLO Detector": YOLODetector, 
        "DNN Detector": DNNDetector
    }
    
    selected_detector = tk.StringVar(value="HOG Detector")
    
    # Model selection frame
    model_frame = tk.Frame(root)
    model_frame.pack(pady=10)
    
    tk.Label(model_frame, text="Select Detection Model:").pack()
    for name in detectors.keys():
        tk.Radiobutton(model_frame, text=name, variable=selected_detector, 
                      value=name).pack(anchor=tk.W)
    
    # Image selection button
    def select_and_run():
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select an image with people",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if not file_path:
            root.destroy()
            return
        
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not load image")
            root.destroy()
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
            cv2.imshow(f"Results - {detector.get_name()}", img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No detections found")
        
        root.destroy()
    
    tk.Button(root, text="Select Image & Run Detection", 
             command=select_and_run).pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    demo_person_detection()