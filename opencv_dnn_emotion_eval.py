"""
OpenCV Emotion Recognition Accuracy Test Script

This script tests emotion recognition accuracy on the CMED dataset using OpenCV DNN.

Features:
- Face detection using OpenCV DNN (or Haar Cascade fallback)
- Emotion recognition using ONNX model (emotion-ferplus-8.onnx)
- Per-class accuracy calculation
- Confusion matrix generation
- CSV export for further analysis

Optional: For better face detection, download the DNN face detector:
1. deploy.prototxt: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
2. res10_300x300_ssd_iter_140000.caffemodel: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
   Place both files in the same directory as this script.

If not available, the script will automatically use Haar Cascade (built into OpenCV).
"""

import os
import glob
import cv2
import numpy as np
import csv
from tqdm import tqdm
from collections import Counter
import time
from sklearn.metrics import confusion_matrix
import pandas as pd

# --- 1. Configs ---
# Dataset path (adjust if your images are in a different location)
# Try proc/appearance/ if images are there, or data/CMED_Dataset/ if that's your structure
DATASET_PATH = r"C:\Users\tangy\Desktop\CMED\proc\appearance"  # Change this to your image dataset path
# Alternative: DATASET_PATH = r"C:\Users\tangy\Desktop\CMED\data\CMED_Dataset"

# ONNX model path
MODEL_PATH = r"C:\Users\tangy\Desktop\CMED\emotion-ferplus-8.onnx"

# Face detection model (OpenCV DNN - more robust than Haar Cascade)
FACE_DETECTION_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_DETECTION_MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
FACE_DETECTION_PROTO_LOCAL = "deploy.prototxt"
FACE_DETECTION_MODEL_LOCAL = "res10_300x300_ssd_iter_140000.caffemodel"

# CMED labels (folder names)
CMED_LABELS = ['anger', 'disgust', 'fear', 'happy', 'no emotion', 'sad', 'surprise']
# ONNX model labels (must match model output order)
ONNX_LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# Output file
RESULTS_CSV = "experiment_results.csv" 

# --- 2. Face Detection Function ---
def detect_face_dnn(image, face_net, confidence_threshold=0.5):
    """
    Detect face using OpenCV DNN face detector.
    Returns: (x, y, w, h) bounding box or None if no face found.
    """
    try:
        h, w = image.shape[:2]
        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), [104, 117, 123])
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Find the best face detection
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold and confidence > best_confidence:
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                best_confidence = confidence
                best_box = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)
        
        return best_box
    except Exception as e:
        return None

def detect_face_haar(image, face_cascade):
    """
    Fallback: Detect face using Haar Cascade (if DNN fails).
    Returns: (x, y, w, h) bounding box or None if no face found.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            # Return the largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            return tuple(faces[0])  # (x, y, w, h)
        return None
    except Exception as e:
        return None

# --- 3. Preprocessing function ---
def preprocess_for_onnx(face_roi):
    """
    Pre-processing for ferplus-8 model.
    Needs: 1x64x64 grayscale
    """
    try:
        # 1. Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            img_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = face_roi
        # 2. Resize to 64x64
        img_resized = cv2.resize(img_gray, (64, 64))
        # 3. Convert to float32
        img_float = img_resized.astype(np.float32)
        # 4. Add batch & channel dims (N, C, H, W) -> [1, 1, 64, 64]
        blob = np.expand_dims(np.expand_dims(img_float, axis=0), axis=0)
        return blob
    except Exception as e:
        return None

# --- 4. Label Mapping Function ---
def map_onnx_to_cmed(onnx_label):
    """
    Map ONNX model output labels to CMED labels.
    """
    mapping = {
        'neutral': 'no emotion',
        'happiness': 'happy',
        'sadness': 'sad',
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'surprise': 'surprise',
        'contempt': 'no emotion'  # Map contempt to no emotion
    }
    return mapping.get(onnx_label, 'no emotion')

# --- 5. Main test function ---
def run_opencv_test():
    print("--- Starting OpenCV (DNN) Image Accuracy Test ---")
    print("=" * 60)
    start_time = time.time()
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found! {MODEL_PATH}")
        print("\nTo download a compatible emotion recognition model:")
        print("1. Visit: https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus")
        print("2. Download emotion-ferplus-8.onnx")
        print("3. Place it in the same directory as this script")
        return
    
    # Load the emotion recognition DNN model
    try:
        emotion_net = cv2.dnn.readNet(MODEL_PATH)
        print(f" Emotion model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"FATAL ERROR: Could not load model {MODEL_PATH}. Error: {e}")
        return
    
    # Load face detection model (try DNN first, fallback to Haar Cascade)
    face_net = None
    face_cascade = None
    
    # Try to load DNN face detector
    if os.path.exists(FACE_DETECTION_PROTO_LOCAL) and os.path.exists(FACE_DETECTION_MODEL_LOCAL):
        try:
            face_net = cv2.dnn.readNetFromCaffe(FACE_DETECTION_PROTO_LOCAL, FACE_DETECTION_MODEL_LOCAL)
            print(f" DNN Face detector loaded")
        except:
            pass
    
    # Fallback to Haar Cascade (built into OpenCV)
    if face_net is None:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                raise Exception("Haar Cascade not loaded")
            print(f"✓ Haar Cascade face detector loaded (fallback)")
        except Exception as e:
            print(f"WARNING: Could not load face detector. Error: {e}")
            print("Continuing without face detection (will use full image)...")
    
    print()
    
    # Storage for results
    all_results = []  # For CSV: True_Label, Predicted_Label, Is_Correct, Confidence_Score
    class_counts = {label: 0 for label in CMED_LABELS}  # Count per class
    class_correct = {label: 0 for label in CMED_LABELS}  # Correct predictions per class
    
    total_images = 0
    images_without_face = 0
    
    # Loop over each emotion folder
    for true_label in tqdm(CMED_LABELS, desc="Overall Progress"):
        folder_path = os.path.join(DATASET_PATH, true_label)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
        
        # Find all images (recursive)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
        
        if not image_paths:
            print(f"Warning: No images found in {folder_path}")
            continue
        
        # Process each image
        for image_path in tqdm(image_paths, desc=f"Processing {true_label}", leave=False):
            try:
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                total_images += 1
                class_counts[true_label] += 1
                
                # Detect face
                face_box = None
                if face_net is not None:
                    face_box = detect_face_dnn(image, face_net)
                elif face_cascade is not None:
                    face_box = detect_face_haar(image, face_cascade)
                
                # Extract face ROI or use full image if no face detected
                if face_box is not None:
                    x, y, w, h = face_box
                    # Add some padding
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    face_roi = image[y:y+h, x:x+w]
                else:
                    images_without_face += 1
                    # Use full image if no face detected
                    face_roi = image
                
                if face_roi.size == 0:
                    continue
                
                # Preprocess for emotion model
                blob = preprocess_for_onnx(face_roi)
                if blob is None:
                    continue
                
                # Emotion inference
                try:
                    emotion_net.setInput(blob)
                    predictions = emotion_net.forward()
                    
                    # Get predicted emotion and confidence
                    predicted_index = np.argmax(predictions)
                    confidence_score = float(predictions[0][predicted_index])
                    predicted_emotion_onnx = ONNX_LABELS[predicted_index]
                    
                    # Map to CMED label
                    predicted_emotion = map_onnx_to_cmed(predicted_emotion_onnx)
                    
                    # Check if correct
                    is_correct = 1 if predicted_emotion.lower() == true_label.lower() else 0
                    
                    if is_correct:
                        class_correct[true_label] += 1
                    
                    # Store result for CSV
                    all_results.append({
                        'True_Label': true_label,
                        'Predicted_Label': predicted_emotion,
                        'Is_Correct': is_correct,
                        'Confidence_Score': confidence_score
                    })
                    
                except Exception as e:
                    # Inference failed
                    all_results.append({
                        'True_Label': true_label,
                        'Predicted_Label': 'unknown',
                        'Is_Correct': 0,
                        'Confidence_Score': 0.0
                    })
                    
            except Exception as e:
                # Image processing failed
                continue
    
    if total_images == 0:
        print("ERROR: No processable image files were found in the dataset.")
        print(f"Checked path: {DATASET_PATH}")
        return
    
    # Calculate metrics
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for label in CMED_LABELS:
        if class_counts[label] > 0:
            class_accuracy[label] = (class_correct[label] / class_counts[label]) * 100
        else:
            class_accuracy[label] = 0.0
    
    # Overall accuracy
    total_correct = sum(class_correct.values())
    overall_accuracy = (total_correct / total_images) * 100
    
    # Build confusion matrix
    true_labels_list = [r['True_Label'] for r in all_results]
    pred_labels_list = [r['Predicted_Label'] for r in all_results]
    cm = confusion_matrix(true_labels_list, pred_labels_list, labels=CMED_LABELS)
    
    # --- 6. Print Results ---
    print("\n" + "=" * 60)
    print("--- OpenCV (DNN) Emotion Recognition Test Results ---")
    print("=" * 60)
    print(f"Test Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Total Images Processed: {total_images}")
    print(f"Images without detected face: {images_without_face}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print()
    
    # Per-class accuracy
    print("--- Per-Class Accuracy ---")
    for label in CMED_LABELS:
        count = class_counts[label]
        correct = class_correct[label]
        acc = class_accuracy[label]
        print(f"{label.capitalize():15s}: {acc:6.2f}% ({correct}/{count})")
    print()
    
    # Confusion Matrix
    print("--- Confusion Matrix ---")
    print("Rows = True Label, Columns = Predicted Label")
    print()
    # Header
    header = "True\\Pred".ljust(15)
    for label in CMED_LABELS:
        header += f"{label[:8]:>8}"
    print(header)
    print("-" * (15 + 8 * len(CMED_LABELS)))
    
    # Matrix rows
    for i, true_label in enumerate(CMED_LABELS):
        row = f"{true_label[:14]:14s}"
        for j, pred_label in enumerate(CMED_LABELS):
            row += f"{cm[i, j]:8d}"
        print(row)
    print()
    
    # --- 7. Save Results to CSV ---
    try:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_CSV, index=False, encoding='utf-8')
        print(f" Results saved to: {RESULTS_CSV}")
        print(f"  Total rows: {len(all_results)}")
    except Exception as e:
        print(f"ERROR: Could not write CSV to {RESULTS_CSV}. Error: {e}")
        # Fallback: try with csv module
        try:
            with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['True_Label', 'Predicted_Label', 'Is_Correct', 'Confidence_Score'])
                writer.writeheader()
                writer.writerows(all_results)
            print(f" Results saved to: {RESULTS_CSV} (using csv module)")
        except Exception as e2:
            print(f"ERROR: CSV write failed completely. Error: {e2}")
    
    print("=" * 60)

if __name__ == "__main__":
    run_opencv_test()
