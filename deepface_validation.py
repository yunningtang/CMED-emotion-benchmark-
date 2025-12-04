import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
from deepface import DeepFace


# Configuration

# MODEL SELECTION (Choose one)
# Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace"
MODEL_NAME = "VGG-Face"  # 

# Paths
APEX_IMAGES_ROOT = r"C:\Users\tangy\Desktop\CMED\data\Apex_Images"
CSV_PATH = r"C:\Users\tangy\Desktop\CMED\video_emotion_metadata_cleaned.csv"
OUTPUT_DIR = "deepface_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate output filenames
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, f"deepface_{MODEL_NAME.lower().replace('-', '_')}_results_{timestamp}.xlsx")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"deepface_{MODEL_NAME.lower().replace('-', '_')}_results_{timestamp}.json")

# DeepFace emotion labels (7 classes, matches CMED!)
DEEPFACE_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# CMED emotion labels
CMED_LABELS = ['anger', 'disgust', 'fear', 'happy', 'no emotion', 'sad', 'surprise']

# Label mapping: DeepFace → CMED
DEEPFACE_TO_CMED = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprise',
    'neutral': 'neutral'  # CMED的"no emotion"
}

# CMED → DeepFace (for ground truth)
CMED_TO_DEEPFACE = {
    'anger': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'no emotion': 'neutral',
    'sad': 'sad',
    'surprise': 'surprise'
}


# DeepFace Model Wrapper

class DeepFaceModel:
    """
    DeepFace model wrapper with consistent interface
    """
    
    def __init__(self, model_name='VGG-Face'):
        """
        Initialize DeepFace model
        
        Args:
            model_name: Backbone model name
        """
        self.model_name = model_name
        print(f"  DeepFace initialized with {model_name} backbone")
        print(f"  Note: Model will be downloaded on first use if needed")
    
    def predict(self, image_path):
        """
        Predict emotion for a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            dict with prediction results, or None if failed
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                img_path=img,
                actions=['emotion'],
                enforce_detection=False,  # Skip face detection (already cropped)
                detector_backend='skip',   # Explicitly skip detection
                silent=True                # Suppress output
            )
            
            # Handle result format (can be list or dict)
            if isinstance(result, list):
                result = result[0]  # Take first face
            
            # Extract emotion predictions
            emotion_scores = result['emotion']
            
            # Get predicted emotion
            predicted_label = result['dominant_emotion']
            
            # Convert to probabilities (DeepFace outputs are already probabilities/percentages)
            # Need to check if they sum to 100 or 1
            prob_sum = sum(emotion_scores.values())
            
            if prob_sum > 10:  # If sum ~100, it's percentage
                probabilities = {k: v/100.0 for k, v in emotion_scores.items()}
            else:  # Already probabilities
                probabilities = emotion_scores.copy()
            
            # Get confidence (probability of predicted class)
            confidence = float(probabilities[predicted_label])
            
            # Calculate confidence gap
            sorted_probs = sorted(probabilities.values(), reverse=True)
            confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            
            # Verify probability format
            prob_sum_normalized = sum(probabilities.values())
            if abs(prob_sum_normalized - 1.0) > 0.01:
                print(f" Warning: Probability sum = {prob_sum_normalized:.4f} for {image_path.name}")
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'confidence_gap': confidence_gap,
                'probabilities': probabilities,
                'raw_scores': emotion_scores  # Original scores
            }
            
        except Exception as e:
            print(f" Prediction failed for {image_path}: {str(e)}")
            return None



# Dataset Testing

def find_image_path(emotion, subject, filename, apex_frame, images_root):
    """
    Find the actual image file path
    """
    video_base = filename.replace('.mp4', '')
    image_filename = f"{subject}_{video_base}_f{int(apex_frame)}.jpg"
    
    emotion_variants = [
        emotion,
        emotion.replace(' ', '_'),
        emotion.replace(' ', '_').title(),
        emotion.title()
    ]
    
    for emo_var in emotion_variants:
        image_path = Path(images_root) / emo_var / image_filename
        if image_path.exists():
            return image_path
    
    return None


def test_dataset(model, csv_path, images_root, output_excel, output_json):
    """
    Test model on entire CMED dataset
    """
    print("\n" + "=" * 80)
    print(f"Testing DeepFace ({model.model_name}) on CMED Dataset")
    print("=" * 80)
    
    # Load metadata
    print(f"\nLoading metadata: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f" Loaded {len(df)} samples")
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Statistics
    stats = {
        'total': len(df),
        'processed': 0,
        'success': 0,
        'failed': 0,
        'correct': 0,
        'errors': [],
        'model_name': model.model_name
    }
    
    results = []
    
    # Process each sample
    print("\nProcessing samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
        try:
            # Get metadata
            subject = str(row['subject'])
            emotion_cmed = str(row['emotion'])
            filename = str(row['filename'])
            apex_frame = row['apex_frame']
            
            # Find image path
            image_path = find_image_path(
                emotion_cmed, subject, filename, apex_frame, images_root
            )
            
            if image_path is None:
                stats['failed'] += 1
                stats['errors'].append({
                    'row': idx,
                    'reason': 'Image file not found',
                    'emotion': emotion_cmed,
                    'subject': subject,
                    'filename': filename
                })
                continue
            
            # Predict
            prediction = model.predict(image_path)
            
            if prediction is None:
                stats['failed'] += 1
                stats['errors'].append({
                    'row': idx,
                    'reason': 'Prediction failed',
                    'emotion': emotion_cmed,
                    'image': image_path.name
                })
                continue
            
            # Get predicted label (DeepFace format)
            predicted_deepface = prediction['predicted_label']
            
            # Map to CMED format
            predicted_cmed = DEEPFACE_TO_CMED[predicted_deepface]
            
            # Normalize true emotion
            true_emotion_normalized = emotion_cmed.lower().replace(' ', '_')
            if true_emotion_normalized == 'no_emotion':
                true_emotion_normalized = 'neutral'
            elif true_emotion_normalized == 'anger':
                true_emotion_normalized = 'angry'
            
            # Check correctness
            is_correct = (predicted_cmed == true_emotion_normalized)
            
            if is_correct:
                stats['correct'] += 1
            
            # Store result (keep the same format as OpenCV)
            result = {
                'file_name': image_path.name,
                'parent_folder': subject,
                'folder_label': emotion_cmed,
                'true_emotion': true_emotion_normalized,
                'predicted_emotion': predicted_cmed,
                'predicted_deepface_label': predicted_deepface,  # DeepFace original label
                'is_correct': is_correct,
                'confidence': prediction['confidence'],
                'confidence_gap': prediction['confidence_gap'],
                'probabilities': prediction['probabilities'],
                'raw_scores': prediction['raw_scores']
            }
            
            results.append(result)
            stats['success'] += 1
            stats['processed'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append({
                'row': idx,
                'reason': str(e),
                'emotion': emotion_cmed if 'emotion_cmed' in locals() else 'Unknown'
            })
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save Excel
    print(f"\nSaving to Excel: {output_excel}")
    df_results.to_excel(output_excel, index=False)
    print(f" Excel saved: {len(df_results)} rows")
    
    # Save JSON
    print(f"\nSaving to JSON: {output_json}")
    results_json = {
        'metadata': {
            'model': f'DeepFace ({model.model_name})',
            'dataset': 'CMED',
            'total_samples': stats['total'],
            'processed': stats['processed'],
            'timestamp': datetime.now().isoformat()
        },
        'statistics': stats,
        'results': results
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f" JSON saved")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Testing Summary")
    print("=" * 80)
    print(f"\nModel: DeepFace ({model.model_name})")
    print(f"\nTotal samples: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nCorrect predictions: {stats['correct']}")
    
    if stats['success'] > 0:
        accuracy = stats['correct'] / stats['success']
        print(f"Accuracy: {accuracy:.4f} ({stats['correct']}/{stats['success']})")
    
    if stats['errors']:
        print(f"\nErrors: {len(stats['errors'])}")
        print("First 5 errors:")
        for error in stats['errors'][:5]:
            print(f"  - Row {error['row']}: {error['reason']}")
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Excel: {output_excel}")
    print(f"  JSON: {output_json}")
    
    return df_results, stats

# Validation

def validate_setup():
    """
    Validate setup before running full test
    """
    print("=" * 80)
    print("Validating Setup")
    print("=" * 80)
    
    # Check CSV
    print("\n1. Checking metadata CSV...")
    if not os.path.exists(CSV_PATH):
        print(f" CSV file not found: {CSV_PATH}")
        return False
    print(f" CSV file exists")
    
    # Check images directory
    print("\n2. Checking images directory...")
    if not os.path.exists(APEX_IMAGES_ROOT):
        print(f" Images directory not found: {APEX_IMAGES_ROOT}")
        return False
    print(f" Images directory exists")
    
    # Test DeepFace
    print("\n3. Testing DeepFace...")
    try:
        print(f"  Initializing {MODEL_NAME}...")
        model = DeepFaceModel(MODEL_NAME)
        print(f" DeepFace initialized")
    except Exception as e:
        print(f" Failed to initialize DeepFace: {e}")
        return False
    
    # Test on sample
    print("\n4. Testing on sample image...")
    try:
        apex_root = Path(APEX_IMAGES_ROOT)
        sample_image = None
        
        for emotion in ['happy', 'sad', 'anger']:
            emotion_dir = apex_root / emotion
            if emotion_dir.exists():
                images = list(emotion_dir.glob('*.jpg'))
                if images:
                    sample_image = images[0]
                    break
        
        if sample_image is None:
            print(" No sample image found")
            return True
        
        print(f"  Testing with: {sample_image.name}")
        
        # Predict
        prediction = model.predict(sample_image)
        
        if prediction is None:
            print(f" Prediction failed")
            return False
        
        print(f"   Prediction successful")
        print(f"    Predicted: {prediction['predicted_label']}")
        print(f"    Confidence: {prediction['confidence']:.4f}")
        
        # Verify probability format
        prob_sum = sum(prediction['probabilities'].values())
        print(f"    Probability sum: {prob_sum:.6f}")
        
        if abs(prob_sum - 1.0) > 0.01:
            print(f"  Probability sum != 1.0")
            return False
        
        print(f"  Probability format correct (sum=1.0)")
        
        # Check probability range
        all_probs = list(prediction['probabilities'].values())
        if min(all_probs) < 0 or max(all_probs) > 1:
            print(f" Probabilities out of range [0, 1]")
            return False
        
        print(f" ll probabilities in [0, 1]")
        
        # Print top 3
        sorted_emotions = sorted(
            prediction['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        print(f"  Top 3 predictions:")
        for emotion, prob in sorted_emotions[:3]:
            print(f"    {emotion:10s}: {prob:.4f} ({prob*100:.2f}%)")
        
    except Exception as e:
        print(f" Test prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print(" All validation checks passed!")
    print("=" * 80)
    
    return True


# Model Comparison# 

def compare_models(csv_path, images_root):
    """
    Test multiplpe Deepface backbone models
    """
    print("\n" + "=" * 80)
    print("COMPARING DEEPFACE BACKBONE MODELS")
    print("=" * 80)
    
    #  Models to test
    models_to_test = ['VGG-Face', 'Facenet', 'Facenet512']
    
    comparison_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Create model
            model = DeepFaceModel(model_name)
            
            # Generate output paths
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_excel = os.path.join(OUTPUT_DIR, f"deepface_{model_name.lower().replace('-', '_')}_results_{timestamp}.xlsx")
            output_json = os.path.join(OUTPUT_DIR, f"deepface_{model_name.lower().replace('-', '_')}_results_{timestamp}.json")
            
            # Test
            df_results, stats = test_dataset(model, csv_path, images_root, output_excel, output_json)
            
            # Store comparison data
            accuracy = stats['correct'] / stats['success'] if stats['success'] > 0 else 0
            comparison_results[model_name] = {
                'total': stats['total'],
                'processed': stats['processed'],
                'success': stats['success'],
                'correct': stats['correct'],
                'accuracy': accuracy,
                'failed': stats['failed']
            }
            
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            comparison_results[model_name] = {
                'error': str(e)
            }
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(comparison_results).T
    df_comparison.index.name = 'Model'
    
    # Save comparison
    comparison_path = os.path.join(OUTPUT_DIR, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    df_comparison.to_excel(comparison_path)
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print("\n" + df_comparison.to_string())
    print(f"\n Comparison saved to: {comparison_path}")
    
    return df_comparison


# Main Function

def main():
    """
    Main function
    """
    print("=" * 80)
    print("DeepFace Emotion Recognition Testing")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    
    # Validate
    if not validate_setup():
        print("\n Validation failed")
        return
    
    # Ask user
    print("\n" + "=" * 80)
    print("Choose Testing Mode:")
    print("=" * 80)
    print("1. Test with current model only")
    print("2. Compare multiple models (VGG-Face, Facenet, Facenet512)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '2':
        # Compare models
        df_comparison = compare_models(CSV_PATH, APEX_IMAGES_ROOT)
    else:
        # Test current model
        model = DeepFaceModel(MODEL_NAME)
        df_results, stats = test_dataset(model, CSV_PATH, APEX_IMAGES_ROOT, 
                                         OUTPUT_EXCEL, OUTPUT_JSON)
    
    print(f"\n{'='*80}")
    print("All Done!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
