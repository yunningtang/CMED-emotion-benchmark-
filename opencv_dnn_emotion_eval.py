''' 

Model: Emotion FERPlus
Input: (N, 1, 64, 64) - Grayscale images
Output: (1, 8) - Scores for 8 emotion classes
Post-processing: Softmax to get probabilities [0, 1]

'''

import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

# Configuration

# CONTEMPT HANDLING STRATEGY (Choose one)
# Options: "mask", "disgust", "neutral", "exclude"
CONTEMPT_STRATEGY = "mask"  # Masking Strategy

# Paths
MODEL_PATH = r"C:\Users\tangy\Desktop\CMED\models\emotion-ferplus-8.onnx"
APEX_IMAGES_ROOT = r"C:\Users\tangy\Desktop\CMED\data\Apex_Images"
CSV_PATH = r"C:\Users\tangy\Desktop\CMED\video_emotion_metadata_cleaned.csv"
OUTPUT_DIR = "opencv_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate output filenames with strategy suffix
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, f"opencv_fer_results_{CONTEMPT_STRATEGY}_{timestamp}.xlsx")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"opencv_fer_results_{CONTEMPT_STRATEGY}_{timestamp}.json")
OUTPUT_COMPARISON = os.path.join(OUTPUT_DIR, f"strategy_comparison_{timestamp}.xlsx")

# OpenCV FER+ Official Label Mapping
OPENCV_FER_LABELS = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear',
    7: 'contempt'
}

# CMED emotion labels (no contempt)
CMED_LABELS = ['anger', 'disgust', 'fear', 'happy', 'no emotion', 'sad', 'surprise']

# Contempt mapping strategies
CONTEMPT_MAPPINGS = {
    'disgust': 'disgust',  # Mapping Contempt to disgust
    'neutral': 'neutral',  # Mapping Contempt to neutral（visual similary）
}

# Preprocessing Functions


def preprocess_image(image_path):
    """
    Preprocess image according to OpenCV FER+ requirements
    
    Official requirements:
    - Input shape: (N, 1, 64, 64)
    - Grayscale image
    - Values: 0-255 (no normalization)
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        # Resize to 64x64
        img_resized = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 (keep range 0-255)
        img_float = img_resized.astype(np.float32)
        
        # Add batch and channel dimensions: (1, 1, 64, 64)
        img_preprocessed = np.expand_dims(np.expand_dims(img_float, axis=0), axis=0)
        
        return img_preprocessed
        
    except Exception as e:
        raise Exception(f"Preprocessing failed for {image_path}: {str(e)}")


def softmax(scores):
    """
    Softmax function to convert scores to probabilities
    """
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities


# Contempt Handling Strategies


def handle_contempt_mask(probabilities, raw_scores):
    """
    Strategy 1: Probability Masking 
    Masking the probability of contempt, only choose the highest probability emotion from the remaining 7 emotions. 

    Returns:
        dict with prediction results
    """
    # Create a copy to avoid modifying the original data
    masked_probs = probabilities.copy()
    
    # Set the probability of contempt (index 7) to 0
    masked_probs[7] = 0.0
    
    # Find the highest probability emotion from the remaining 7 emotions
    predicted_index = int(np.argmax(masked_probs))
    predicted_label = OPENCV_FER_LABELS[predicted_index]
    confidence = float(probabilities[predicted_index])  # Use the original probability as confidence
    
    # Create a probability dictionary (keep the original probabilities of all 8 emotions)
    prob_dict = {
        OPENCV_FER_LABELS[i]: float(probabilities[i])
        for i in range(len(OPENCV_FER_LABELS))
    }
    
    # Mark this is a prediction through mask
    is_masked = (np.argmax(probabilities) == 7)  # Whether the originally highest emotion is contempt
    
    return {
        'predicted_label': predicted_label,
        'predicted_index': predicted_index,
        'confidence': confidence,
        'probabilities': prob_dict,
        'raw_scores': raw_scores.tolist(),
        'is_masked': is_masked,  # Whether contempt is masked
        'original_top_class': OPENCV_FER_LABELS[np.argmax(probabilities)],  # The originally highest emotion
        'strategy': 'mask'
    }


def handle_contempt_mapping(probabilities, raw_scores, mapping):
    """
    Strategy 2 & 3: Map contempt to another emotion
    
    Args:
        probabilities: Probabilities after softmax (8,)
        raw_scores: Original scores (8,)
        mapping: 'disgust' or 'neutral'
    
    Returns:
        dict with prediction results
    """
    # Normal argmax prediction
    predicted_index = int(np.argmax(probabilities))
    predicted_label = OPENCV_FER_LABELS[predicted_index]
    confidence = float(probabilities[predicted_index])
    
    # If the prediction is contempt, map to the specified category
    if predicted_label == 'contempt':
        predicted_label = CONTEMPT_MAPPINGS[mapping]
    
    prob_dict = {
        OPENCV_FER_LABELS[i]: float(probabilities[i])
        for i in range(len(OPENCV_FER_LABELS))
    }
    
    return {
        'predicted_label': predicted_label,
        'predicted_index': predicted_index,
        'confidence': confidence,
        'probabilities': prob_dict,
        'raw_scores': raw_scores.tolist(),
        'is_mapped': (OPENCV_FER_LABELS[predicted_index] == 'contempt'),
        'strategy': mapping
    }


def handle_contempt_exclude(probabilities, raw_scores):
    """
    Strategy 4: Exclude contempt predictions
    
    Returns None if predicted as contempt
    """
    predicted_index = int(np.argmax(probabilities))
    predicted_label = OPENCV_FER_LABELS[predicted_index]
    
    # If the prediction is contempt, return None to exclude
    if predicted_label == 'contempt':
        return None
    
    confidence = float(probabilities[predicted_index])
    
    prob_dict = {
        OPENCV_FER_LABELS[i]: float(probabilities[i])
        for i in range(len(OPENCV_FER_LABELS))
    }
    
    return {
        'predicted_label': predicted_label,
        'predicted_index': predicted_index,
        'confidence': confidence,
        'probabilities': prob_dict,
        'raw_scores': raw_scores.tolist(),
        'strategy': 'exclude'
    }



# Model Class

class OpenCVFERModel:
    """
    OpenCV FER+ model wrapper with configurable contempt handling
    """
    
    def __init__(self, model_path, contempt_strategy='mask'):
        """
        Initialize model
        
        Args:
            model_path: Path to ONNX model file
            contempt_strategy: How to handle contempt predictions
                - 'mask': Probability masking 
                - 'disgust': Map to disgust
                - 'neutral': Map to neutral
                - 'exclude': Exclude from evaluation
        """
        self.model_path = model_path
        self.contempt_strategy = contempt_strategy
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f" Model loaded: {self.model_path}")
            print(f" Contempt strategy: {self.contempt_strategy}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def predict(self, image_path):
        """
        Predict emotion for a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            dict with prediction results, or None if excluded
        """
        # Preprocess image
        img_preprocessed = preprocess_image(image_path)
        
        # Run inference - get raw scores
        raw_scores = self.session.run(
            [self.output_name],
            {self.input_name: img_preprocessed}
        )[0][0]  # Shape: (8,)
        
        # Apply softmax to get probabilities
        probabilities = softmax(raw_scores)
        
        # Handle contempt based on strategy
        if self.contempt_strategy == 'mask':
            return handle_contempt_mask(probabilities, raw_scores)
        
        elif self.contempt_strategy in ['disgust', 'neutral']:
            return handle_contempt_mapping(probabilities, raw_scores, self.contempt_strategy)
        
        elif self.contempt_strategy == 'exclude':
            return handle_contempt_exclude(probabilities, raw_scores)
        
        else:
            raise ValueError(f"Unknown strategy: {self.contempt_strategy}")



# Label Mapping for Evaluation

def get_cmed_label(opencv_label):
    """
    Map OpenCV label to CMED label for evaluation
    """
    mapping = {
        'neutral': 'neutral',
        'happiness': 'happy',
        'surprise': 'surprise',
        'sadness': 'sad',
        'anger': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'contempt': None  # Should not appear if using mask strategy
    }
    return mapping.get(opencv_label)


# Dataset Testing


def find_image_path(emotion, subject, filename, apex_frame, images_root):
    """Find the actual image file path"""
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
    print(f"Testing OpenCV FER+ with Strategy: {model.contempt_strategy.upper()}")
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
        'excluded': 0,  # The number of contempt excluded
        'masked': 0,    # The number of contempt masked
        'correct': 0,
        'errors': [],
        'contempt_cases': []  # Record contempt related cases
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
            
            # Check if excluded (strategy='exclude' and predicted contempt)
            if prediction is None:
                stats['excluded'] += 1
                stats['contempt_cases'].append({
                    'row': idx,
                    'true_emotion': emotion_cmed,
                    'action': 'excluded'
                })
                continue
            
            # Record if masked
            if prediction.get('is_masked', False):
                stats['masked'] += 1
                stats['contempt_cases'].append({
                    'row': idx,
                    'true_emotion': emotion_cmed,
                    'original_prediction': prediction['original_top_class'],
                    'masked_prediction': prediction['predicted_label'],
                    'action': 'masked'
                })
            
            # Get predicted label
            predicted_opencv = prediction['predicted_label']
            predicted_cmed = get_cmed_label(predicted_opencv)
            
            # Map true emotion to comparable format
            true_emotion_normalized = emotion_cmed.lower().replace(' ', '_')
            if true_emotion_normalized == 'no_emotion':
                true_emotion_normalized = 'neutral'
            elif true_emotion_normalized == 'anger':
                true_emotion_normalized = 'angry'
            
            # Check correctness
            is_correct = (predicted_cmed == true_emotion_normalized)
            
            if is_correct:
                stats['correct'] += 1
            
            # Calculate confidence gap
            sorted_probs = sorted(prediction['probabilities'].values(), reverse=True)
            confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            
            # Store result
            result = {
                'file_name': image_path.name,
                'parent_folder': subject,
                'folder_label': emotion_cmed,
                'true_emotion': true_emotion_normalized,
                'predicted_emotion': predicted_cmed,
                'predicted_opencv_label': predicted_opencv,
                'is_correct': is_correct,
                'confidence': prediction['confidence'],
                'confidence_gap': confidence_gap,
                'probabilities': prediction['probabilities'],
                'raw_scores': prediction['raw_scores'],
                'strategy': prediction['strategy'],
                'is_masked': prediction.get('is_masked', False),
                'original_top_class': prediction.get('original_top_class', predicted_opencv)
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
            'model': 'OpenCV FER+',
            'model_path': MODEL_PATH,
            'dataset': 'CMED',
            'contempt_strategy': model.contempt_strategy,
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
    print(f"\nStrategy: {model.contempt_strategy.upper()}")
    print(f"\nTotal samples: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    
    if model.contempt_strategy == 'mask':
        print(f"Masked (contempt was top-1): {stats['masked']}")
    elif model.contempt_strategy == 'exclude':
        print(f"Excluded (contempt predictions): {stats['excluded']}")
    
    print(f"\nCorrect predictions: {stats['correct']}")
    
    if stats['success'] > 0:
        accuracy = stats['correct'] / stats['success']
        print(f"Accuracy: {accuracy:.4f} ({stats['correct']}/{stats['success']})")
    
    # Print contempt cases summary
    if stats['contempt_cases']:
        print(f"\nContempt-related cases: {len(stats['contempt_cases'])}")
        if model.contempt_strategy == 'mask':
            print(f"  (Cases where contempt was originally predicted as top-1)")
            print(f"  These were re-predicted using the next highest probability")
        elif model.contempt_strategy == 'exclude':
            print(f"  (Cases excluded from evaluation)")
    
    if stats['errors']:
        print(f"\nErrors: {len(stats['errors'])}")
        print("First 5 errors:")
        for error in stats['errors'][:5]:
            print(f"  - Row {error['row']}: {error['reason']}")
    
    print("\n" + "=" * 80)
    print("Testing Complete. ")
    print("=" * 80)
    
    return df_results, stats



# Strategy Comparison

def compare_strategies(csv_path, images_root):
    """
    Test all strategies and compare results
    """
    print("\n" + "=" * 80)
    print("COMPARING ALL CONTEMPT HANDLING STRATEGIES")
    print("=" * 80)
    
    strategies = ['mask', 'disgust', 'neutral', 'exclude']
    comparison_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Testing Strategy: {strategy.upper()}")
        print(f"{'='*80}")
        
        # Create model with this strategy
        model = OpenCVFERModel(MODEL_PATH, contempt_strategy=strategy)
        
        # Generate output paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_excel = os.path.join(OUTPUT_DIR, f"opencv_fer_results_{strategy}_{timestamp}.xlsx")
        output_json = os.path.join(OUTPUT_DIR, f"opencv_fer_results_{strategy}_{timestamp}.json")
        
        # Test
        df_results, stats = test_dataset(model, csv_path, images_root, output_excel, output_json)
        
        # Store comparison data
        accuracy = stats['correct'] / stats['success'] if stats['success'] > 0 else 0
        comparison_results[strategy] = {
            'total': stats['total'],
            'processed': stats['processed'],
            'success': stats['success'],
            'correct': stats['correct'],
            'accuracy': accuracy,
            'excluded': stats.get('excluded', 0),
            'masked': stats.get('masked', 0),
            'contempt_cases': len(stats.get('contempt_cases', []))
        }
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(comparison_results).T
    df_comparison.index.name = 'Strategy'
    
    # Save comparison
    comparison_path = OUTPUT_COMPARISON
    df_comparison.to_excel(comparison_path)
    
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print("\n" + df_comparison.to_string())
    print(f"\n Comparison saved to: {comparison_path}")
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
'mask' (Probability Masking) 
  - Most fair and rigorous approach
  - Asks: "If not contempt, what's the next best match?"
  - Reflects model's true ability on CMED's 7 emotions
  
'disgust' (Map to Disgust)
  - Psychologically justified (Ekman's theory)
  - Use if contempt predictions are rare (<1%)
  
'neutral' (Map to Neutral)
  - Visually similar expressions
  - Conservative approach
  
'exclude' (Exclude Contempt)
  - Most conservative
  - Use only if contempt predictions are very frequent (>5%)
    """)
    
    return df_comparison



# Validation


def validate_model_setup():
    """Validate setup before running full test"""
    print("=" * 80)
    print("Validating Model Setup")
    print("=" * 80)
    
    # Check model file
    print("\n1. Checking model file...")
    if not os.path.exists(MODEL_PATH):
        print(f" Model file not found: {MODEL_PATH}")
        return False
    print(f" Model file exists")
    
    # Check CSV
    print("\n2. Checking metadata CSV...")
    if not os.path.exists(CSV_PATH):
        print(f" CSV file not found: {CSV_PATH}")
        return False
    print(f" CSV file exists")
    
    # Check images directory
    print("\n3. Checking images directory...")
    if not os.path.exists(APEX_IMAGES_ROOT):
        print(f" Images directory not found: {APEX_IMAGES_ROOT}")
        return False
    print(f" Images directory exists")
    
    # Load model
    print("\n4. Loading model...")
    try:
        model = OpenCVFERModel(MODEL_PATH, contempt_strategy=CONTEMPT_STRATEGY)
        print(" Model loaded successfully")
    except Exception as e:
        print(f" Failed to load model: {e}")
        return False
    
    # Test on sample
    print("\n5. Testing on sample image...")
    try:
        # Find a test image
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
            print("   Prediction was excluded (contempt)")
        else:
            print(f"    Prediction successful")
            print(f"    Strategy: {prediction['strategy']}")
            print(f"    Predicted: {prediction['predicted_label']}")
            print(f"    Confidence: {prediction['confidence']:.4f}")
            
            if prediction.get('is_masked'):
                print(f"    Original top-1: {prediction['original_top_class']}")
                print(f"    After masking: {prediction['predicted_label']}")
            
            # Verify probability format
            prob_sum = sum(prediction['probabilities'].values())
            print(f"    Probability sum: {prob_sum:.6f}")
            
            if abs(prob_sum - 1.0) > 0.01:
                print(f"   Probability sum != 1.0")
                return False
            
            print(f"   Probability format correct")
    
    except Exception as e:
        print(f" Test prediction failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print(" All validation checks passed!")
    print("=" * 80)
    
    return True


# Main Function


def main():
    """Main function"""
    print("=" * 80)
    print("OpenCV FER+ Testing with Configurable Contempt Handling")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate
    if not validate_model_setup():
        print("\n✗ Validation failed")
        return
    
    # Ask user what to do
    print("\n" + "=" * 80)
    print("Choose Testing Mode:")
    print("=" * 80)
    print("1. Test with current strategy only")
    print("2. Compare all strategies")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '2':
        # Compare all strategies
        df_comparison = compare_strategies(CSV_PATH, APEX_IMAGES_ROOT)
    else:
        # Test with current strategy
        model = OpenCVFERModel(MODEL_PATH, contempt_strategy=CONTEMPT_STRATEGY)
        df_results, stats = test_dataset(model, CSV_PATH, APEX_IMAGES_ROOT, 
                                         OUTPUT_EXCEL, OUTPUT_JSON)
    
    print(f"\n{'='*80}")
    print("All Done!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
