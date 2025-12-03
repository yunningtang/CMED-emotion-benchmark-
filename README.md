# CMED-emotion-benchmark
CMED: Lightweight benchmark of DeepFace, OpenCV, and LibraFace emotion classifiers with aggregated accuracy metrics


## Dataset Cleaning and APEX Frame Extraction

### Cleaning Process

The CMED dataset underwent a rigorous validation and cleaning process to ensure data quality and consistency. From the original metadata containing **10,924 video samples**, we performed the following validation steps:

1. **Video File Validation**: Verified the physical existence of all video files referenced in the metadata
2. **Frame Index Validation**: Checked that APEX frame indices were within valid ranges for each video
3. **Extractability Testing**: Ensured all frames could be successfully extracted and processed

### Validation Results

- **Total Original Samples**: 10,924
- **Successfully Validated**: 10,731 (98.2%)
- **Removed Samples**: 193 (1.8%)

**Breakdown of Removed Samples**:
- Missing video files: 136 samples (primarily "no emotion" category)
- Frame index out of range: 57 samples (APEX frame exceeded video length)

All removed samples were documented to ensure transparency and reproducibility of the cleaning process.

### Final APEX Frame Distribution

After extraction, I obtained **10,731 APEX frames** representing the moments of peak emotional expression. The distribution across emotion categories is as follows:

| Emotion | Count | Percentage | 
|---------|-------|------------|
| No Emotion | 5,297 | 49.4% |
| Happy | 4,806 | 44.8% |
| Surprise | 350 | 3.3% |
| Anger | 113 | 1.1% |
| Disgust | 105 | 1.0% | 
| Sad | 41 | 0.4% | 
| Fear | 19 | 0.2% | **Total**: 10,731 frames



# 1. OpenCV model

#### Emotion Recognition Benchmark: OpenCV (FER+)

To evaluate the performance of established models on the CMED (Child Micro-Expression Dataset), we integrated the **Emotion FERPlus** Deep Convolutional Neural Network via OpenCV's DNN module.

###### 1\. Model Selection

We utilized the **Emotion FERPlus (Opset 8)** version for benchmarking.

  * **Architecture:** Deep Convolutional Neural Network (CNN).
  * **Selection Rationale:** We selected the full-precision (Float32) version over the quantized int8 version to ensure maximum inference accuracy during benchmarking. The Opset 8 version provides high stability with the `cv2.dnn` module.
  * **Training Background:** The model was trained using Cross Entropy Loss on FER+ annotations, which corrects labelling errors found in the standard FER dataset.

**Model Details:**

  * **Filename:** `emotion-ferplus-8.onnx`
  * **Size:** 34 MB 
  * **Source:** https://huggingface.co/onnxmodelzoo/emotion-ferplus-8

##### 2\. Pipeline & Preprocessing

Since the CMED dataset consists of video files, while the FER+ model requires specific static image inputs, we implemented the following processing pipeline:

##### A. APEX Frame Extraction (Input Data)

Instead of analyzing random video frames, we utilized the CMED metadata CSV to extract the **APEX Frame** for each subject.

  * **Definition:** The APEX frame represents the moment of maximum facial muscle intensity within the micro-expression sequence.
  * **Benefit:** Extracting this specific frame ensures the model is evaluated on the most expressive data point available for each sample.

##### B. Image Preprocessing

The FER+ model mandates a specific input shape of `(N, 1, 64, 64)`. Our testing script (`test_opencv_final.py`) performs the following transformations:

1.  **Grayscale Conversion:** The input channel is set to 1, so RGB/BGR images are converted to grayscale.
2.  **Resize:** Images are resized to **64x64** pixels.
3.  **Reshape:** Dimensions are expanded to `(1, 1, 64, 64)` to satisfy the batch input requirement.

<!-- end list -->

```python
# Core preprocessing logic
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (64, 64))
# Reshape to (1, 1, 64, 64)
blob = np.expand_dims(np.expand_dims(img_resized, axis=0), axis=0)
```

#### 3\. Inference & Output Mapping

##### Output Labels

The model outputs a `1x8` array of scores corresponding to 8 emotion classes. We map these to the CMED labels as follows:

| FER+ Output Label | Index | Mapped CMED Label |
| :--- | :--- | :--- |
| **Neutral** | 0 | No Emotion |
| **Happiness** | 1 | Happy |
| **Surprise** | 2 | Surprise |
| **Sadness** | 3 | Sad |
| **Anger** | 4 | Anger |
| **Disgust** | 5 | Disgust |
| **Fear** | 6 | Fear |
| **Contempt** | 7 | *Ignored (Not in CMED)* |
##### Handling Label Discrepancy: The "Contempt" 

A key challenge in benchmarking the FER+ model against the CMED dataset is that FER+ includes an 8th class, 'Contempt', which does not exist in the 7-class CMED ground truth.
Since the model is pre-trained to recognize 'Contempt', it may predict this emotion for certain CMED samples. To fairly evaluate accuracy, we implemented a configurable handling strategy.
Tested Strategy: Probability Masking ('mask')

For the primary benchmark results, we utilized the Probability Masking strategy. This is considered the most rigorous approach as it forces the model to select its "second-best" guess without altering the underlying model weights or arbitrarily mapping labels.

How it works:

    Inference: The model produces raw probability scores for all 8 classes.

    Detection: We check if 'Contempt' has the highest probability.

    Masking: If true, I manually set the probability of the 'Contempt' class to 0.0.

    Re-evaluation: Then re-selecting the emotion with the next highest probability from the remaining 7 valid classes (e.g., Neutral, Anger, Disgust).

    Prediction: This new top-scoring emotion is used as the final prediction for accuracy calculation.


###### Post-Processing

I apply `np.argmax` to the output scores to identify the dominant emotion. This prediction is then compared against the CMED ground truth to calculate the final accuracy metric.


# 2. Deepface 



# 3. LibraFace 
