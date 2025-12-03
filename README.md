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
Here is the English version formatted for a GitHub README. It integrates the information about the FER+ model, the CMED dataset requirements (APEX frames), and the preprocessing steps you implemented.

-----

#### Emotion Recognition Benchmark: OpenCV (FER+)

To evaluate the performance of established models on the CMED (Child Micro-Expression Dataset), we integrated the **Emotion FERPlus** Deep Convolutional Neural Network via OpenCV's DNN module.

###### 1\. Model Selection

We utilized the **Emotion FERPlus (Opset 8)** version for benchmarking.

  * [cite_start]**Architecture:** Deep Convolutional Neural Network (CNN)[cite: 1, 48].
  * **Selection Rationale:** We selected the full-precision (Float32) version over the quantized int8 version to ensure maximum inference accuracy during benchmarking. [cite_start]The Opset 8 version provides high stability with the `cv2.dnn` module[cite: 8, 12].
  * [cite_start]**Training Background:** The model was trained using Cross Entropy Loss on FER+ annotations, which correct labeling errors found in the standard FER dataset[cite: 48].

**Model Details:**

  * **Filename:** `emotion-ferplus-8.onnx`
  * [cite_start]**Size:** 34 MB [cite: 8]
  * **Source:** [ONNX Model Zoo](https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx)

##### 2\. Pipeline & Preprocessing

[cite_start]Since the CMED dataset consists of video files[cite: 3], while the FER+ model requires specific static image inputs, we implemented the following processing pipeline:

##### A. APEX Frame Extraction (Input Data)

[cite_start]Instead of analyzing random video frames, we utilized the CMED metadata CSV to extract the **APEX Frame** for each subject[cite: 28].

  * [cite_start]**Definition:** The APEX frame represents the moment of maximum facial muscle intensity within the micro-expression sequence[cite: 5, 35].
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

##### Post-Processing

I apply `np.argmax` to the output scores to identify the dominant emotion. This prediction is then compared against the CMED ground truth to calculate the final accuracy metric.


# 2. Deepface 



# 3. LibraFace 
