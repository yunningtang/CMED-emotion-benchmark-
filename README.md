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



# 2. Deepface 



# 3. LibraFace 
