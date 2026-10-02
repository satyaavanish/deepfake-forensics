# Deepfake Forensics — Multimodal Deepfake Detection System

A multimodal deepfake forensics system that analyzes **video, audio, and behavioral signals** to detect manipulated media.

The system combines:

- **Vision Transformer (ViT)** for visual representation
- **Audio Spectrogram Transformer (AST)** for audio analysis
- **Behavioral features** including eye blinks, lip movement, speech–lip consistency, and rPPG
- **Cross-modal attention fusion** for multimodal classification
- **Explainability modules** for confidence scores, anomaly timestamps, forensic reports, and attention visualization
- **Streamlit dashboard** for interactive inference

---

## 1. Project Overview

Deepfakes can manipulate different modalities independently. A video may contain:

- fake video + fake audio
- fake video + real audio
- real video + fake audio
- real video + real audio

This project therefore treats deepfake detection as a **multimodal forensic problem** rather than relying only on visual artifacts.

The system extracts information from:

```text
Video
  ├── Face frames → ViT
  └── Behavioral signals
        ├── Eye blink
        ├── Lip movement
        ├── Speech–lip consistency
        └── rPPG

Audio
  └── Spectrogram → AST

        ↓

Cross-Modal Attention Fusion

        ↓

REAL / DEEPFAKE

        ↓

Forensic Evidence + Explainability
```

---

## 2. Main Features

### Visual Analysis

- Face detection using MTCNN
- Uniform frame sampling
- 224 × 224 face crops
- ViT-based visual embeddings
- Visual attention analysis

### Audio Analysis

- Audio extraction using FFmpeg
- 16 kHz mono audio
- Mel-spectrogram generation
- AST-based audio representation
- Separate audio manipulation classifier

### Behavioral Analysis

The system extracts:

- Eye blink count
- Blink rate
- Mean blink duration
- Blink interval statistics
- Eye Aspect Ratio (EAR)
- Mouth opening
- Lip distance
- Mouth movement velocity
- Mouth movement frequency
- Speech activity ratio
- Lip activity ratio
- Lip–speech consistency
- Heart-rate estimate
- Pulse consistency
- Temporal signal quality
- rPPG availability and duration

### Multimodal Fusion

The fusion model combines:

```text
Visual Embedding  → 512
Audio Embedding   → 512
Behavior Features → 512

        ↓

Cross-Modal Attention

        ↓

Fused Representation

        ↓

Binary Classification
```

The model uses cross-attention between visual and audio representations and incorporates behavioral features before final classification.

### Explainability

The project includes modules for:

- Forensic prediction
- Confidence scores
- Anomaly timestamps
- Anomaly timeline
- ViT attention visualization
- Causal-style evidence aggregation
- Forensic report generation

> The causal reasoning module is an evidence aggregation mechanism and is not intended to claim formal causal inference.

---

## 3. Dataset

The project uses the **FakeAVCeleb** dataset.

The local dataset contains:

- **1006 videos**
- **91 identities**
- Average duration: approximately **5.07 seconds**
- Average FPS: approximately **25.06**
- All videos contain audio

### Manipulation categories

| Category | Description |
|---|---|
| `fake_video_fake_audio` | Fake video + fake audio |
| `fake_video_real_audio` | Fake video + real audio |
| `real_video_fake_audio` | Real video + fake audio |
| `real_video_real_audio` | Real video + real audio |

### Manipulation methods found

- Wav2Lip
- FaceSwap
- Unknown / unspecified

The raw dataset is intentionally **not included in this GitHub repository** because of its size and dataset distribution considerations.

---

## 4. Identity-Aware Dataset Split

To reduce identity leakage, the dataset was split at the identity level.

```text
Train: 64 identities
Validation: 14 identities
Test: 13 identities
```

Video counts:

```text
Train:      711
Validation: 179
Test:       116
Total:     1006
```

The raw dataset and generated preprocessing files are excluded from Git using `.gitignore`.

---

## 5. Project Structure

```text
deepfake-forensics/
│
├── app/
│   └── dashboard.py
│
├── src/
│   ├── preprocessing/
│   │   ├── explore_dataset.py
│   │   ├── create_split.py
│   │   ├── preprocess_dataset.py
│   │   ├── extract_audio.py
│   │   ├── generate_spectrograms.py
│   │   └── ...
│   │
│   ├── models/
│   │   ├── train_vit.py
│   │   ├── train_ast.py
│   │   ├── extract_vit_embeddings.py
│   │   ├── extract_ast_embeddings.py
│   │   └── ...
│   │
│   ├── features/
│   │   ├── extract_eye_blink.py
│   │   ├── extract_lip_speech.py
│   │   ├── extract_rppg.py
│   │   └── merge_behavior_features.py
│   │
│   ├── fusion/
│   │   ├── fusion_model.py
│   │   ├── cross_attention.py
│   │   ├── cross_attention_architecture.py
│   │   └── train_fusion.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_vit.py
│   │   ├── evaluate_ast.py
│   │   ├── evaluate_ast_audio.py
│   │   └── evaluate_fusion.py
│   │
│   └── explainability/
│       ├── generate_forensic_prediction.py
│       ├── generate_forensic_report.py
│       ├── generate_confidence_score.py
│       ├── generate_anomaly_timestamps.py
│       ├── generate_anomaly_timeline.py
│       ├── visualize_vit_attention.py
│       └── causal_reasoning.py
│
├── notebooks/
├── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

Large datasets, generated outputs, and model checkpoints are excluded from the GitHub repository.

---

## 6. Technologies Used

### Programming

- Python
- PyTorch
- NumPy
- Pandas

### Computer Vision

- OpenCV
- MTCNN
- MediaPipe
- Hugging Face Transformers

### Deep Learning Models

- Vision Transformer (`google/vit-base-patch16-224-in21k`)
- Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`)
- Cross-modal attention fusion network

### Audio Processing

- FFmpeg
- Librosa
- SciPy

### Application

- Streamlit

---

## 7. Local Environment

The project was developed and tested with:

```text
Python: 3.10.14
PyTorch: 2.4.0+cu121
CUDA: 12.1
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
NumPy: 1.26.4
Transformers: 4.46.3
OpenCV: 4.10.0
MediaPipe: 0.10.14
```

Create/activate the environment:

```powershell
conda activate deepfake_forensics
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 8. Processing Pipeline

### Phase 1 — Dataset Exploration

```text
FakeAVCeleb
    ↓
Metadata extraction
    ↓
Identity identification
    ↓
Category analysis
```

### Phase 2 — Identity-Aware Split

```text
91 identities
    ↓
70 / 15 / 15 identity split
    ↓
Train / Validation / Test
```

### Phase 3 — Video Processing

```text
Video
 ↓
Uniform frame sampling
 ↓
MTCNN face detection
 ↓
224 × 224 face crops
```

16 face frames are used per video.

### Phase 4 — Audio Processing

```text
Video
 ↓
FFmpeg
 ↓
16 kHz mono WAV
 ↓
Mel spectrogram
 ↓
AST
```

### Phase 5 — Visual Model

```text
Face frames
 ↓
ViT
 ↓
768-dimensional representation
 ↓
Video-level representation
```

### Phase 6 — Audio Model

The audio manipulation classifier uses:

```text
Audio
 ↓
AST backbone
 ↓
768-dimensional audio representation
 ↓
256-dimensional classifier
 ↓
REAL AUDIO / FAKE AUDIO
```

### Phase 7 — Behavioral Features

```text
Face frames + video
        ↓
Eye blink
Lip movement
Speech–lip consistency
rPPG
        ↓
22 behavioral features
```

### Phase 8 — Cross-Modal Fusion

```text
ViT embedding
      +
AST embedding
      +
Behavioral features
      ↓
Cross-modal attention
      ↓
Fusion representation
      ↓
REAL / DEEPFAKE
```

---

## 9. Model Results

### ViT Baseline

The final ViT evaluation on the test split produced:

```text
Accuracy : 0.9828
Precision: 0.9817
Recall   : 1.0000
F1       : 0.9907
ROC-AUC  : 1.0000
```

### Fusion Model

Using the final four-category-to-binary label definition:

```text
fake_video_fake_audio → DEEPFAKE
fake_video_real_audio → DEEPFAKE
real_video_fake_audio → DEEPFAKE
real_video_real_audio → REAL
```

The test set contained:

```text
REAL      : 4
DEEPFAKE  : 112
```

The reported test result was:

```text
Accuracy : 1.0000
Precision: 1.0000
Recall   : 1.0000
F1       : 1.0000
ROC-AUC  : 1.0000
```

The test set contains only a small number of real samples and only 5 `real_video_fake_audio` samples, so these results should be interpreted with that limitation in mind.

### Audio Classifier

The dedicated AST audio classifier was trained with:

```text
fake_video_fake_audio → FAKE AUDIO
fake_video_real_audio → REAL AUDIO
real_video_fake_audio → FAKE AUDIO
real_video_real_audio → REAL AUDIO
```

Its final training run reached:

```text
Validation Accuracy: 1.0000
```

Test-set performance should be taken from `evaluate_ast_audio.py` after evaluation rather than inferred from training/validation performance.

---

## 10. Dashboard

The project includes a Streamlit dashboard:

```text
app/dashboard.py
```

Run locally:

```powershell
streamlit run app/dashboard.py
```

Then open:

```text
http://localhost:8501
```

The dashboard supports:

```text
Upload video
     ↓
Frame extraction
     ↓
Face detection
     ↓
ViT analysis
     ↓
Audio extraction
     ↓
AST audio analysis
     ↓
Behavioral analysis
     ↓
Cross-modal fusion
     ↓
Forensic result
```

The dashboard reports both:

```text
Audio Result
    REAL AUDIO / FAKE AUDIO

Multimodal Result
    REAL / DEEPFAKE
```

---

## 11. Model Checkpoints

The trained checkpoints are intentionally excluded from GitHub because they are large.

The local project uses:

```text
checkpoints/
├── ast_audio_best.pth
├── ast_baseline_best.pth
├── cross_modal_fusion_best.pth
└── vit_baseline_best.pth
```

The active dashboard requires the trained:

```text
ast_audio_best.pth
cross_modal_fusion_best.pth
vit_baseline_best.pth
```

The baseline AST checkpoint is retained locally for comparison.

If deploying the application, the trained checkpoints must be provided through a suitable model-storage mechanism rather than committing the large files directly to GitHub.

---

## 12. Important Notes

### Dataset

The FakeAVCeleb dataset is not included in this repository.

### Generated Data

The following are excluded from GitHub:

```text
dataset/
outputs/
checkpoints/
```

### GPU

The project was developed using an NVIDIA RTX 3060 Laptop GPU. CPU execution may require substantially more time and may require additional memory optimizations.

### rPPG

rPPG extraction requires sufficient temporal video length. Videos that are too short may not produce valid rPPG features.

### Explainability

Attention maps, anomaly timestamps, and behavioral evidence are intended as forensic indicators. They should not be interpreted as definitive proof that a particular frame or timestamp was manipulated.

---

## 13. Research Architecture

The core research architecture is:

```text
                    ┌─────────────────┐
                    │     Video       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Face Extraction │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        ┌───────▼────────┐       ┌───────▼────────┐
        │      ViT       │       │   Behavioral   │
        │ Visual Feature │       │    Features    │
        └───────┬────────┘       └───────┬────────┘
                │                        │
                │                        │
        ┌───────▼────────┐       ┌───────▼────────┐
        │ Visual 512-d   │       │ Behavior 512-d │
        └───────┬────────┘       └───────┬────────┘
                │                        │
                │      ┌─────────────────┘
                │      │
        ┌───────▼──────▼─────────────────┐
        │       Cross-Modal Attention    │
        └───────────────┬────────────────┘
                        │
              ┌─────────▼─────────┐
              │ Fusion Embedding  │
              └─────────┬─────────┘
                        │
                  ┌─────▼─────┐
                  │ Classifier│
                  └─────┬─────┘
                        │
                 REAL / DEEPFAKE

       Audio ──→ Mel Spectrogram ──→ AST
```

---

## 14. Repository

Source code:

**GitHub:** `https://github.com/satyaavanish/deepfake-forensics`

---

## 15. Future Work

Potential extensions include:

- Larger and more diverse datasets
- Additional deepfake manipulation types
- Stronger temporal modeling
- Improved speech–lip synchronization analysis
- More robust rPPG estimation
- End-to-end multimodal training
- Larger-scale external test sets
- Improved localization of manipulated regions
- Cloud deployment of the Streamlit dashboard

---

## 16. Disclaimer

This project is intended for **research and educational purposes** in multimedia forensics and deepfake detection.

Model predictions are probabilistic and should not be treated as definitive forensic conclusions without appropriate validation and expert analysis.
