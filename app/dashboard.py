import os



import sys



import re



import tempfile



import subprocess

import shutil



import warnings



import cv2



import numpy as np



import pandas as pd



import streamlit as st



import torch



import torch.nn as nn



import librosa



from PIL import Image



from facenet_pytorch import MTCNN



from transformers import (



    ViTImageProcessor,



    ViTModel,



    ASTFeatureExtractor,



    ASTModel,



)



# ============================================================



# PROJECT ROOT



# ============================================================



PROJECT_ROOT = os.path.abspath(



    os.path.join(



        os.path.dirname(__file__),



        ".."



    )



)



sys.path.insert(0, PROJECT_ROOT)



# ============================================================



# IMPORT YOUR TRAINED FUSION MODEL



# ============================================================



from src.fusion.fusion_model import CrossModalFusionModel



# ============================================================



# PATHS



# ============================================================



CHECKPOINT_PATH = os.path.join(



    PROJECT_ROOT,



    "checkpoints",



    "cross_modal_fusion_best.pth"



)

AST_AUDIO_CHECKPOINT_PATH = os.path.join(

    PROJECT_ROOT,

    "checkpoints",

    "ast_audio_best.pth"

)



BEHAVIOR_FEATURE_COLUMNS = [

    "blink_count", "blink_rate", "mean_blink_duration",

    "blink_interval_mean", "blink_interval_variance", "mean_ear",

    "ear_variance", "mean_mouth_opening", "mouth_opening_variance",

    "mean_lip_distance", "lip_distance_variance",

    "mouth_movement_velocity", "mouth_movement_frequency",

    "speech_activity_ratio", "lip_activity_ratio",

    "lip_speech_consistency", "heart_rate_estimate",

    "pulse_consistency", "temporal_signal_quality", "rppg_frames",

    "rppg_duration_seconds", "rppg_available"

]





BEHAVIOR_CSV = os.path.join(



    PROJECT_ROOT,



    "outputs",



    "features",



    "behavior_features.csv"



)



HEATMAP_DIR = os.path.join(



    PROJECT_ROOT,



    "outputs",



    "explainability",



    "vit_heatmaps"



)



# ============================================================



# SETTINGS



# ============================================================



DEVICE = torch.device(



    "cuda"



    if torch.cuda.is_available()



    else "cpu"



)



NUM_FRAMES = 16



FACE_SIZE = 224



AUDIO_SR = 16000



# ============================================================



# PAGE CONFIG



# ============================================================



st.set_page_config(



    page_title="Deepfake Forensics",



    page_icon="🔬",



    layout="wide"



)



# ============================================================



# CSS



# ============================================================



st.markdown(



    """



    <style>



    .stApp {



        background:



        radial-gradient(



            circle at top right,



            #18243a 0%,



            #0a0f1d 40%,



            #050810 100%



        );



    }



    .title {



        font-size: 42px;



        font-weight: 900;



        letter-spacing: 1px;



    }



    .subtitle {



        color: #9aa8bd;



        font-size: 16px;



        margin-bottom: 30px;



    }



    .result-card {



        padding: 30px;



        border-radius: 18px;



        text-align: center;



        margin: 10px 0 25px 0;



    }



    .real-card {



        background: rgba(34,197,94,0.12);



        border: 1px solid rgba(34,197,94,0.4);



    }



    .fake-card {



        background: rgba(239,68,68,0.12);



        border: 1px solid rgba(239,68,68,0.4);



    }



    .result {



        font-size: 44px;



        font-weight: 900;



    }



    .confidence {



        font-size: 20px;



        margin-top: 8px;



    }



    .evidence-card {



        background: rgba(20,27,43,0.9);



        border: 1px solid rgba(255,255,255,0.08);



        border-radius: 14px;



        padding: 20px;



        min-height: 120px;



    }



    .high {



        color: #ff6b6b;



        font-weight: 800;



    }



    .medium {



        color: #f5c451;



        font-weight: 800;



    }



    .low {



        color: #62d68b;



        font-weight: 800;



    }



    .unavailable {



        color: #9ca8bd;



        font-weight: 800;



    }



    .timestamp {



        display: inline-block;



        padding: 10px 14px;



        margin: 5px;



        border-radius: 10px;



        background: rgba(239,68,68,0.12);



        border: 1px solid rgba(239,68,68,0.3);



        color: #ff8a8a;



        font-weight: 700;



    }



    </style>



    """,



    unsafe_allow_html=True



)



# ============================================================



# LOAD MODELS



# ============================================================



@st.cache_resource



class ASTAudioClassifier(nn.Module):

    """Exact architecture used by train_ast.py.

    0 = REAL AUDIO, 1 = FAKE AUDIO.

    """



    def __init__(self):

        super().__init__()

        self.ast = ASTModel.from_pretrained(

            "MIT/ast-finetuned-audioset-10-10-0.4593"

        )

        for param in self.ast.parameters():

            param.requires_grad = False



        self.classifier = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 2)

        )



    def forward(self, input_values):

        outputs = self.ast(input_values=input_values)

        audio_embedding = outputs.last_hidden_state.mean(dim=1)

        return self.classifier(audio_embedding)





def load_models():

    print("Loading models...")



    vit_processor = ViTImageProcessor.from_pretrained(

        "google/vit-base-patch16-224-in21k"

    )

    vit_model = ViTModel.from_pretrained(

        "google/vit-base-patch16-224-in21k"

    ).to(DEVICE)

    vit_model.eval()



    ast_processor = ASTFeatureExtractor.from_pretrained(

        "MIT/ast-finetuned-audioset-10-10-0.4593"

    )

    ast_model = ASTModel.from_pretrained(

        "MIT/ast-finetuned-audioset-10-10-0.4593"

    ).to(DEVICE)

    ast_model.eval()



    ast_audio_classifier = ASTAudioClassifier().to(DEVICE)

    audio_checkpoint = torch.load(

        AST_AUDIO_CHECKPOINT_PATH,

        map_location=DEVICE,

        weights_only=False

    )

    audio_state = (

        audio_checkpoint["model_state_dict"]

        if isinstance(audio_checkpoint, dict)

        and "model_state_dict" in audio_checkpoint

        else audio_checkpoint

    )

    ast_audio_classifier.load_state_dict(audio_state)

    ast_audio_classifier.eval()



    mtcnn = MTCNN(

        image_size=224,

        margin=20,

        min_face_size=20,

        thresholds=[0.6, 0.7, 0.7],

        factor=0.709,

        post_process=False,

        keep_all=True,

        device=DEVICE

    )



    fusion_model = CrossModalFusionModel(

        visual_dim=768,

        audio_dim=768,

        behavior_dim=22,

        common_dim=512,

        behavior_projection_dim=128,

        num_heads=8,

        dropout=0.2

    )



    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE,

        weights_only=False

    )

    fusion_state = (

        checkpoint["model_state_dict"]

        if isinstance(checkpoint, dict)

        and "model_state_dict" in checkpoint

        else checkpoint

    )

    fusion_model.load_state_dict(fusion_state)

    fusion_model = fusion_model.to(DEVICE)

    fusion_model.eval()



    return (

        vit_processor,

        vit_model,

        ast_processor,

        ast_model,

        ast_audio_classifier,

        mtcnn,

        fusion_model

    )









# LOAD BEHAVIOR TRAINING STATISTICS



# ============================================================



@st.cache_data



def load_behavior_statistics():

    df = pd.read_csv(BEHAVIOR_CSV)



    missing = [

        c for c in BEHAVIOR_FEATURE_COLUMNS

        if c not in df.columns

    ]

    if missing:

        raise ValueError(

            "Missing behavior features: " + ", ".join(missing)

        )



    train_df = df[

        df["split"].astype(str).str.lower() == "train"

    ].copy()



    for column in BEHAVIOR_FEATURE_COLUMNS:

        train_df[column] = pd.to_numeric(

            train_df[column],

            errors="coerce"

        )



    medians = train_df[BEHAVIOR_FEATURE_COLUMNS].median()

    filled = train_df[BEHAVIOR_FEATURE_COLUMNS].fillna(medians)

    means = filled.mean()

    stds = filled.std().replace(0, 1.0).fillna(1.0)



    return (

        BEHAVIOR_FEATURE_COLUMNS,

        medians.fillna(0.0),

        means.fillna(0.0),

        stds

    )









# EXTRACT VIDEO FRAMES



# ============================================================



def extract_frames(



    video_path,



    num_frames=16



):



    cap = cv2.VideoCapture(



        video_path



    )



    if not cap.isOpened():



        raise RuntimeError(



            "Could not open uploaded video."



        )



    total_frames = int(



        cap.get(



            cv2.CAP_PROP_FRAME_COUNT



        )



    )



    fps = float(



        cap.get(



            cv2.CAP_PROP_FPS



        )



    )



    if total_frames <= 0:



        cap.release()



        raise RuntimeError(



            "Video contains no readable frames."



        )



    indices = np.linspace(



        0,



        total_frames - 1,



        num_frames



    ).astype(int)



    frames = []



    actual_indices = []



    for index in indices:



        cap.set(



            cv2.CAP_PROP_POS_FRAMES,



            int(index)



        )



        success, frame = cap.read()



        if success:



            frame_rgb = cv2.cvtColor(



                frame,



                cv2.COLOR_BGR2RGB



            )



            frames.append(



                frame_rgb



            )



            actual_indices.append(



                index



            )



    cap.release()



    if len(frames) == 0:



        raise RuntimeError(



            "Could not extract frames."



        )



    duration = (



        total_frames / fps



        if fps > 0



        else 0



    )



    return (



        frames,



        actual_indices,



        fps,



        duration



    )



# ============================================================



# FACE DETECTION



# ============================================================



def detect_faces(frames, mtcnn):

    """Detect the largest face in each sampled frame.



    Uses float32 input first to avoid the facenet-pytorch/numpy dtype

    conversion problem encountered with mtcnn.detect() in this environment.

    If detection fails for a frame, the frame is resized as a safe fallback

    so the dashboard does not crash on an otherwise readable video.

    """

    face_frames = []

    face_boxes = []

    fallback_count = 0



    for frame in frames:

        frame_rgb = np.asarray(frame, dtype=np.uint8)

        frame_rgb = np.ascontiguousarray(frame_rgb)



        boxes = None

        probabilities = None



        try:

            detection_input = np.ascontiguousarray(

                frame_rgb.astype(np.float32)

            )

            boxes, probabilities = mtcnn.detect(detection_input)

        except Exception:

            try:

                pil_image = Image.fromarray(frame_rgb, mode="RGB")

                boxes, probabilities = mtcnn.detect(pil_image)

            except Exception:

                boxes = None

                probabilities = None



        if boxes is None or len(boxes) == 0:

            resized = cv2.resize(

                frame_rgb,

                (FACE_SIZE, FACE_SIZE),

                interpolation=cv2.INTER_AREA

            )

            face_frames.append(resized.astype(np.uint8))

            face_boxes.append(None)

            fallback_count += 1

            continue



        valid_boxes = []

        areas = []



        for box in boxes:

            if box is None or len(box) != 4:

                continue

            values = np.asarray(box, dtype=float)

            if not np.all(np.isfinite(values)):

                continue



            x1, y1, x2, y2 = values

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            if area > 0:

                valid_boxes.append((x1, y1, x2, y2))

                areas.append(area)



        if not valid_boxes:

            resized = cv2.resize(

                frame_rgb,

                (FACE_SIZE, FACE_SIZE),

                interpolation=cv2.INTER_AREA

            )

            face_frames.append(resized.astype(np.uint8))

            face_boxes.append(None)

            fallback_count += 1

            continue



        best_index = int(np.argmax(areas))

        box = valid_boxes[best_index]

        x1, y1, x2, y2 = [int(round(v)) for v in box]



        x1 = max(0, min(frame_rgb.shape[1] - 1, x1))

        y1 = max(0, min(frame_rgb.shape[0] - 1, y1))

        x2 = max(x1 + 1, min(frame_rgb.shape[1], x2))

        y2 = max(y1 + 1, min(frame_rgb.shape[0], y2))



        crop = frame_rgb[y1:y2, x1:x2]



        if crop.size == 0:

            crop = frame_rgb

            face_boxes.append(None)

            fallback_count += 1

        else:

            face_boxes.append(box)



        crop = cv2.resize(

            crop,

            (FACE_SIZE, FACE_SIZE),

            interpolation=cv2.INTER_AREA

        )

        face_frames.append(crop.astype(np.uint8))



    if not face_frames:

        raise RuntimeError("No usable frames were available for face analysis.")



    detect_faces.last_fallback_count = fallback_count

    return face_frames, face_boxes



detect_faces.last_fallback_count = 0



def extract_vit_embedding(



    face_frames,



    processor,



    model



):



    inputs = processor(



        images=face_frames,



        return_tensors="pt"



    )



    inputs = {



        key: value.to(DEVICE)



        for key, value in inputs.items()



    }



    with torch.no_grad():



        outputs = model(



            **inputs



        )



        cls_embeddings = (



            outputs.last_hidden_state[:, 0, :]



        )



    video_embedding = (



        cls_embeddings



        .mean(dim=0)



    )



    return (



        video_embedding,



        cls_embeddings



    )



# ============================================================



# EXTRACT AUDIO



# ============================================================



def find_ffmpeg():

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:

        return ffmpeg



    candidates = [

        r"C:\Users\siris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",

        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",

        r"C:\ffmpeg\bin\ffmpeg.exe",

    ]



    for candidate in candidates:

        if os.path.exists(candidate):

            return candidate



    raise FileNotFoundError(

        "FFmpeg was not found on PATH or in the known installation paths."

    )





def extract_audio(video_path, output_wav):

    command = [

        find_ffmpeg(),

        "-y",

        "-i",

        video_path,

        "-vn",

        "-ac",

        "1",

        "-ar",

        "16000",

        "-sample_fmt",

        "s16",

        output_wav

    ]



    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE

    )



    if result.returncode != 0:

        raise RuntimeError(

            "FFmpeg audio extraction failed:\n"

            + result.stderr.decode(errors="ignore")[-3000:]

        )



    if not os.path.exists(output_wav):

        raise RuntimeError("FFmpeg did not create the WAV file.")









# AST EMBEDDING



# ============================================================



def extract_ast_embedding(



    wav_path,



    processor,



    model



):



    audio, sr = librosa.load(



        wav_path,



        sr=16000,



        mono=True



    )



    inputs = processor(



        audio,



        sampling_rate=16000,



        return_tensors="pt"



    )



    inputs = {



        key: value.to(DEVICE)



        for key, value in inputs.items()



    }



    with torch.no_grad():



        outputs = model(



            **inputs



        )



        audio_embedding = (



            outputs.last_hidden_state



            .mean(dim=1)



            .squeeze(0)



        )



    return audio_embedding



# ============================================================
# AST AUDIO MANIPULATION PREDICTION
# ============================================================

def predict_audio_manipulation(wav_path, processor, classifier):

    audio, _ = librosa.load(

        wav_path,

        sr=AUDIO_SR,

        mono=True

    )



    if audio is None or len(audio) == 0:

        raise RuntimeError("Extracted audio is empty.")



    inputs = processor(

        audio,

        sampling_rate=AUDIO_SR,

        return_tensors="pt"

    )



    input_values = inputs["input_values"].to(DEVICE)



    with torch.no_grad():

        logits = classifier(input_values)

        probabilities = torch.softmax(logits, dim=1)[0]



    real_audio_probability = float(probabilities[0].item())

    fake_audio_probability = float(probabilities[1].item())



    if fake_audio_probability >= real_audio_probability:

        prediction = "FAKE AUDIO"

        confidence = fake_audio_probability

    else:

        prediction = "REAL AUDIO"

        confidence = real_audio_probability



    return (

        prediction,

        real_audio_probability,

        fake_audio_probability,

        confidence

    )









# MEDIAPIPE



# ============================================================



def get_face_mesh():



    import mediapipe as mp



    return mp.solutions.face_mesh.FaceMesh(



        static_image_mode=True,



        max_num_faces=1,



        refine_landmarks=True,



        min_detection_confidence=0.5



    )



# ============================================================



# LANDMARK DISTANCE



# ============================================================



def distance(a, b):



    return np.sqrt(



        (a.x - b.x) ** 2



        + (a.y - b.y) ** 2



    )



# ============================================================



# EAR



# ============================================================



LEFT_EYE = [



    33,



    160,



    158,



    133,



    153,



    144



]



RIGHT_EYE = [



    362,



    385,



    387,



    263,



    373,



    380



]



def calculate_ear(



    landmarks,



    indices



):



    p1 = landmarks[



        indices[0]



    ]



    p2 = landmarks[



        indices[1]



    ]



    p3 = landmarks[



        indices[2]



    ]



    p4 = landmarks[



        indices[3]



    ]



    p5 = landmarks[



        indices[4]



    ]



    p6 = landmarks[



        indices[5]



    ]



    vertical_1 = distance(



        p2,



        p6



    )



    vertical_2 = distance(



        p3,



        p5



    )



    horizontal = distance(



        p1,



        p4



    )



    if horizontal == 0:



        return 0.0



    return (



        vertical_1



        + vertical_2



    ) / (



        2.0 * horizontal



    )



# ============================================================



# MOUTH



# ============================================================



UPPER_LIP = 13



LOWER_LIP = 14



LEFT_MOUTH = 61



RIGHT_MOUTH = 291



# ============================================================



# BEHAVIOR EXTRACTION



# ============================================================



def extract_behavior(



    face_frames,



    video_path,



    fps,



    audio_wav=None



):



    mesh = get_face_mesh()



    ears = []



    mouth_openings = []



    lip_distances = []



    for frame in face_frames:



        rgb = frame.copy()



        result = mesh.process(



            rgb



        )



        if not result.multi_face_landmarks:



            ears.append(



                np.nan



            )



            mouth_openings.append(



                np.nan



            )



            lip_distances.append(



                np.nan



            )



            continue



        landmarks = (



            result



            .multi_face_landmarks[0]



            .landmark



        )



        left_ear = calculate_ear(



            landmarks,



            LEFT_EYE



        )



        right_ear = calculate_ear(



            landmarks,



            RIGHT_EYE



        )



        ear = (



            left_ear



            + right_ear



        ) / 2.0



        mouth_opening = distance(



            landmarks[UPPER_LIP],



            landmarks[LOWER_LIP]



        ) * 100



        lip_distance = distance(



            landmarks[LEFT_MOUTH],



            landmarks[RIGHT_MOUTH]



        ) * 100



        ears.append(



            ear



        )



        mouth_openings.append(



            mouth_opening



        )



        lip_distances.append(



            lip_distance



        )



    mesh.close()



    ears = np.array(



        ears,



        dtype=float



    )



    mouth_openings = np.array(



        mouth_openings,



        dtype=float



    )



    lip_distances = np.array(



        lip_distances,



        dtype=float



    )



    # Fill missing frame measurements



    # using sequence medians.



    def fill_nan(values):



        if np.all(



            np.isnan(values)



        ):



            return np.zeros_like(



                values



            )



        median = np.nanmedian(



            values



        )



        return np.where(



            np.isnan(values),



            median,



            values



        )



    ears = fill_nan(



        ears



    )



    mouth_openings = fill_nan(



        mouth_openings



    )



    lip_distances = fill_nan(



        lip_distances



    )



    # ========================================================



    # BLINK FEATURES



    # ========================================================



    blink_threshold = 0.21



    below_threshold = (



        ears < blink_threshold



    )



    blink_count = 0



    blink_frames = 0



    current = 0



    for value in below_threshold:



        if value:



            current += 1



        else:



            if current >= 2:



                blink_count += 1



                blink_frames += current



            current = 0



    if current >= 2:



        blink_count += 1



        blink_frames += current



    duration_per_frame = (



        1.0 / fps



        if fps > 0



        else 0.0



    )



    mean_blink_duration = (



        (



            blink_frames



            * duration_per_frame



        )



        / blink_count



        if blink_count > 0



        else 0.0



    )



    total_duration = (



        len(face_frames)



        * duration_per_frame



    )



    blink_rate = (



        blink_count



        / total_duration



        if total_duration > 0



        else 0.0



    )



    if blink_count > 1:



        blink_interval_mean = (



            total_duration



            / blink_count



        )



        blink_interval_variance = 0.0



    else:



        blink_interval_mean = 0.0



        blink_interval_variance = 0.0



    mean_ear = float(



        np.mean(ears)



    )



    ear_variance = float(



        np.var(ears)



    )



    # ========================================================



    # LIP FEATURES



    # ========================================================



    mean_mouth_opening = float(



        np.mean(mouth_openings)



    )



    mouth_opening_variance = float(



        np.var(mouth_openings)



    )



    mean_lip_distance = float(



        np.mean(lip_distances)



    )



    movement = np.abs(



        np.diff(



            mouth_openings



        )



    )



    mouth_movement_velocity = float(



        np.mean(movement)



        if len(movement) > 0



        else 0.0



    )



    if len(mouth_openings) > 2:



        differences = np.diff(



            mouth_openings



        )



        sign_changes = np.sum(



            differences[1:]



            * differences[:-1]



            < 0



        )



        mouth_movement_frequency = (



            float(sign_changes)



            / len(mouth_openings)



        )



    else:



        mouth_movement_frequency = 0.0



    # ========================================================



    # AUDIO / LIP SYNC



    # ========================================================



    try:



        if audio_wav is not None and os.path.exists(audio_wav):



            audio, _ = librosa.load(



                audio_wav,



                sr=AUDIO_SR,



                mono=True



            )



            speech_activity = calculate_speech_activity(



                audio,



                num_segments=len(face_frames)



            )



        else:



            speech_activity = np.zeros(



                len(face_frames),



                dtype=bool



            )



    except Exception:



        speech_activity = np.zeros(



            len(face_frames),



            dtype=bool



        )



    speech_ratio = float(



        np.mean(



            speech_activity



        )



    )



    lip_movement = np.abs(



        np.diff(



            mouth_openings,



            prepend=mouth_openings[0]



        )



    )



    if np.max(



        lip_movement



    ) > 0:



        lip_threshold = max(



            np.median(



                lip_movement



            ),



            0.5



        )



        lip_active = (



            lip_movement



            >= lip_threshold



        )



    else:



        lip_active = np.zeros(



            len(mouth_openings),



            dtype=bool



        )



    lip_activity_ratio = float(



        np.mean(



            lip_active



        )



    )



    min_length = min(



        len(speech_activity),



        len(lip_active)



    )



    if (



        min_length > 0



        and np.sum(



            speech_activity[:min_length]



        ) > 0



    ):



        consistency = (



            np.sum(



                speech_activity[:min_length]



                & lip_active[:min_length]



            )



            / np.sum(



                speech_activity[:min_length]



            )



        )



    else:



        consistency = 0.0



    # ========================================================



    # rPPG



    # ========================================================



    (



        heart_rate,



        pulse_consistency,



        temporal_quality,



        rppg_available



    ) = extract_rppg(



        video_path



    )



    features = {



        "blink_count":



            blink_count,



        "blink_rate":



            blink_rate,



        "mean_blink_duration":



            mean_blink_duration,



        "blink_interval_mean":



            blink_interval_mean,



        "blink_interval_variance":



            blink_interval_variance,



        "mean_ear":



            mean_ear,



        "ear_variance":



            ear_variance,



        "mean_mouth_opening":



            mean_mouth_opening,



        "mouth_opening_variance":



            mouth_opening_variance,



        "mean_lip_distance":



            mean_lip_distance,



        "mouth_movement_velocity":



            mouth_movement_velocity,



        "mouth_movement_frequency":



            mouth_movement_frequency,



        "speech_activity_ratio":



            speech_ratio,



        "lip_activity_ratio":



            lip_activity_ratio,



        "lip_speech_consistency":



            float(consistency),



        "heart_rate_estimate":



            heart_rate,



        "pulse_consistency":



            pulse_consistency,



        "temporal_signal_quality":



            temporal_quality,



        "rppg_available":



            float(rppg_available),



    }



    return features



# ============================================================



# SPEECH ACTIVITY



# ============================================================



def calculate_speech_activity(



    audio,



    num_segments=16



):



    if len(audio) == 0:



        return np.zeros(



            num_segments,



            dtype=bool



        )



    segment_length = (



        len(audio)



        // num_segments



    )



    energies = []



    for i in range(



        num_segments



    ):



        start = (



            i



            * segment_length



        )



        end = (



            (i + 1)



            * segment_length



            if i < num_segments - 1



            else len(audio)



        )



        segment = audio[



            start:end



        ]



        if len(segment) == 0:



            energy = 0.0



        else:



            energy = float(



                np.sqrt(



                    np.mean(



                        segment ** 2



                    )



                )



            )



        energies.append(



            energy



        )



    energies = np.array(



        energies



    )



    threshold = (



        0.20



        * np.max(energies)



    )



    return (



        energies >= threshold



    )



# ============================================================



# rPPG



# ============================================================



def extract_rppg(



    video_path



):



    try:



        cap = cv2.VideoCapture(



            video_path



        )



        if not cap.isOpened():



            return (



                0.0,



                0.0,



                0.0,



                0



            )



        fps = cap.get(



            cv2.CAP_PROP_FPS



        )



        if fps <= 0:



            fps = 25.0



        max_frames = int(



            fps * 6



        )



        green = []



        red = []



        blue = []



        count = 0



        while count < max_frames:



            success, frame = (



                cap.read()



            )



            if not success:



                break



            h, w = frame.shape[:2]



            # Central facial skin ROI



            # approximating forehead/cheek area.



            x1 = int(



                0.30 * w



            )



            x2 = int(



                0.70 * w



            )



            y1 = int(



                0.20 * h



            )



            y2 = int(



                0.65 * h



            )



            roi = frame[



                y1:y2,



                x1:x2



            ]



            if roi.size == 0:



                count += 1



                continue



            b, g, r = cv2.mean(



                roi



            )[:3]



            blue.append(b)



            green.append(g)



            red.append(r)



            count += 1



        cap.release()



        if len(green) < 60:



            return (



                0.0,



                0.0,



                0.0,



                0



            )



        green = np.array(



            green,



            dtype=float



        )



        red = np.array(



            red,



            dtype=float



        )



        blue = np.array(



            blue,



            dtype=float



        )



        # Normalize channels



        red = (



            red



            / (



                np.mean(red)



                + 1e-8



            )



        )



        green = (



            green



            / (



                np.mean(green)



                + 1e-8



            )



        )



        blue = (



            blue



            / (



                np.mean(blue)



                + 1e-8



            )



        )



        # CHROM-style signal



        x_signal = (



            3.0 * red



            - 2.0 * green



        )



        y_signal = (



            1.5 * red



            + green



            - 1.5 * blue



        )



        alpha = (



            np.std(x_signal)



            / (



                np.std(y_signal)



                + 1e-8



            )



        )



        pulse = (



            x_signal



            - alpha * y_signal



        )



        # Band-pass filter



        from scipy.signal import (



            butter,



            filtfilt,



            periodogram



        )



        low = 0.7



        high = 4.0



        nyquist = (



            fps / 2.0



        )



        if high >= nyquist:



            high = (



                nyquist



                * 0.9



            )



        if low >= high:



            return (



                0.0,



                0.0,



                0.0,



                0



            )



        b_filter, a_filter = butter(



            3,



            [



                low / nyquist,



                high / nyquist



            ],



            btype="band"



        )



        filtered = filtfilt(



            b_filter,



            a_filter,



            pulse



        )



        frequencies, power = (



            periodogram(



                filtered,



                fs=fps



            )



        )



        valid = (



            (frequencies >= low)



            & (frequencies <= high)



        )



        if not np.any(valid):



            return (



                0.0,



                0.0,



                0.0,



                0



            )



        valid_freq = frequencies[



            valid



        ]



        valid_power = power[



            valid



        ]



        peak_index = int(



            np.argmax(



                valid_power



            )



        )



        heart_rate = float(



            valid_freq[



                peak_index



            ]



            * 60.0



        )



        total_power = (



            np.sum(



                valid_power



            )



            + 1e-8



        )



        pulse_consistency = float(



            np.max(



                valid_power



            )



            / total_power



        )



        temporal_quality = float(



            np.std(filtered)



            / (



                np.mean(



                    np.abs(filtered)



                )



                + 1e-8



            )



        )



        return (



            heart_rate,



            pulse_consistency,



            temporal_quality,



            1



        )



    except Exception:



        return (



            0.0,



            0.0,



            0.0,



            0



        )



# ============================================================



# BEHAVIOR NORMALIZATION



# ============================================================



def prepare_behavior_tensor(



    features,



    columns,



    medians,



    means,



    stds



):



    values = []



    for column in columns:



        value = features.get(



            column,



            np.nan



        )



        try:



            value = float(



                value



            )



        except Exception:



            value = np.nan



        if np.isnan(value):



            value = float(



                medians[column]



            )



        normalized = (



            value



            - means[column]



        ) / (



            stds[column]



            + 1e-8



        )



        values.append(



            normalized



        )



    tensor = torch.tensor(



        values,



        dtype=torch.float32,



        device=DEVICE



    )



    return tensor.unsqueeze(0)



# ============================================================



# FUSION PREDICTION



# ============================================================



def predict(



    visual_embedding,



    audio_embedding,



    behavior_tensor,



    fusion_model



):



    visual = (



        visual_embedding



        .unsqueeze(0)



        .to(DEVICE)



    )



    audio = (



        audio_embedding



        .unsqueeze(0)



        .to(DEVICE)



    )



    with torch.no_grad():



        output = fusion_model(



            visual,



            audio,



            behavior_tensor



        )



    if isinstance(



        output,



        dict



    ):



        logits = output[



            "logits"



        ]



    elif isinstance(



        output,



        tuple



    ):



        logits = output[0]



    else:



        logits = output



    probabilities = torch.softmax(



        logits,



        dim=1



    )[0]



    real_probability = float(



        probabilities[0].item()



    )



    fake_probability = float(



        probabilities[1].item()



    )



    if fake_probability >= real_probability:



        prediction = "FAKE"



        confidence = fake_probability



    else:



        prediction = "REAL"



        confidence = real_probability



    return (



        prediction,



        real_probability,



        fake_probability,



        confidence



    )



# ============================================================



# EVIDENCE REASONING



# ============================================================



def level(



    score



):



    if score < 0.33:



        return "LOW"



    elif score < 0.66:



        return "MEDIUM"



    else:



        return "HIGH"



def level(score):

    if score < 0.33:

        return "LOW"

    if score < 0.66:

        return "MEDIUM"

    return "HIGH"





def calculate_evidence(

    features,

    fake_probability,

    audio_fake_probability=None

):

    visual_score = float(np.clip(fake_probability, 0.0, 1.0))



    lip_score = float(np.clip(

        1.0 - float(features.get("lip_speech_consistency", 0.0)),

        0.0,

        1.0

    ))



    blink_score = 0.0

    if features.get("mean_ear", 1.0) < 0.18:

        blink_score += 0.4

    if features.get("blink_rate", 0.0) < 0.10:

        blink_score += 0.2

    if features.get("blink_rate", 0.0) > 1.0:

        blink_score += 0.3

    if features.get("blink_count", 1.0) == 0:

        blink_score += 0.2



    blink_score = float(np.clip(blink_score, 0.0, 1.0))



    if features.get("rppg_available", 0.0) > 0.5:

        rppg_score = float(np.clip(

            1.0 - float(features.get("pulse_consistency", 0.0)),

            0.0,

            1.0

        ))

    else:

        rppg_score = None



    if audio_fake_probability is not None:

        audio_score = float(

            np.clip(audio_fake_probability, 0.0, 1.0)

        )

        audio_level = level(audio_score)

    else:

        audio_score = None

        audio_level = "UNAVAILABLE"



    return {

        "Visual": (visual_score, level(visual_score)),

        "Audio": (audio_score, audio_level),

        "Lip-Sync": (lip_score, level(lip_score)),

        "Blink": (blink_score, level(blink_score)),

        "rPPG": (

            rppg_score,

            level(rppg_score) if rppg_score is not None else "UNAVAILABLE"

        )

    }









# UI HEADER



# ============================================================



st.markdown(



    '<div class="title">🔬 DEEPFAKE FORENSICS</div>',



    unsafe_allow_html=True



)



st.markdown(



    '<div class="subtitle">'



    'Explainable Multimodal Deepfake Detection System'



    '</div>',



    unsafe_allow_html=True



)



# ============================================================



# SIDEBAR



# ============================================================



with st.sidebar:



    st.markdown(



        "## 🧠 Pipeline"



    )



    st.write(



        "🎞️ ViT Visual Analysis"



    )



    st.write(



        "🎧 AST Audio Manipulation Analysis"



    )



    st.write(



        "👄 Lip-Sync Analysis"



    )



    st.write(



        "👁️ Blink Analysis"



    )



    st.write(



        "❤️ rPPG Analysis"



    )



    st.write(



        "🔗 Cross-Modal Fusion"



    )



    st.write(



        "🧠 Evidence Reasoning"



    )



    st.divider()



    st.write(



        f"Device: `{DEVICE}`"



    )



# ============================================================



# LOAD MODELS



# ============================================================



try:



    (

        vit_processor,

        vit_model,

        ast_processor,

        ast_model,

        ast_audio_classifier,

        mtcnn,

        fusion_model

    ) = load_models()



    (



        behavior_columns,



        behavior_medians,



        behavior_means,



        behavior_stds



    ) = load_behavior_statistics()



except Exception as e:



    st.error(



        "Model loading failed."



    )



    st.exception(



        e



    )



    st.stop()



# ============================================================



# UPLOAD



# ============================================================



st.markdown(



    "## 🎥 Upload Video"



)



uploaded_file = st.file_uploader(



    "Choose a video",



    type=[



        "mp4",



        "avi",



        "mov",



        "mkv"



    ]



)



if uploaded_file:



    st.video(



        uploaded_file



    )



# ============================================================



# ANALYZE



# ============================================================



analyze = st.button(



    "🔍 ANALYZE VIDEO",



    type="primary",



    use_container_width=True



)



if analyze:



    if uploaded_file is None:



        st.warning(



            "Please upload a video first."



        )



        st.stop()



    # ========================================================



    # TEMPORARY FILE



    # ========================================================



    suffix = os.path.splitext(



        uploaded_file.name



    )[1]



    with tempfile.NamedTemporaryFile(



        delete=False,



        suffix=suffix



    ) as temp_video:



        temp_video.write(



            uploaded_file.getbuffer()



        )



        video_path = (



            temp_video.name



        )



    try:



        progress = st.progress(



            0



        )



        status = st.empty()



        # ====================================================



        # FRAMES



        # ====================================================



        status.write(



            "🎞️ Extracting video frames..."



        )



        (



            frames,



            frame_indices,



            fps,



            duration



        ) = extract_frames(



            video_path



        )



        progress.progress(



            10



        )



        # ====================================================



        # FACES



        # ====================================================



        status.write(



            "🙂 Detecting faces..."



        )



        (



            face_frames,



            face_boxes



        ) = detect_faces(



            frames,



            mtcnn



        )



        fallback_count = getattr(

            detect_faces,

            "last_fallback_count",

            0

        )



        if fallback_count > 0:

            st.warning(

                f"Face detection fallback was used for "

                f"{fallback_count} of {len(frames)} sampled frames."

            )



        progress.progress(

            25

        )



        # ====================================================



        # VIT



        # ====================================================



        status.write(



            "🎞️ Running ViT visual analysis..."



        )



        (



            visual_embedding,



            frame_embeddings



        ) = extract_vit_embedding(



            face_frames,



            vit_processor,



            vit_model



        )



        progress.progress(



            40



        )



        # ====================================================



        # AUDIO



        # ====================================================



        status.write(



            "🎧 Extracting audio..."



        )



        audio_wav = os.path.join(

            tempfile.gettempdir(),

            f"deepfake_forensics_audio_{os.getpid()}_{id(uploaded_file)}.wav"

        )



        extract_audio(



            video_path,



            audio_wav



        )



        progress.progress(



            50



        )



        # ====================================================



        # AST



        # ====================================================



        status.write(



            "🎧 Running AST audio analysis..."



        )



        audio_embedding = (



            extract_ast_embedding(



                audio_wav,



                ast_processor,



                ast_model



            )



        )



        (

            audio_prediction,

            real_audio_probability,

            fake_audio_probability,

            audio_confidence

        ) = predict_audio_manipulation(

            audio_wav,

            ast_processor,

            ast_audio_classifier

        )



        progress.progress(



            60



        )



        # ====================================================



        # BEHAVIOR



        # ====================================================



        status.write(



            "👁️ Extracting behavioral evidence..."



        )



        behavior_features = (



            extract_behavior(



                face_frames,



                video_path,



                fps,



                audio_wav



            )



        )



        progress.progress(



            75



        )



        # ====================================================



        # NORMALIZE BEHAVIOR



        # ====================================================



        behavior_tensor = (



            prepare_behavior_tensor(



                behavior_features,



                behavior_columns,



                behavior_medians,



                behavior_means,



                behavior_stds



            )



        )



        # ====================================================



        # FUSION



        # ====================================================



        status.write(



            "🔗 Running cross-modal fusion..."



        )



        (



            prediction,



            real_probability,



            fake_probability,



            confidence



        ) = predict(



            visual_embedding,



            audio_embedding,



            behavior_tensor,



            fusion_model



        )



        progress.progress(



            90



        )



        # ====================================================



        # EVIDENCE



        # ====================================================



        evidence = calculate_evidence(

            behavior_features,

            fake_probability,

            fake_audio_probability

        )



        progress.progress(



            100



        )



        status.success(



            "✓ Analysis completed successfully."



        )



        # ====================================================



        # RESULT



        # ====================================================



        st.markdown(



            "## 🎯 Forensic Result"



        )



        if prediction == "FAKE":



            card_class = (



                "fake-card"



            )



            icon = "⚠️"



        else:



            card_class = (



                "real-card"



            )



            icon = "✓"



        # Native Streamlit result display.

        # HTML is intentionally not used here because raw HTML tags were

        # appearing in the rendered result card.

        if prediction == "FAKE":

            st.error(f"{icon} DEEPFAKE")

        else:

            st.success(f"{icon} REAL")



        st.metric(

            label="Model confidence",

            value=f"{confidence * 100:.2f}%"

        )



        # ====================================================



        # PROBABILITIES



        # ====================================================



        col1, col2 = st.columns(2)



        with col1:



            st.metric(



                "P(REAL)",



                f"{real_probability * 100:.2f}%"



            )



        with col2:



            st.metric(



                "P(FAKE)",



                f"{fake_probability * 100:.2f}%"



            )



        st.markdown("### 🔍 Branch Summary")



        branch_col1, branch_col2 = st.columns(2)



        with branch_col1:

            st.metric(

                "Audio Result",

                audio_prediction,

                f"{audio_confidence * 100:.2f}% confidence"

            )



        with branch_col2:

            st.metric(

                "Multimodal Result",

                "DEEPFAKE" if prediction == "FAKE" else "REAL",

                f"{confidence * 100:.2f}% confidence"

            )



        # ====================================================



        # EVIDENCE



        # ====================================================



        st.markdown(



            "## 🧩 Multimodal Evidence"



        )



        evidence_cols = st.columns(5)



        for column, (

            name,

            (

                score,

                evidence_level

            )

        ) in zip(

            evidence_cols,

            evidence.items()

        ):

            if score is None:

                score_text = "Unavailable"

            else:

                score_text = f"{score:.3f}"



            with column:

                st.markdown(f"### {name}")



                if evidence_level == "HIGH":

                    st.error(evidence_level)

                elif evidence_level == "MEDIUM":

                    st.warning(evidence_level)

                elif evidence_level == "LOW":

                    st.success(evidence_level)

                else:

                    st.info(evidence_level)



                st.caption(f"Score: {score_text}")



        # ====================================================



        # BEHAVIOR DETAILS



        # ====================================================



        st.markdown(



            "## 🔬 Behavioral Analysis"



        )



        c1, c2, c3, c4 = st.columns(4)



        c1.metric(



            "Blink Count",



            f"{behavior_features['blink_count']:.2f}"



        )



        c2.metric(



            "Blink Rate",



            f"{behavior_features['blink_rate']:.3f}"



        )



        c3.metric(



            "Lip-Speech Consistency",



            f"{behavior_features['lip_speech_consistency']:.3f}"



        )



        if behavior_features[



            "rppg_available"



        ] > 0.5:



            c4.metric(



                "Heart Rate",



                f"{behavior_features['heart_rate_estimate']:.1f} BPM"



            )



        else:



            c4.metric(



                "Heart Rate",



                "Unavailable"



            )



        # ====================================================



        # AUDIO



        # ====================================================



        st.markdown(

            "## 🎧 Audio Analysis"

        )



        audio_col1, audio_col2, audio_col3 = st.columns(3)



        with audio_col1:

            if audio_prediction == "FAKE AUDIO":

                st.error(f"⚠️ {audio_prediction}")

            else:

                st.success(f"✓ {audio_prediction}")



        with audio_col2:

            st.metric(

                "P(REAL AUDIO)",

                f"{real_audio_probability * 100:.2f}%"

            )



        with audio_col3:

            st.metric(

                "P(FAKE AUDIO)",

                f"{fake_audio_probability * 100:.2f}%"

            )



        st.caption(

            "Standalone audio analysis uses ast_audio_best.pth. "

            "The same AST backbone embedding is supplied to "

            "the cross-modal fusion model."

        )







        # SUSPICIOUS TIMESTAMPS



        # ====================================================



        st.markdown(



            "## ⏱️ Suspicious Timestamps"



        )



        frame_scores = torch.norm(



            frame_embeddings



            - visual_embedding.unsqueeze(0),



            dim=1



        ).detach().cpu().numpy()



        if np.max(



            frame_scores



        ) > 0:



            normalized_scores = (



                frame_scores



                / np.max(



                    frame_scores



                )



            )



        else:



            normalized_scores = (



                frame_scores



            )



        top_indices = np.argsort(



            normalized_scores



        )[



            -3:



        ][::-1]



        for index in top_indices:



            if len(frame_indices) == 0:



                continue



            actual_frame = (



                frame_indices[



                    min(



                        index,



                        len(frame_indices) - 1



                    )



                ]



            )



            timestamp = (



                actual_frame



                / fps



                if fps > 0



                else 0



            )



            minutes = int(



                timestamp // 60



            )



            seconds = (



                timestamp



                - minutes * 60



            )



            st.markdown(



                f"""



                <span class="timestamp">



                    {minutes:02d}:{seconds:04.1f}



                </span>



                """,



                unsafe_allow_html=True



            )



        st.caption(



            "These timestamps are approximate visual "



            "embedding-deviation indicators, not validated "



            "frame-level manipulation probabilities."



        )



        # ====================================================



        # HEATMAP



        # ====================================================



        st.markdown(



            "## 🔥 Visual Analysis"



        )



        # Create an approximate attention-style visualization



        # from frame-level face embedding deviation.



        heatmap_frame_index = int(



            top_indices[0]



        )



        selected_frame = (



            face_frames[



                heatmap_frame_index



            ]



        )



        selected_score = (



            normalized_scores[



                heatmap_frame_index



            ]



        )



        display_frame = selected_frame.copy()



        # Create simple spatial heatmap centered on face.



        # This is explicitly an evidence visualization,



        # not a trained manipulation localization map.



        h, w = display_frame.shape[:2]



        heat = np.zeros(



            (h, w),



            dtype=np.float32



        )



        center_x = w // 2



        center_y = h // 2



        yy, xx = np.mgrid[



            0:h,



            0:w



        ]



        sigma = min(



            h,



            w



        ) * 0.25



        heat = np.exp(



            -(



                (



                    xx - center_x



                ) ** 2



                +



                (



                    yy - center_y



                ) ** 2



            )



            / (



                2 * sigma ** 2



            )



        )



        heat = (



            heat



            * selected_score



        )



        heat_uint8 = (



            heat



            * 255



        ).astype(



            np.uint8



        )



        color_heatmap = cv2.applyColorMap(



            heat_uint8,



            cv2.COLORMAP_JET



        )



        overlay = cv2.addWeighted(



            cv2.cvtColor(



                display_frame,



                cv2.COLOR_RGB2BGR



            ),



            0.65,



            color_heatmap,



            0.35,



            0



        )



        overlay = cv2.cvtColor(



            overlay,



            cv2.COLOR_BGR2RGB



        )



        st.image(



            overlay,



            caption=(



                "Visual evidence visualization"



            ),



            use_container_width=True



        )



        st.caption(



            "The uploaded-video dashboard currently "



            "uses an embedding-deviation visualization. "



            "It should not be interpreted as pixel-level "



            "proof of manipulation."



        )



        # ====================================================



        # FORENSIC REASONING



        # ====================================================



        st.markdown(



            "## 🧠 Forensic Evidence Reasoning"



        )



        available_scores = []



        for name, (



            score,



            evidence_level



        ) in evidence.items():



            if score is not None:



                available_scores.append(



                    score



                )



        overall_evidence = (



            float(



                np.mean(



                    available_scores



                )



            )



            if available_scores



            else 0.0



        )



        reasoning_output = (



            "FAKE"



            if overall_evidence >= 0.50



            else "REAL"



        )



        st.write(



            f"**Visual manipulation:** "



            f"{evidence['Visual'][1]}"



        )



        st.write(



            f"**Audio manipulation:** "



            f"{evidence['Audio'][1]}"



        )



        st.write(



            f"**Lip synchronization:** "



            f"{evidence['Lip-Sync'][1]}"



        )



        st.write(



            f"**Blink inconsistency:** "



            f"{evidence['Blink'][1]}"



        )



        st.write(



            f"**rPPG anomaly:** "



            f"{evidence['rPPG'][1]}"



        )



        st.markdown(



            f"### Overall evidence reasoning: "



            f"**{reasoning_output}**"



        )



        st.caption(



            f"Evidence aggregation score: "



            f"{overall_evidence:.4f}"



        )



        # ====================================================



        # REPORT



        # ====================================================



        st.markdown(



            "## 📄 Forensic Report"



        )



        report = f"""



DEEPFAKE FORENSICS REPORT



\\\\=========================



Input video:



{uploaded_file.name}



Prediction:



{'DEEPFAKE' if prediction == 'FAKE' else 'REAL'}



P(REAL):



{real_probability:.6f}



P(FAKE):



{fake_probability:.6f}



AUDIO P(REAL):



{real_audio_probability:.6f}



AUDIO P(FAKE):



{fake_audio_probability:.6f}



Audio prediction:



{audio_prediction}



Audio confidence:



{audio_confidence * 100:.2f}%



Model confidence:



{confidence * 100:.2f}%



\\\\----------------------------------------



MULTIMODAL EVIDENCE



\\\\----------------------------------------



Visual:



{evidence['Visual'][1]}



Audio:



{evidence['Audio'][1]}



Lip-Sync:



{evidence['Lip-Sync'][1]}



Blink:



{evidence['Blink'][1]}



rPPG:



{evidence['rPPG'][1]}



\\\\----------------------------------------



BEHAVIOR



\\\\----------------------------------------



Blink count:



{behavior_features['blink_count']}



Blink rate:



{behavior_features['blink_rate']}



Mean EAR:



{behavior_features['mean_ear']}



Lip-speech consistency:



{behavior_features['lip_speech_consistency']}



Heart rate:



{behavior_features['heart_rate_estimate']}



Pulse consistency:



{behavior_features['pulse_consistency']}



Temporal signal quality:



{behavior_features['temporal_signal_quality']}



\\\\----------------------------------------



FORENSIC REASONING



\\\\----------------------------------------



Overall evidence score:



{overall_evidence:.6f}



Evidence reasoning:



{reasoning_output}



\\\\----------------------------------------



LIMITATIONS



\\\\----------------------------------------



Model confidence represents the trained classifier's



output probability and is not proof of authenticity.



Behavioral and rPPG evidence are supporting evidence.



Suspicious timestamps are approximate.



The heatmap is an evidence visualization and should



not be interpreted as pixel-level proof of manipulation.



The AST backbone embedding is used by multimodal fusion.



A separately trained AST audio classifier provides the



standalone REAL AUDIO / FAKE AUDIO probability.



"""



        st.download_button(



            "⬇️ Download Forensic Report",



            report,



            file_name="deepfake_forensic_report.txt",



            mime="text/plain",



            use_container_width=True



        )



    except Exception as e:



        st.error(



            "Analysis failed."



        )



        st.exception(



            e



        )



    finally:



        try:



            os.remove(



                video_path



            )



        except Exception:



            pass


