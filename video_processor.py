import cv2
import numpy as np

mp_face_mesh = None
face_mesh = None

try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_face_mesh = mp.solutions.face_mesh
except Exception as exc:
    mp_face_mesh = None

if mp_face_mesh is not None:
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        refine_landmarks=True,
    )

LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS = [61, 291, 13, 14]
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def point_xy(landmarks, idx):
    p = landmarks.landmark[idx]
    return np.array([p.x, p.y], dtype=np.float32)


def euclidean(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(landmarks, eye_points):
    p1 = point_xy(landmarks, eye_points[0])
    p2 = point_xy(landmarks, eye_points[1])
    p3 = point_xy(landmarks, eye_points[2])
    p4 = point_xy(landmarks, eye_points[3])
    p5 = point_xy(landmarks, eye_points[4])
    p6 = point_xy(landmarks, eye_points[5])

    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def mouth_openness(landmarks):
    left = point_xy(landmarks, MOUTH_POINTS[0])
    right = point_xy(landmarks, MOUTH_POINTS[1])
    upper = point_xy(landmarks, MOUTH_POINTS[2])
    lower = point_xy(landmarks, MOUTH_POINTS[3])

    width = euclidean(left, right)
    height = euclidean(upper, lower)
    if width == 0:
        return 0.0
    return height / width


def head_pose_proxy(landmarks):
    left_cheek = point_xy(landmarks, LEFT_CHEEK)
    right_cheek = point_xy(landmarks, RIGHT_CHEEK)
    nose = point_xy(landmarks, NOSE_TIP)
    chin = point_xy(landmarks, CHIN)

    eye_center = (point_xy(landmarks, 33) + point_xy(landmarks, 263)) / 2.0
    face_center = (left_cheek + right_cheek) / 2.0

    face_width = euclidean(left_cheek, right_cheek) + 1e-6
    face_height = euclidean(eye_center, chin) + 1e-6

    yaw = abs((nose[0] - face_center[0]) / face_width)
    pitch_baseline = (nose[1] - eye_center[1]) / face_height
    pitch = abs(pitch_baseline - 0.56)
    return yaw, pitch, nose


def analyze_frame(frame):
    if face_mesh is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]

    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_POINTS)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_POINTS)
    ear = (left_ear + right_ear) / 2.0
    blink = 1.0 if ear < 0.21 else 0.0

    mouth_ratio = mouth_openness(landmarks)
    yaw, pitch, nose = head_pose_proxy(landmarks)

    eye_contact = clamp(1.0 - ((yaw * 1.5) + (pitch * 1.2)))
    mouth_control = clamp(1.0 - abs(mouth_ratio - 0.22) * 3.0)
    confidence = clamp((0.55 * eye_contact) + (0.30 * mouth_control) + (0.15 * (1.0 - blink)))
    nervousness = clamp((0.45 * blink) + (0.30 * (1.0 - eye_contact)) + (0.25 * (1.0 - mouth_control)))
    engagement = clamp((0.65 * eye_contact) + (0.35 * mouth_control))

    return {
        "confidence": confidence,
        "nervousness": nervousness,
        "eye_contact": eye_contact,
        "engagement": engagement,
        "blink": blink,
        "nose": nose,
    }


def calculate_stability(nose_points):
    if len(nose_points) < 2:
        return 0.6
    diffs = [np.linalg.norm(nose_points[i] - nose_points[i - 1]) for i in range(1, len(nose_points))]
    jitter = float(np.mean(diffs))
    return clamp(1.0 - (jitter * 18.0), 0.0, 1.0)


def calculate_score(metrics):
    score = (
        (metrics["confidence"] * 0.34)
        + (metrics["eye_contact"] * 0.30)
        + (metrics["engagement"] * 0.20)
        + (metrics["stability"] * 0.16)
        - (metrics["nervousness"] * 0.22)
    ) * 10
    return min(max(score, 0), 10)


def generate_detailed_feedback(metrics):
    feedback = []

    if metrics["eye_contact"] >= 0.75:
        feedback.append("Strong eye contact and camera focus created an engaged presence.")
    elif metrics["eye_contact"] >= 0.55:
        feedback.append("Eye contact was decent; keep your gaze on the camera a bit more consistently.")
    else:
        feedback.append("Eye contact dropped often. Practice answering while looking near the camera lens.")

    if metrics["stability"] >= 0.70:
        feedback.append("Head movement stayed stable and composed.")
    elif metrics["stability"] >= 0.45:
        feedback.append("Movement was natural but occasionally restless.")
    else:
        feedback.append("Frequent head movement suggested nervousness; slow your pace and pause between points.")

    if metrics["confidence"] >= 0.75:
        feedback.append("You projected confident and interview-ready body language.")
    elif metrics["confidence"] >= 0.55:
        feedback.append("You appeared reasonably confident with room to be more assertive.")
    else:
        feedback.append("Confidence signals were weak; rehearse key stories and opening statements.")

    if metrics["nervousness"] >= 0.65:
        feedback.append("Nervous cues were visible. Use breathing and shorter sentences to stay steady.")
    elif metrics["nervousness"] >= 0.45:
        feedback.append("Some nervous energy appeared, which is normal in interviews.")
    else:
        feedback.append("You looked calm and well-controlled throughout the response.")

    return " ".join(feedback)


def process_video(video_path):
    if face_mesh is None:
        return {
            "score": 5.0,
            "feedback": "Facial analysis is temporarily unavailable on this instance. Audio scoring still works.",
            "metrics": {
                "confidence": 0.5,
                "nervousness": 0.5,
                "eye_contact": 0.5,
                "engagement": 0.5,
                "stability": 0.5,
            },
        }

    cap = cv2.VideoCapture(video_path)
    frame_results = []
    nose_points = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 4 == 0:
            analysis = analyze_frame(frame)
            if analysis:
                frame_results.append(analysis)
                nose_points.append(analysis["nose"])
        frame_count += 1

    cap.release()

    if not frame_results:
        return {
            "score": 0,
            "feedback": "No face detected. Ensure your face is visible with stable lighting.",
            "metrics": {
                "confidence": 0,
                "nervousness": 1,
                "eye_contact": 0,
                "engagement": 0,
                "stability": 0,
            },
        }

    overall_metrics = {
        "confidence": float(np.mean([r["confidence"] for r in frame_results])),
        "nervousness": float(np.mean([r["nervousness"] for r in frame_results])),
        "eye_contact": float(np.mean([r["eye_contact"] for r in frame_results])),
        "engagement": float(np.mean([r["engagement"] for r in frame_results])),
        "stability": float(calculate_stability(nose_points)),
    }

    score = calculate_score(overall_metrics)
    feedback = generate_detailed_feedback(overall_metrics)

    return {
        "score": round(score, 1),
        "feedback": feedback,
        "metrics": {k: round(v, 3) for k, v in overall_metrics.items()},
    }
