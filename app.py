from flask import Flask, request, jsonify
import os
import threading
import time
import uuid
from queue import Queue, Empty
from werkzeug.utils import secure_filename
from video_processor import process_video
from audio_processor import process_audio
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=os.getenv("FRONTEND_ORIGIN", "*"))
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "25"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))

job_queue = Queue(maxsize=MAX_QUEUE_SIZE)
jobs = {}
jobs_lock = threading.Lock()


def _now():
    return time.time()


def _cleanup_old_jobs():
    cutoff = _now() - JOB_TTL_SECONDS
    with jobs_lock:
        stale_ids = [job_id for job_id, job in jobs.items() if job.get("createdAt", 0) < cutoff]
        for job_id in stale_ids:
            jobs.pop(job_id, None)


def _queue_position(job_id):
    if job_queue.empty():
        return 0
    queued_ids = [item.get("jobId") for item in list(job_queue.queue)]
    try:
        return queued_ids.index(job_id) + 1
    except ValueError:
        return 0


def _process_file(file_path, question_text):
    print("Starting video processing")
    video_result = process_video(file_path)
    print("Finished video processing")

    print("Starting audio processing")
    audio_result = process_audio(file_path, question_text)
    print("Finished audio processing")

    audio_score = float(audio_result.get("score", 0))
    video_score = float(video_result.get("score", 0))
    total_score = (0.65 * audio_score) + (0.35 * video_score)
    weighted_score = min(max(total_score, 0), 10)

    return {
        "question": question_text,
        "finalScore": round(weighted_score, 2),
        "audioScore": round(audio_score, 2),
        "videoScore": round(video_score, 2),
        "bodyLanguage": video_result["feedback"],
        "answerQuality": audio_result["feedback"],
        "transcript": audio_result.get("transcript", ""),
        "speechMetrics": audio_result.get("speechMetrics", {}),
        "audioAnalysis": audio_result.get("audioAnalysis", {}),
        "visualMetrics": {
            "confidence": video_result.get("metrics", {}).get("confidence", 0),
            "nervousness": video_result.get("metrics", {}).get("nervousness", 0),
            "eyeContact": video_result.get("metrics", {}).get("eye_contact", 0),
            "engagement": video_result.get("metrics", {}).get("engagement", 0),
            "stability": video_result.get("metrics", {}).get("stability", 0),
        },
    }


def _worker_loop():
    while True:
        try:
            job = job_queue.get(timeout=1)
        except Empty:
            _cleanup_old_jobs()
            continue

        job_id = job["jobId"]
        file_path = job["filePath"]
        question_text = job["questionText"]

        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "processing"
                jobs[job_id]["startedAt"] = _now()

        try:
            result = _process_file(file_path, question_text)
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["finishedAt"] = _now()
                    jobs[job_id]["result"] = result
        except Exception as exc:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["finishedAt"] = _now()
                    jobs[job_id]["error"] = str(exc)
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            finally:
                job_queue.task_done()


worker_thread = threading.Thread(target=_worker_loop, daemon=True)
worker_thread.start()


@app.get("/health")
def health():
    with jobs_lock:
        queued = job_queue.qsize()
        processing = sum(1 for j in jobs.values() if j.get("status") == "processing")
        done_recent = sum(1 for j in jobs.values() if j.get("status") == "done")
    return jsonify(
        {
            "ok": True,
            "service": "video-api",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "queue": {
                "queued": queued,
                "processing": processing,
                "done": done_recent,
                "maxSize": MAX_QUEUE_SIZE,
            },
        }
    )


@app.post("/jobs")
def submit_job():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    question_text = request.form.get("questionText", "")

    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Invalid file type. Allowed types: mp4, avi, mov"}), 400

    if job_queue.full():
        return jsonify({"error": "Server busy. Queue is full, try again soon."}), 503

    job_id = uuid.uuid4().hex
    filename = f"{job_id}-{secure_filename(file.filename)}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    with jobs_lock:
        jobs[job_id] = {
            "jobId": job_id,
            "status": "queued",
            "createdAt": _now(),
            "question": question_text,
        }

    job_queue.put({"jobId": job_id, "filePath": file_path, "questionText": question_text})
    queue_position = _queue_position(job_id)

    return jsonify(
        {
            "jobId": job_id,
            "status": "queued",
            "queuePosition": queue_position,
            "message": "Queued on the NASA-grade interview computer. This may take a bit.",
        }
    ), 202


@app.get("/jobs/<job_id>")
def get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    payload = {
        "jobId": job["jobId"],
        "status": job.get("status", "queued"),
        "question": job.get("question", ""),
    }
    if job.get("status") == "queued":
        payload["queuePosition"] = _queue_position(job_id)
    if job.get("status") == "done":
        payload["result"] = job.get("result")
    if job.get("status") == "failed":
        payload["error"] = job.get("error", "Job failed")
    return jsonify(payload)


@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if a file was uploaded
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files['video']
    question_text = request.form.get('questionText', '')

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the file to the uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            response = _process_file(file_path, question_text)
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        finally:
            # Clean up: Delete the uploaded file after processing
            if os.path.exists(file_path):
                os.remove(file_path)

    else:
        return jsonify({"error": "Invalid file type. Allowed types: mp4, avi, mov"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
