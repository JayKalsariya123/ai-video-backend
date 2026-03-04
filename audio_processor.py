import json
import os
import re
import subprocess

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "whisper-large-v3-turbo")
TEXT_ANALYSIS_MODEL = os.getenv(
    "TEXT_ANALYSIS_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
)

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Add it to ai-video-backend/.env")

groq_client = Groq(api_key=GROQ_API_KEY)

FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "you know",
    "actually",
    "basically",
    "literally",
    "kind of",
    "sort of",
}


def transcribe_audio(audio_path):
    """Transcribe audio with Groq Whisper large v3 turbo."""
    try:
        print("Starting transcription...")
        with open(audio_path, "rb") as audio_file:
            result = groq_client.audio.transcriptions.create(
                file=audio_file,
                model=TRANSCRIPTION_MODEL,
                response_format="verbose_json",
                temperature=0,
            )
        transcript = (getattr(result, "text", None) or "").strip()
        print("Transcription successful")
        return transcript
    except Exception as exc:
        print(f"Error during transcription: {exc}")
        raise


def analyze_text(question, answer):
    """Analyze a transcribed interview answer and return structured JSON."""
    try:
        print("Starting text analysis...")

        if not answer or len(answer.strip()) < 2:
            return {
                "score": 0,
                "feedback": "No meaningful verbal answer was detected.",
                "strengths": [],
                "improvements": [
                    "Answer the question directly with one clear point and one concrete example."
                ],
            }

        prompt = f"""
You are a strict senior interviewer scoring a spoken interview answer.
The transcript may contain minor ASR mistakes. Do not invent missing content.

Scoring dimensions:
1) Relevance to the question
2) Clarity and structure
3) Technical depth or practical reasoning
4) Professional communication quality

Rules:
- If the answer is off-topic, empty, or mostly filler: score = 0.
- Otherwise score from 1 to 10.
- Keep feedback concise and actionable.
- Return ONLY valid JSON.

Required JSON shape:
{{
  "score": <integer 0-10>,
  "feedback": "<12-30 words>",
  "strengths": ["<short point>", "<short point>"],
  "improvements": ["<actionable suggestion>", "<actionable suggestion>"]
}}

Question: {question}
Answer transcript: {answer}
"""

        response = groq_client.chat.completions.create(
            model=TEXT_ANALYSIS_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        analysis_json = json.loads(content)
        score = int(analysis_json.get("score", 0))
        score = min(max(score, 0), 10)

        feedback = str(analysis_json.get("feedback", "No feedback provided.")).strip()
        strengths = analysis_json.get("strengths", [])
        improvements = analysis_json.get("improvements", [])

        if not isinstance(strengths, list):
            strengths = []
        if not isinstance(improvements, list):
            improvements = []

        return {
            "score": score,
            "feedback": feedback,
            "strengths": [str(item) for item in strengths][:3],
            "improvements": [str(item) for item in improvements][:3],
        }
    except Exception as exc:
        print(f"Error during text analysis: {exc}")
        return {
            "score": 0,
            "feedback": "Unable to analyze this answer.",
            "strengths": [],
            "improvements": ["Retry with clearer audio and a more direct answer."],
        }


def extract_speech_metrics(transcript):
    tokens = re.findall(r"[A-Za-z']+", transcript.lower())
    word_count = len(tokens)
    filler_count = 0
    transcript_lower = transcript.lower()

    for filler in FILLER_WORDS:
        if " " in filler:
            filler_count += transcript_lower.count(filler)
        else:
            filler_count += tokens.count(filler)

    filler_ratio = (filler_count / word_count) if word_count else 0.0
    return {
        "wordCount": word_count,
        "fillerWordCount": filler_count,
        "fillerWordRatio": round(filler_ratio, 3),
    }


def process_audio(video_path, question_text):
    """Process video audio and return transcript + quality analysis."""
    audio_path = None
    try:
        print("Processing audio...")

        audio_dir = os.path.join("uploads", "audio")
        os.makedirs(audio_dir, exist_ok=True)

        audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".wav"
        audio_path = os.path.join(audio_dir, audio_filename)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                audio_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        transcript = transcribe_audio(audio_path)
        analysis_result = analyze_text(question_text, transcript)

        score = float(analysis_result["score"])
        speech_metrics = extract_speech_metrics(transcript)

        word_count = speech_metrics["wordCount"]
        penalty = 1.0
        if word_count < 5:
            penalty = 0.3
        elif word_count < 15:
            penalty = 0.6
        elif word_count < 30:
            penalty = 0.8

        adjusted_score = min(max(score * penalty, 0), 10)

        return {
            "transcript": transcript,
            "score": round(adjusted_score, 1),
            "feedback": analysis_result["feedback"],
            "audioAnalysis": {
                "strengths": analysis_result["strengths"],
                "improvements": analysis_result["improvements"],
            },
            "speechMetrics": speech_metrics,
        }
    except Exception as exc:
        print(f"Error during audio processing: {exc}")
        raise
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
