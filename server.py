"""Tiny Clipper local web app. Run `python server.py`, then open http://127.0.0.1:8765.

Listens on 127.0.0.1 only: keys live in settings.json next to this file and the
API has no login, so it must never be reachable from another machine.
"""
import json
import logging
import logging.handlers
import math
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import httpx

import clipper
import social

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.path.join(ROOT, "library")
SETTINGS_PATH = os.path.join(ROOT, "settings.json")
PORT = int(os.environ.get("TINY_CLIPPER_PORT", "8765"))
SERVICES = ("groq", "gemini")

os.makedirs(LIBRARY, exist_ok=True)
LOG_PATH = os.path.join(ROOT, "tiny-clipper.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.StreamHandler(),
    logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")])
logging.getLogger("httpx").setLevel(logging.WARNING)  # our own lines already say what was called
# Windows asyncio logs a traceback each time the browser drops a connection: noise, not an error.
logging.getLogger("asyncio").addFilter(
    lambda r: not (r.exc_info and isinstance(r.exc_info[1], ConnectionResetError)))
log = logging.getLogger("tinyclipper")
app = FastAPI(title="Tiny Clipper")
# One video at a time: the free tiers limit requests per minute, so parallel
# jobs would only trip each other's rate limits.
worker = ThreadPoolExecutor(max_workers=1)
# Clips of one job made at once: most of a clip's time is waiting on Gemini (framing,
# thumbnail), and NVENC takes several encodes. ponytail: fixed 3, raise with a faster GPU/CPU.
CLIP_WORKERS = int(os.environ.get("CLIP_WORKERS", 3))
lock = threading.RLock()  # re-entrant: read-modify-write helpers call write_job inside it


# --- settings ---------------------------------------------------------------

def replace_file(tmp, path):
    """os.replace that rides out the brief locks Windows allows other programs
    (antivirus, search indexer) to hold on a file."""
    for attempt in range(40):
        try:
            return os.replace(tmp, path)
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    return saved if isinstance(saved, dict) else {}


def save_settings(changes):
    """Merge changes into settings.json (a None value removes the entry)."""
    with lock:
        saved = load_settings()
        saved.update(changes)
        saved = {k: v for k, v in saved.items() if v is not None}
        with open(SETTINGS_PATH + ".tmp", "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2)
        replace_file(SETTINGS_PATH + ".tmp", SETTINGS_PATH)


def load_keys():
    """Each service's keys, main one first, then backups (settings.json keeps a
    single key as a plain string, several as a list)."""
    saved = load_settings()
    out = {}
    for s in SERVICES:
        keys = clipper.key_list(saved.get(s)) or clipper.key_list(os.environ.get(f"{s.upper()}_API_KEY", ""))
        out[s] = list(dict.fromkeys(keys))
    # Gemini keys whose Cloud project has Vertex AI on: those spend Google Cloud credit.
    out["vertex"] = [k for k in clipper.key_list(saved.get("vertex")) if k in out["gemini"]]
    return out


def provider_mode():
    """hybrid_auto (every key, Cloud credit first), free_local (free tiers only: no
    Cloud credit, no paid sign-in) or gcp_native (Google Cloud only). An older
    settings.json with "cloud_only" on reads as gcp_native."""
    saved = load_settings()
    mode = saved.get("provider_mode") or ("gcp_native" if saved.get("cloud_only") else "hybrid_auto")
    return mode if mode in clipper.PROVIDER_MODES else "hybrid_auto"


def cloud_only():
    return provider_mode() == "gcp_native"


def use_keys():
    """The keys the pipeline may use, plus "mode" for what has no key (Cloud
    sign-in, edge-tts). Keys a mode leaves out stay saved for later, uncalled."""
    keys, mode = load_keys(), provider_mode()
    if mode == "gcp_native":
        return {"gemini": [], "groq": [], "vertex": keys["vertex"], "mode": mode}
    if mode == "free_local":  # a key that works on Google Cloud has billing on: on AI Studio it'd be paid
        return {"gemini": [k for k in keys["gemini"] if k not in keys["vertex"]], "groq": keys["groq"],
                "vertex": [], "mode": mode}
    return {**keys, "mode": mode}


def gemini_keys(keys):
    """What the Gemini helpers take: AI Studio keys plus the Google Cloud ones."""
    return {"gemini": keys["gemini"], "vertex": keys["vertex"]}


def vertex_ok(key):
    """(works, message) for one key on Google Cloud (Vertex AI): a tiny request,
    a fraction of a cent."""
    try:
        resp = httpx.post(f"{clipper.VERTEX_BASE}/{clipper.VERTEX_MODELS[-1]}:generateContent", timeout=60,
                          headers={"x-goog-api-key": key},
                          json={"contents": [{"role": "user", "parts": [{"text": "Reply with: ok"}]}],
                                "generationConfig": {"maxOutputTokens": 5}})
    except httpx.HTTPError as e:
        return False, f"Couldn't reach Google Cloud ({e})"
    if resp.status_code == 200:
        return True, ""
    try:
        return False, resp.json()["error"]["message"][:400]
    except (ValueError, KeyError, TypeError):
        return False, f"Google Cloud answered {resp.status_code}"


def check_vertex(keys):
    """Try each Gemini key on Vertex AI and save the ones that work.
    Returns [{"key", "ok", "message"}]."""
    results = [{"key": masked(k), "raw": k, **dict(zip(("ok", "message"), vertex_ok(k)))} for k in keys["gemini"]]
    save_settings({"vertex": [r.pop("raw") for r in results if r["ok"]]})
    for r in results:
        r.pop("raw", None)
    return results


def save_keys(keys):
    save_settings(keys)


def masked(key):
    return f"••••{key[-4:]}" if key else None


class KeyBody(BaseModel):
    key: str


@app.get("/api/keys")
def get_keys():
    keys, signin = load_keys(), clipper.google_signin()  # signin: Google sign-in on this computer, for Speech-to-Text
    return {**{s: [masked(k) for k in keys[s]] for s in SERVICES},
            "vertex": [masked(k) for k in keys["vertex"]], "mode": provider_mode(),
            "signin": signin and (signin[1] or "your default project")}


class ModeBody(BaseModel):
    mode: str


@app.post("/api/keys/mode")
def set_mode(body: ModeBody):
    """Switch the provider stack; keys a mode doesn't use are kept, unused."""
    if body.mode not in clipper.PROVIDER_MODES:
        raise HTTPException(400, "Unknown mode.")
    keys = load_keys()
    if body.mode == "gcp_native" and not keys["vertex"]:
        raise HTTPException(400, "No key works on Google Cloud yet. Add one, or click Check again.")
    if body.mode == "free_local" and not (keys["groq"] or set(keys["gemini"]) - set(keys["vertex"])):
        raise HTTPException(400, "100% free needs a free key: a Gemini key from a Google Cloud project without "
                                 "billing, or a Groq key. Your Gemini key spends Google Cloud credit.")
    save_settings({"provider_mode": body.mode, "cloud_only": None})
    return {"ok": True, "mode": body.mode}


@app.post("/api/keys/vertex/check")
def vertex_check():
    """Does any Gemini key reach Google Cloud (Vertex AI), to spend Cloud credit?"""
    return {"results": check_vertex(load_keys())}


@app.post("/api/keys/{service}")
def add_key(service: str, body: KeyBody):
    """Test a key and add it: the first becomes the main key, later ones backups."""
    if service not in SERVICES:
        raise HTTPException(404, "Unknown service")
    key = body.key.strip()
    ok, message = clipper.check_key(service, key) if key else (False, "Paste a key first.")
    if not ok and key and service == "gemini":
        # A key limited to Vertex AI (Google Cloud) is fine: that's where it gets used.
        cloud_works, cloud_message = vertex_ok(key)
        ok, message = (True, "Key works on Google Cloud.") if cloud_works else             (False, f"Not a free Gemini key, and Google Cloud said: {cloud_message}")
    keys = load_keys()
    if ok and key in keys[service]:
        return {"ok": False, "message": "That key is already saved."}
    if ok:
        keys[service].append(key)
        save_keys({service: keys[service]})
        if service == "gemini":
            check_vertex(load_keys())  # does it also spend Google Cloud credit?
    return {"ok": ok, "message": message, "keys": [masked(k) for k in keys[service]]}


@app.delete("/api/keys/{service}/{index}")
def remove_key(service: str, index: int):
    keys = load_keys()
    if service not in SERVICES or not 0 <= index < len(keys[service]):
        raise HTTPException(404, "Not found")
    keys[service].pop(index)
    save_keys({service: keys[service]})
    return {"ok": True, "keys": [masked(k) for k in keys[service]]}


# --- jobs -------------------------------------------------------------------
# Each job is a folder in library/ holding job.json and its clips, so the
# library survives restarts and is just files the user can open.

def unique_job_folder(name, current=None):
    clean = clipper.safe_filename(name) or "Video"
    folder = os.path.join(LIBRARY, clean)
    if current and os.path.abspath(folder) == os.path.abspath(current):
        return folder
    if not os.path.exists(folder):
        return folder
    i = 2
    while os.path.exists(os.path.join(LIBRARY, f"{clean} ({i})")):
        candidate = os.path.join(LIBRARY, f"{clean} ({i})")
        if current and os.path.abspath(candidate) == os.path.abspath(current):
            return candidate
        i += 1
    return os.path.join(LIBRARY, f"{clean} ({i})")


def job_dir(job_id):
    if not job_id or ".." in job_id or "/" in job_id or "\\" in job_id:
        raise HTTPException(404, "Not found")
    path = os.path.join(LIBRARY, job_id)
    if not os.path.isdir(path):
        raise HTTPException(404, "Not found")
    return path


def read_job(job_id):
    # Under the lock: on Windows a file that's open for reading can't be
    # replaced, so a read racing write_job used to fail the save (and the job).
    path = os.path.join(job_dir(job_id), "job.json")
    try:
        with lock, open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        raise HTTPException(404, "Not found")


def write_job(job):
    path = os.path.join(LIBRARY, job["id"], "job.json")
    with lock:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2, ensure_ascii=False)
        replace_file(path + ".tmp", path)


NO_VOICE = "Some hooks have no AI voice: every voice service failed (see the log)."


def clean_voice(path, text, groq_key):
    """Cut anything spoken before the hook line, for voices recorded by Puter's
    Gemini voice in older versions (it read Puter's "Say the following text aloud."
    wrapper). Each voice is checked once, then marked ".clean"; voices made here
    are ".clean" from the start and cost no Groq at all."""
    if ".clean." in os.path.basename(path):
        return path
    try:
        path = clipper.trim_spoken_prefix(path, text, groq_key)
    except Exception as e:  # never fail the job over it
        log.warning("Couldn't check the hook voice: %s", e)
        return path
    base, ext = os.path.splitext(path)
    checked = f"{base}.clean{ext}"
    replace_file(path, checked)
    return checked


def server_voices(folder, wanted, keys):
    """Hook voices, all made here (see clipper.speak): {clip index: path} for the
    ones that worked. Marked ".clean": only the hook text was sent."""
    def speak(w):
        try:
            return clipper.speak(w["text"], keys, os.path.join(folder, f"voice_{w['i']:02d}.clean"))
        except clipper.ClipperError as e:
            log.warning("No voice for clip %d: %s", w["i"], e)

    with ThreadPoolExecutor(max_workers=8) as pool:  # ~3 s each; all at once instead of one by one
        return {w["i"]: path for w, path in zip(wanted, pool.map(speak, wanted)) if path}


def save_words(job_id, words, **extra):
    """Save the transcript's words, keeping the segments and language already in it."""
    path = os.path.join(LIBRARY, job_id, "transcript.json")
    data = load_transcript(os.path.dirname(path)) or {}
    data.update(extra, words=words)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    replace_file(path + ".tmp", path)


def load_transcript(folder):
    try:
        with open(os.path.join(folder, "transcript.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_words(folder):
    return load_transcript(folder)["words"]


def saved_voice(folder, i):
    return next((os.path.join(folder, f) for f in os.listdir(folder)
                 if f.startswith(f"voice_{i:02d}.") and not f.endswith(".err")), None)


def source_of(folder):
    return next((os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("source.")), None)


def run_job(job, source_url, source_path):
    folder = os.path.join(LIBRARY, job["id"])

    saved = {"at": 0.0, "message": None}

    def progress(message, fraction=None):
        job.update(message=message, progress=fraction)
        # Downloads report many times a second; saving twice a second is plenty.
        if message != saved["message"] or time.time() - saved["at"] > 0.5:
            saved.update(at=time.time(), message=message)
            try:
                write_job(job)
            except OSError as e:  # only the progress display would miss an update
                log.warning("Couldn't save progress: %s", e)

    def stage(step, message):
        log.info("job %s: %s", job["id"], step)
        job.update(status="running", step=step)
        progress(message)

    # Every finished step is kept on disk (video, transcript, picks, clips), so
    # Retry carries on from where a failure stopped instead of starting over.
    try:
        keys = use_keys()
        # A link's download is only reused once it finished (job has a duration);
        # half-done ones leave source.* pieces that yt-dlp resumes itself.
        if not source_path and (job.get("duration") or not job.get("url")):
            source_path = source_of(folder)
        if not source_path:
            stage("download", "Downloading the video")
            source_path, job["name"] = clipper.download(source_url or job.get("url"), folder, progress)
        if job.get("name"):
            target_folder = unique_job_folder(job["name"], current=folder)
            if os.path.abspath(folder) != os.path.abspath(target_folder):
                try:
                    os.rename(folder, target_folder)
                    folder = target_folder
                    source_path = os.path.join(folder, os.path.basename(source_path))
                    job["id"] = os.path.basename(target_folder)
                    write_job(job)
                except OSError as e:
                    log.warning("Couldn't rename folder to %s: %s", target_folder, e)
        duration, has_audio = clipper.media_info(source_path)
        if duration < clipper.MIN_CLIP:
            raise clipper.ClipperError("This video is too short to clip. Use one longer than a minute.")
        job["duration"] = round(duration)

        words, segments = [], []
        is_visual = not has_audio or bool(job.get("visual"))
        if not is_visual:
            transcript = load_transcript(folder)
            if transcript and "segments" in transcript:
                words, segments, job["language"] = transcript["words"], transcript["segments"], transcript.get("language")
            else:
                stage("transcribe", "Turning speech into text")
                try:
                    words, segments, job["language"] = clipper.transcribe(source_path, keys, progress)
                except Exception as e:
                    log.warning("Speech transcription error (%s); switching to visual moment detection", e)
                    words, segments = [], []
                if not words or len(words) < 5:
                    log.info("No speech or minimal spoken words found; switching to visual moment detection")
                    is_visual = True
                else:
                    save_words(job["id"], words, segments=segments, language=job["language"])

        job["visual"] = is_visual
        if is_visual:
            if not job.get("picked"):
                stage("pick_visual", "Finding visual hooks and action highlights")
                job["picked"], job["ai"] = clipper.pick_visual_moments(source_path, duration, job["count"], keys, progress)
                write_job(job)
        else:
            if not job.get("picked"):
                stage("pick", "Finding the best moments")  # then a second pass checks each clip's story
                sentences = clipper.build_sentences(words, segments)
                job["picked"], job["ai"] = clipper.pick_moments(sentences, duration, job["count"], keys, progress)
                write_job(job)
        if job.get("review") and not job.get("reviewed"):
            # Checkpoint: nothing is voiced or cut until a person (or the auto-approve
            # timer) okays the picks. Returning frees the worker for the next video.
            job.update(status=REVIEW, step="review", message="Waiting for you to review the picks",
                       progress=None, review_since=job.get("review_since") or time.time())
            log.info("job %s: waiting for review", job["id"])
            return
        made = {c.get("i") for c in job["clips"]}
        clips = [(i, c) for i, c in enumerate(job["picked"], 1) if i not in made and not c.get("dropped")]

        voices = {}
        if job.get("intro", True):
            wanted = [{"i": i, "text": c["voiceover"]} for i, c in clips if c["intro"] and c["voiceover"]]
            voices = {w["i"]: saved_voice(folder, w["i"]) for w in wanted if saved_voice(folder, w["i"])}
            wanted = [w for w in wanted if w["i"] not in voices]
            if wanted:
                stage("voice", "Recording the AI voices")
                voices.update(server_voices(folder, wanted, keys))
                if any(w["i"] not in voices for w in wanted):
                    job["warning"] = NO_VOICE

        stage("render", f"Cutting your clips (0 of {len(clips)} ready)")
        captions_on = job.get("captions", True) and not job.get("visual")

        def make_clip(i, picked):
            """One finished clip, made on a pool thread: it only touches its own dict and files,
            the job itself is updated by run_job's thread as clips come back."""
            clip = dict(picked, i=i)  # i names its files (clip_NN, voice_NN, thumb_NN) for good
            if not job.get("intro", True):
                clip["intro"] = None
            plan = None
            if job["layout"] != "fit":
                try:
                    plan = clipper.plan_parts(source_path, clip["parts"], gemini_keys(keys))
                except Exception as e:  # framing is a nicety: fall back to a centre crop
                    log.warning("Framing failed for clip %d: %s", i, e)
                clip["framing"] = clipper.describe_framing(plan)
            clip["layout"] = job["layout"]
            clip["file"] = clipper.render_clip(source_path, clip, words, folder, i, job["layout"],
                                               captions_on, voices.get(i), plan)
            clip["length"] = round(clipper.media_info(os.path.join(folder, clip["file"]))[0])
            try:
                clip["thumb"], clip["thumb_v"] = make_thumb(folder, source_path, clip, i, keys), int(time.time())
            except Exception as e:  # a thumbnail is a bonus: never fail the job over it
                clip["thumb_error"] = f"Couldn't make the thumbnail: {e}"
            return clip

        failed = []
        with ThreadPoolExecutor(max_workers=CLIP_WORKERS) as pool:
            running = {pool.submit(make_clip, i, picked): i for i, picked in clips}
            for k, done in enumerate(as_completed(running), 1):
                i = running[done]
                try:
                    clip = done.result()
                except Exception as e:  # one bad clip doesn't cost the others: Retry makes it later
                    log.error("job %s: clip %d failed", job["id"], i, exc_info=e)
                    failed.append(f"clip {i} ({e})")
                    continue
                finally:
                    progress(f"Cutting your clips ({k} of {len(clips)} ready)", k / len(clips))
                job["clips"].append(clip)
                job["clips"].sort(key=lambda c: c.get("i", 0))
                write_job(job)  # each clip shows up as soon as it's ready
        job["missing"] = len(failed)
        if failed and not job["clips"]:
            raise clipper.ClipperError("Couldn't cut any clip: " + "; ".join(failed))
        if failed:
            job["warning"] = f"Couldn't make {', '.join(failed)}. Click Retry to try again."
        job.update(status="done", step="done", message=f"{len(job['clips'])} clips ready", progress=1)
    except clipper.ClipperError as e:
        log.error("job %s failed at step %s: %s", job["id"], job.get("step"), e)
        job.update(status="error", error=str(e))
    except Exception as e:  # anything unexpected still ends the job cleanly
        log.exception("job %s crashed at step %s", job["id"], job.get("step"))
        job.update(status="error", error=f"Something went wrong: {e}")
    finally:
        # Every job keeps its original, transcript and voices: a finished one so
        # clips can be edited, a failed one so Retry picks up where it stopped.
        # Delete removes them.
        # ponytail: originals stay until deleted; add an auto-cleanup age if disks fill up.
        for name in os.listdir(folder):
            if name.startswith(".upload_") or name.endswith((".err", ".tmp")):  # leftovers of older versions
                try:
                    os.remove(os.path.join(folder, name))
                except OSError:  # still locked: leaving a stray file beats losing the job's status
                    pass
        write_job(job)


@app.post("/api/jobs")
def create_job(file: Optional[UploadFile] = File(None), url: str = Form(""),
               count: int = Form(5), layout: str = Form("auto"), captions: bool = Form(True),
               intro: bool = Form(True), review: bool = Form(False)):
    url = url.strip()
    keys = use_keys()
    if cloud_only() and not keys["vertex"]:
        raise HTTPException(400, "\"100% Google Cloud\" is on, but no key works on Google Cloud right now. "
                                 "Click Check again in Keys, or switch to Auto at the top.")
    if keys["mode"] == "free_local" and not (keys["gemini"] or keys["groq"]):
        raise HTTPException(400, "\"100% free\" is on, but you have no free key (your Gemini key spends Google "
                                 "Cloud credit). Add a Groq key in Keys, or switch to Auto at the top.")
    if not (keys["gemini"] or keys["groq"] or keys["vertex"]):
        raise HTTPException(400, "Add your Gemini key first (button: Keys).")
    if not (file and file.filename) and not url:
        raise HTTPException(400, "Choose a video file or paste a link.")
    if url and not url.startswith(("http://", "https://")):
        raise HTTPException(400, "That link doesn't look right. It should start with https://")
    if layout not in clipper.LAYOUTS or not 1 <= count <= 10:
        raise HTTPException(400, "Invalid options.")

    folder = None
    job_name = url or (file.filename if file else "")
    if file and file.filename:
        title = os.path.splitext(file.filename)[0]
        folder = unique_job_folder(title)
        job_id = os.path.basename(folder)
        job_name = title
    elif url:
        title = None
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True, "socket_timeout": 3}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title")
        except Exception:
            title = None
        if title:
            folder = unique_job_folder(title)
            job_name = title
        else:
            folder = unique_job_folder(uuid.uuid4().hex[:12])
        job_id = os.path.basename(folder)
    else:
        folder = unique_job_folder(uuid.uuid4().hex[:12])
        job_id = os.path.basename(folder)

    os.makedirs(folder, exist_ok=True)
    source_path = None
    if not url:
        ext = os.path.splitext(file.filename)[1].lower()[:8] or ".mp4"
        source_path = os.path.join(folder, "source" + ext)
        with open(source_path, "wb") as out:
            shutil.copyfileobj(file.file, out, 1024 * 1024)
    job = {"id": job_id, "name": job_name, "created": time.time(), "status": "queued",
           "step": "queued", "message": "Waiting for the previous video to finish", "progress": None,
           "count": count, "layout": layout, "captions": captions, "intro": intro, "review": review, "clips": [],
           "error": None, "warning": None, "url": url}
    write_job(job)
    worker.submit(run_job, job, url, source_path)
    return job


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    """Run a failed job (or the clips it missed) again from its last finished step."""
    job = read_job(job_id)
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "This video is still being processed.")
    if not (job.get("url") or source_of(job_dir(job_id))):
        raise HTTPException(409, "The original video is gone. Add it again.")
    job.update(status="queued", step="queued", message="Waiting for the previous video to finish",
               progress=None, error=None, warning=None)
    write_job(job)
    worker.submit(run_job, job, "", None)
    return job


@app.get("/api/jobs")
def list_jobs():
    jobs = []
    for job_id in os.listdir(LIBRARY):
        try:
            job = read_job(job_id)
        except HTTPException:
            continue
        folder = os.path.join(LIBRARY, job_id)
        job["editable"] = bool(source_of(folder)) and os.path.exists(os.path.join(folder, "transcript.json"))
        if job["status"] == REVIEW and auto_minutes():
            job["auto_approve_at"] = job.get("review_since", time.time()) + auto_minutes() * 60
        jobs.append(job)
    return sorted(jobs, key=lambda j: j["created"], reverse=True)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if read_job(job_id)["status"] in ("queued", "running"):
        raise HTTPException(409, "This video is still being processed.")
    shutil.rmtree(job_dir(job_id))
    return {"ok": True}


# --- review before cutting (human in the loop) ---------------------------------

REVIEW = "awaiting_hitl_review"


def auto_minutes():
    """Minutes a review waits before going ahead on its own (settings.json
    "hitl_auto_approve_minutes", default 30; 0 waits for a person)."""
    try:
        return max(0.0, float(load_settings().get("hitl_auto_approve_minutes", 30)))
    except (TypeError, ValueError):
        return 30.0


def job_sentences(job):
    """The sentences the picks' ids point into ([] for a visual job)."""
    transcript = load_transcript(job_dir(job["id"]))
    if job.get("visual") or not transcript:
        return []
    return clipper.build_sentences(transcript["words"], transcript.get("segments") or [])


def finish_review(job_id, edits, note=None):
    """Save the reviewed picks (edits None: an unattended review) and queue the
    job again: it carries on from the voices, as Retry would."""
    with lock:
        job = read_job(job_id)
        if job["status"] != REVIEW:
            raise HTTPException(409, "This video isn't waiting for a review.")
        try:
            picked = (clipper.auto_review(job["picked"]) if edits is None else
                      clipper.apply_review(job["picked"], edits, job_sentences(job), job["duration"]))
        except clipper.ClipperError as e:
            raise HTTPException(400, str(e))
        folder = job_dir(job_id)
        for i, c in enumerate(picked, 1):  # a previewed voice only counts for the line it read
            voice = saved_voice(folder, i)
            if voice and (c["dropped"] or c.get("voice_for") != c.get("voiceover")):
                os.remove(voice)
        job.update(picked=picked, reviewed=True, status="queued", step="queued", progress=None,
                   message="Waiting for the previous video to finish", warning=note)
        write_job(job)
    worker.submit(run_job, job, "", None)
    return job


@app.get("/api/jobs/{job_id}/review")
def review_data(job_id: str):
    job = read_job(job_id)
    if job["status"] != REVIEW:
        raise HTTPException(409, "This video isn't waiting for a review.")
    return {"clips": job["picked"], "sentences": job_sentences(job), "intro": job.get("intro", True),
            "keep_min": clipper.KEEP_MIN, "keep_max": clipper.KEEP_MAX, "max_parts": clipper.MAX_PARTS}


class ReviewBody(BaseModel):
    clips: list


@app.post("/api/jobs/{job_id}/review")
def approve_picks(job_id: str, body: ReviewBody):
    return finish_review(job_id, body.clips)


class VoiceBody(BaseModel):
    i: int
    text: str


@app.post("/api/jobs/{job_id}/review/voice")
def preview_voice(job_id: str, body: VoiceBody):
    """Record a clip's hook line now, to hear it before approving. It's the real
    voice: kept for the clip if the line doesn't change again."""
    job, folder, text = read_job(job_id), job_dir(job_id), body.text.strip()[:160]
    if job["status"] != REVIEW or not 1 <= body.i <= len(job["picked"]):
        raise HTTPException(409, "This video isn't waiting for a review.")
    if not text:
        raise HTTPException(400, "Write the hook line first.")
    old = saved_voice(folder, body.i)
    if old:
        os.remove(old)
    try:
        path = clipper.speak(text, use_keys(), os.path.join(folder, f"voice_{body.i:02d}.clean"))
    except clipper.ClipperError as e:
        raise HTTPException(502, f"Couldn't record the voice: {e}")
    with lock:
        job = read_job(job_id)
        job["picked"][body.i - 1]["voice_for"] = text
        write_job(job)
    return {"url": f"/library/{urllib.parse.quote(job_id)}/{urllib.parse.quote(os.path.basename(path))}"
                   f"?v={int(time.time())}"}


def auto_approve():
    """Reviews left alone past auto_minutes() go ahead with the clips scoring 85+."""
    while True:
        time.sleep(30)
        minutes = auto_minutes()
        if not minutes:
            continue
        for job in list_jobs():
            if job["status"] == REVIEW and time.time() - job.get("review_since", time.time()) > minutes * 60:
                try:
                    finish_review(job["id"], None, note=f"No review within {minutes:g} min: kept the clips "
                                                        f"scoring {clipper.REVIEW_KEEP_SCORE}+.")
                    log.info("job %s: review auto-approved", job["id"])
                except Exception:
                    log.exception("job %s: auto-approving the review failed", job["id"])


def mark_interrupted():
    """Jobs cut short by closing the app can't resume; say so instead of spinning forever."""
    for job in list_jobs():
        if job["status"] in ("queued", "running"):
            job.update(status="error", error="Stopped because Tiny Clipper was closed. Click Retry to carry on.")
            write_job(job)
            for name in os.listdir(os.path.join(LIBRARY, job["id"])):
                if name.endswith(".tmp"):
                    try:
                        os.remove(os.path.join(LIBRARY, job["id"], name))
                    except OSError:
                        pass
        for clip in job.get("clips", []):
            if clip.get("editing"):
                clip.update(editing=False, edit_status=None, edit_error="Stopped because Tiny Clipper was closed. Edit it again.")
                write_job(job)
            for post in (clip.get("posts") or {}).values():
                if post.get("status") == "posting":
                    post.update(status="error", message="Stopped because Tiny Clipper was closed. Post it again.")
                    write_job(job)


# --- posting to social platforms --------------------------------------------
# Settings hold each platform's app credentials and sign-in tokens; the page
# only ever sees whether a platform is connected, never the secrets.

PLATFORMS = ("instagram", "youtube", "tiktok")
REDIRECT = f"http://127.0.0.1:{PORT}/"  # YouTube sign-in comes back to the home page
poster = ThreadPoolExecutor(max_workers=2)
pending_state = {}  # platform -> random state of the sign-in in progress


def social_cfg(platform):
    return load_settings().get(platform) or {}


@app.get("/api/social")
def social_status():
    ig, yt, tt = social_cfg("instagram"), social_cfg("youtube"), social_cfg("tiktok")
    return {
        "instagram": {"connected": bool(ig.get("token")), "account": ig.get("username")},
        "youtube": {"configured": bool(yt.get("client_id")), "connected": bool(yt.get("refresh_token")),
                    "account": yt.get("channel")},
        "tiktok": {"configured": bool(tt.get("client_key")), "connected": bool(tt.get("refresh_token")),
                   "redirect_uri": tt.get("redirect_uri", "")},
    }


class SocialBody(BaseModel):
    token: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    url: str = ""


@app.post("/api/social/instagram")
def connect_instagram(body: SocialBody):
    token = body.token.strip()
    if not token:
        raise HTTPException(400, "Paste your Instagram access token first.")
    try:
        me = social.ig_check(token)
    except (social.SocialError, httpx.HTTPError) as e:
        return {"ok": False, "message": f"{e}. Generate a new token and paste it again."}
    save_settings({"instagram": {"token": token, **me, "refreshed": time.time()}})
    return {"ok": True, "message": f"Connected as @{me['username']}"}


@app.post("/api/social/youtube")
def start_youtube(body: SocialBody):
    """Save the OAuth client, then hand back Google's sign-in address."""
    cfg = social_cfg("youtube")
    if body.client_id.strip():
        cfg = {"client_id": body.client_id.strip(), "client_secret": body.client_secret.strip()}
    if not cfg.get("client_id") or not cfg.get("client_secret"):
        raise HTTPException(400, "Paste the client ID and client secret first.")
    save_settings({"youtube": cfg})
    pending_state["youtube"] = uuid.uuid4().hex
    return {"auth_url": social.yt_auth_url(cfg, REDIRECT, pending_state["youtube"])}


@app.post("/api/social/tiktok")
def start_tiktok(body: SocialBody):
    cfg = social_cfg("tiktok")
    if body.client_id.strip():
        cfg = {"client_key": body.client_id.strip(), "client_secret": body.client_secret.strip(),
               "redirect_uri": body.redirect_uri.strip()}
    if not all(cfg.get(k) for k in ("client_key", "client_secret", "redirect_uri")):
        raise HTTPException(400, "Fill in the client key, client secret and redirect URI first.")
    if not cfg["redirect_uri"].startswith("https://"):
        raise HTTPException(400, "TikTok needs a redirect URI that starts with https://")
    save_settings({"tiktok": cfg})
    pending_state["tiktok"] = uuid.uuid4().hex
    return {"auth_url": social.tt_auth_url(cfg, pending_state["tiktok"])}


@app.post("/api/social/tiktok/finish")
def finish_tiktok(body: SocialBody):
    try:
        cfg = social.tt_connect(social_cfg("tiktok"), body.url, pending_state.get("tiktok", ""))
    except (social.SocialError, httpx.HTTPError) as e:
        return {"ok": False, "message": str(e)}
    save_settings({"tiktok": cfg})
    return {"ok": True, "message": "TikTok connected"}


@app.delete("/api/social/{platform}")
def disconnect(platform: str):
    if platform not in PLATFORMS:
        raise HTTPException(404, "Unknown platform")
    save_settings({platform: None})
    return {"ok": True}


class PostBody(BaseModel):
    platforms: list[str]
    caption: str = ""


def set_post(job_id, n, platform, **post):
    with lock:
        job = read_job(job_id)
        job["clips"][n - 1].setdefault("posts", {})[platform] = post
        write_job(job)


def thumb_frames(source, clip):
    """JPEG frames for the thumbnail: the hook moment and the middle of the clip's parts."""
    parts = clip.get("parts") or [{"s": clip["start"], "e": clip["end"]}]
    times = ([(clip["intro"]["start"] + clip["intro"]["end"]) / 2] if clip.get("intro") else []) + \
            [(x["s"] + x["e"]) / 2 for x in parts[:2]]
    frames = []
    for t in times[:3]:
        proc = subprocess.run([clipper.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}",
                               "-i", source, "-frames:v", "1", "-vf", "scale=-2:720", "-q:v", "3",
                               "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True)
        if proc.stdout:
            frames.append(proc.stdout)
    return frames


def make_thumb(folder, source, clip, n, keys):
    if clip.get("file"):
        base = os.path.splitext(clip["file"])[0]
    else:
        base = clipper.safe_filename(clip.get("title") or "") or f"thumb_{n:02d}"
    name = f"{base}.jpg"
    clipper.make_thumbnail(thumb_frames(source, clip), clip["title"], clip.get("message", ""),
                           gemini_keys(keys), os.path.join(folder, name))
    return name


def thumb_job(job_id, n):
    folder = job_dir(job_id)
    try:
        keys = use_keys()
        clip = read_job(job_id)["clips"][n - 1]
        name = make_thumb(folder, source_of(folder), clip, clip.get("i", n), keys)
        update_clip(job_id, n, thumb=name, thumb_v=int(time.time()), thumb_status=None, thumb_error=None)
    except Exception as e:
        log.exception("job %s: thumbnail for clip %d failed", job_id, n)
        update_clip(job_id, n, thumb_status=None, thumb_error=f"Couldn't make the thumbnail: {e}"[:400])


@app.post("/api/jobs/{job_id}/clips/{n}/thumbnail")
def thumbnail_endpoint(job_id: str, n: int):
    job, clip, folder = editable_clip(job_id, n)
    if clip.get("thumb_status"):
        raise HTTPException(409, "The thumbnail is already being made.")
    update_clip(job_id, n, thumb_status="making", thumb_error=None)
    poster.submit(thumb_job, job_id, n)
    return {"ok": True}


def post_clip(job_id, n, platforms, caption):
    job = read_job(job_id)
    clip = job["clips"][n - 1]
    path = os.path.join(LIBRARY, job_id, clip["file"])
    title = clip["title"]
    for platform in platforms:
        try:
            cfg = social_cfg(platform)
            if platform == "instagram":
                cfg = social.ig_refresh(cfg)
                save_settings({"instagram": cfg})
                cover = os.path.join(LIBRARY, job_id, clip["thumb"]) if clip.get("thumb") else None
                set_post(job_id, n, platform, status="done", url=social.ig_post(cfg, path, caption, cover))
            elif platform == "youtube":
                set_post(job_id, n, platform, status="done", url=social.yt_post(cfg, path, title, caption),
                         message="Uploaded. It stays private until Google audits your project.")
            else:
                cfg = social.tt_fresh(cfg)
                save_settings({"tiktok": cfg})
                set_post(job_id, n, platform, status="done", message=social.tt_post(cfg, path))
        except (social.SocialError, httpx.HTTPError) as e:
            log.warning("job %s: posting clip %d to %s failed: %s", job_id, n, platform, e)
            set_post(job_id, n, platform, status="error", message=str(e))
        except Exception as e:  # never leave a post spinning
            log.exception("job %s: posting clip %d to %s crashed", job_id, n, platform)
            set_post(job_id, n, platform, status="error", message=f"Something went wrong: {e}")


@app.post("/api/jobs/{job_id}/clips/{n}/post")
def post_endpoint(job_id: str, n: int, body: PostBody):
    job = read_job(job_id)
    if not 1 <= n <= len(job["clips"]):
        raise HTTPException(404, "Not found")
    status = social_status()
    platforms = [p for p in dict.fromkeys(body.platforms) if p in PLATFORMS and status[p]["connected"]]
    if not platforms:
        raise HTTPException(400, "Pick at least one connected platform.")
    for p in platforms:
        set_post(job_id, n, p, status="posting")
    poster.submit(post_clip, job_id, n, platforms, body.caption.strip()[:2000])
    return {"ok": True}


# --- the editor ---------------------------------------------------------------
# The editor page plays a small preview copy of the original around the clip
# and saves an edit list; the final render always uses the original.

PREVIEW_MARGIN = 60  # seconds of original kept on each side of the edit, to trim into
FRAME_MODES = ("auto", "fill", "manual", "fit")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm")


def update_clip(job_id, n, **changes):
    with lock:
        job = read_job(job_id)
        job["clips"][n - 1].update(changes)
        write_job(job)


def editable_clip(job_id, n):
    """(job, clip, folder) for a clip the editor may work on, else an HTTP error."""
    job, folder = read_job(job_id), job_dir(job_id)
    if not 1 <= n <= len(job["clips"]):
        raise HTTPException(404, "Not found")
    if not source_of(folder) or not os.path.exists(os.path.join(folder, "transcript.json")):
        raise HTTPException(409, "This clip was made before editing existed, so its original video is gone. "
                                 "Make the video again to edit its clips.")
    return job, job["clips"][n - 1], folder


def clip_edit(job, clip):
    """The clip's saved edit, or the one it was made with."""
    if clip.get("edit"):
        return clip["edit"]
    edl = clipper.clip_edl(clip, clip.get("layout") or job.get("layout", "auto"), job.get("captions", True))
    edl["hook"]["on"] = bool(clip.get("intro")) and not clip.get("hook_off")
    return edl


@app.get("/api/jobs/{job_id}/clips/{n}/editor")
def editor_data(job_id: str, n: int):
    """Everything the editor page needs; makes the preview copy on first open."""
    job, clip, folder = editable_clip(job_id, n)
    src = source_of(folder)
    duration = clipper.media_info(src)[0]
    edl = clip_edit(job, clip)
    window = [round(max(0.0, min(x["s"] for x in edl["segments"]) - PREVIEW_MARGIN), 2),
              round(min(duration, max(x["e"] for x in edl["segments"]) + PREVIEW_MARGIN), 2)]
    proxy, meta_path = f"preview_{n:02d}.mp4", os.path.join(folder, f"preview_{n:02d}.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = {}
    if meta.get("window") != window or not os.path.exists(os.path.join(folder, proxy)):
        # Small and seek-friendly (a keyframe every 12 frames) so scrubbing feels instant.
        clipper.run_ffmpeg(["-ss", f"{window[0]:.3f}", "-t", f"{window[1] - window[0]:.3f}", "-i", src,
                            "-vf", "scale=-2:540", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                            "-g", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                            "-movflags", "+faststart", proxy], cwd=folder)
        plan = None
        try:
            plan = clipper.plan_framing(src, window[0], window[1], gemini_keys(use_keys()))
        except Exception as e:  # Auto framing then previews and renders centred
            log.warning("Framing failed for the editor: %s", e)
        meta = {"window": window, "plan": plan}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    return {
        "proxy": f"/library/{urllib.parse.quote(job_id)}/{proxy}?v={int(os.path.getmtime(os.path.join(folder, proxy)))}",
        "window": window, "duration": duration, "size": clipper.video_size(src), "plan": meta.get("plan"),
        "words": [w for w in load_words(folder) if window[0] <= w["s"] <= window[1]],
        "edit": edl, "has_hook": bool(clip.get("intro")), "editing": bool(clip.get("editing")),
        "sfx": [{"name": k, "dur": round(clipper.media_info(clipper.sfx_path(k))[0], 2)} for k in clipper.SFX],
    }


@app.get("/sfx/{name}")
def sfx(name: str):
    if name not in clipper.SFX:
        raise HTTPException(404, "Not found")
    return FileResponse(clipper.sfx_path(name), media_type="audio/wav")


@app.post("/api/jobs/{job_id}/sounds")
def upload_sound(job_id: str, file: UploadFile = File(...)):
    """A sound (music, effect) for the editor, kept in the job's sounds/ folder."""
    folder = job_dir(job_id)
    read_job(job_id)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, "Use an audio file (MP3, WAV, M4A, OGG…).")
    os.makedirs(os.path.join(folder, "sounds"), exist_ok=True)
    name = f"sounds/{uuid.uuid4().hex[:10]}{ext}"
    path = os.path.join(folder, name)
    data = file.file.read(50 * 1024 * 1024 + 1)
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "That sound is over 50 MB.")
    with open(path, "wb") as f:
        f.write(data)
    try:
        dur, has_audio = clipper.media_info(path)
    except clipper.ClipperError:
        has_audio = False
    if not has_audio:
        os.remove(path)
        raise HTTPException(400, "Tiny Clipper couldn't read that audio file.")
    return {"file": name, "name": os.path.splitext(file.filename)[0][:40], "dur": round(dur, 2)}


def clean_edl(raw, duration, folder):
    """Bounds-check an edit from the page: numbers clamped, files real, sizes capped."""
    def num(v, lo, hi, default):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return min(hi, max(lo, v)) if math.isfinite(v) else default

    def obj(v):
        return v if isinstance(v, dict) else {}

    raw = obj(raw)
    segments = []
    for seg in (raw.get("segments") or [])[:60]:
        seg = obj(seg)
        start, end = num(seg.get("s"), 0, duration, None), num(seg.get("e"), 0, duration, None)
        if start is None or end is None or end - start < 0.2:
            continue
        frame = obj(seg.get("frame"))
        part = {"s": round(start, 3), "e": round(end, 3), "volume": num(seg.get("volume"), 0, 2, 1.0),
                "frame": {"mode": frame.get("mode") if frame.get("mode") in FRAME_MODES else "fill",
                          "x": num(frame.get("x"), 0, 1, 0.5), "y": num(frame.get("y"), 0, 1, 0.5),
                          "zoom": num(frame.get("zoom"), 1, 3, 1.0)}}
        if isinstance(seg.get("captions_text"), str):
            part["captions_text"] = seg["captions_text"][:5000]
        segments.append(part)
    if not segments:
        raise HTTPException(400, "Keep at least one part of the video.")
    total = sum(x["e"] - x["s"] for x in segments)
    if total > 180:
        raise HTTPException(400, "Keep the clip under 3 minutes.")

    texts = []
    for t in (raw.get("texts") or [])[:40]:
        t = obj(t)
        t0, t1 = num(t.get("t0"), 0, total, 0.0), num(t.get("t1"), 0, total, total)
        if not str(t.get("text", "")).strip() or t1 - t0 < 0.1:
            continue
        color = str(t.get("color", ""))
        texts.append({"id": str(t.get("id", ""))[:24], "text": str(t["text"])[:300], "t0": t0, "t1": t1,
                      "x": num(t.get("x"), 0, 1, 0.5), "y": num(t.get("y"), 0, 1, 0.3),
                      "size": num(t.get("size"), 20, 200, 72), "box": bool(t.get("box")),
                      "color": color if re.fullmatch(r"#[0-9a-fA-F]{6}", color) else "#ffffff"})

    sounds = []
    for snd in (raw.get("sounds") or [])[:30]:
        snd = obj(snd)
        file = str(snd.get("file", ""))
        if not ((file.startswith("sfx:") and file[4:] in clipper.SFX) or
                (re.fullmatch(r"sounds/[0-9a-f]{10}\.[a-z0-9]{2,5}", file) and os.path.exists(os.path.join(folder, file)))):
            continue
        dur = num(snd.get("dur"), 0, 3600, 1.0)
        duck = snd.get("duck")
        if duck is not None:
            duck = bool(duck)
        sounds.append({"id": str(snd.get("id", ""))[:24], "file": file, "name": str(snd.get("name", ""))[:40],
                       "dur": dur, "t0": num(snd.get("t0"), 0, total, 0.0),
                       "volume": num(snd.get("volume"), 0, 2, 1.0),
                       **({"duck": duck} if duck is not None else {})})

    cap, hook = obj(raw.get("captions")), obj(raw.get("hook"))
    return {"title": str(raw.get("title", "")).strip()[:100], "segments": segments, "texts": texts, "sounds": sounds,
            "captions": {"on": bool(cap.get("on", True)), "y": num(cap.get("y"), 0.1, 0.97, clipper.CAPTION_Y),
                         "size": num(cap.get("size"), 40, 140, 86)},
            "hook": {"on": bool(hook.get("on")), "line": str(hook.get("line", "")).strip()[:160]}}


def render_edited(job_id, n, edl):
    folder = job_dir(job_id)
    try:
        job = read_job(job_id)
        clip = dict(job["clips"][n - 1])
        src, words = source_of(folder), load_words(folder)
        fixed = False
        for seg in edl["segments"]:  # caption word fixes go into the transcript itself
            text = seg.pop("captions_text", None)
            if text is not None:
                new = clipper.replace_words(words, seg["s"], seg["e"], text)
                fixed, words = fixed or new is not words, new
        if fixed:
            save_words(job_id, words)

        intro = clip.get("intro") if edl["hook"]["on"] else None
        i = clip.get("i", n)  # the clip's file index; differs from n after a clip failed
        voice = saved_voice(folder, i)
        line = edl["hook"]["line"]
        if intro and line and line != clip.get("voiceover"):
            if voice:
                os.remove(voice)
            update_clip(job_id, n, edit_status="Recording the new hook voice…")
            voice = server_voices(folder, [{"i": i, "text": line}], use_keys()).get(i)
            if not voice:
                clip["edit_warning"] = NO_VOICE
            clip["voiceover"] = line

        plan = None
        spans = [(x["s"], x["e"]) for x in edl["segments"] if x["frame"]["mode"] == "auto"]
        if spans or intro:
            lo = min([a for a, _ in spans] + ([intro["start"]] if intro else []))
            hi = max([b for _, b in spans] + ([intro["end"]] if intro else []))
            try:
                with open(os.path.join(folder, f"preview_{n:02d}.json"), encoding="utf-8") as f:
                    meta = json.load(f)
                if meta["window"][0] <= lo and hi <= meta["window"][1]:
                    plan = meta["plan"]  # already worked out when the editor opened
            except (OSError, ValueError, KeyError):
                pass
            if plan is None:
                update_clip(job_id, n, edit_status="Framing…")
                try:
                    plan = clipper.plan_parts(src, [{"s": a, "e": b} for a, b in spans]
                                              + ([{"s": intro["start"], "e": intro["end"]}] if intro else []),
                                              gemini_keys(use_keys()))
                except Exception as e:
                    log.warning("Framing failed for clip %d: %s", n, e)

        update_clip(job_id, n, edit_status="Rendering…")
        if voice:  # also fixes voices recorded before this check existed
            voice = clean_voice(voice, clip.get("voiceover") or line, use_keys()["groq"])
        old_file = clip.get("file")
        clean_title = clipper.safe_filename(edl.get("title") or clip.get("title") or f"clip_{i:02d}")
        out_name = clean_title or f"clip_{i:02d}"
        new_file = clipper.render_edit(src, edl, words, folder, f"clip_{i:02d}", intro, voice, plan, out_name=out_name)
        if old_file and old_file != new_file and os.path.exists(os.path.join(folder, old_file)):
            try:
                os.remove(os.path.join(folder, old_file))
            except OSError:
                pass
        clip["file"] = new_file
        clip.update(edit=edl, title=edl["title"] or clip["title"], hook_off=not edl["hook"]["on"], framing=None,
                    start=min(x["s"] for x in edl["segments"]), end=max(x["e"] for x in edl["segments"]),
                    parts=[{"s": x["s"], "e": x["e"]} for x in edl["segments"]],
                    length=round(clipper.media_info(os.path.join(folder, clip["file"]))[0]),
                    v=int(time.time()), editing=False, edit_status=None, edit_error=None)
        update_clip(job_id, n, **clip)
    except Exception as e:  # keep the old clip and say why
        log.exception("job %s: editing clip %d failed", job_id, n)
        update_clip(job_id, n, editing=False, edit_status=None, edit_error=f"Couldn't render the edit: {e}")


@app.post("/api/jobs/{job_id}/clips/{n}/render")
def render_endpoint(job_id: str, n: int, body: dict):
    job, clip, folder = editable_clip(job_id, n)
    if clip.get("editing"):
        raise HTTPException(409, "This clip is already being rendered.")
    edl = clean_edl(body.get("edit"), clipper.media_info(source_of(folder))[0], folder)
    update_clip(job_id, n, editing=True, edit_status="Waiting for the current video to finish…",
                edit_error=None, edit_warning=None)
    worker.submit(render_edited, job_id, n, edl)
    return {"ok": True}


@app.get("/editor")
def editor_page():
    return FileResponse(os.path.join(ROOT, "web", "editor.html"), headers={"Cache-Control": "no-store"})


def migrate_library():
    """Migrate older library items so folder is video title and clips use generated titles."""
    if not os.path.exists(LIBRARY):
        return
    for name in list(os.listdir(LIBRARY)):
        folder = os.path.join(LIBRARY, name)
        if not os.path.isdir(folder):
            continue
        try:
            with open(os.path.join(folder, "job.json"), encoding="utf-8") as f:
                job = json.load(f)
        except (OSError, ValueError):
            continue

        changed = False
        current_folder = folder

        # 1. Rename folder if it is an old 12-char hex id and video title is available
        if re.fullmatch(r"[0-9a-f]{12}", name) and job.get("name"):
            target_folder = unique_job_folder(job["name"], current=folder)
            if os.path.abspath(folder) != os.path.abspath(target_folder):
                try:
                    os.rename(folder, target_folder)
                    current_folder = target_folder
                    job["id"] = os.path.basename(target_folder)
                    changed = True
                except OSError as e:
                    log.warning("Migration: couldn't rename %s -> %s: %s", folder, target_folder, e)

        # 2. Rename clips that still use clip_NN.mp4 to their short generated title
        used_names = set()
        for clip in job.get("clips", []):
            title = clip.get("title")
            cur_file = clip.get("file")
            if not title or not cur_file:
                continue
            cur_path = os.path.join(current_folder, cur_file)
            if not os.path.exists(cur_path):
                continue

            if re.match(r"^clip_\d+\.mp4$", cur_file):
                clean = clipper.safe_filename(title) or f"clip_{clip.get('i', 1):02d}"
                candidate = clean
                dup = 2
                while candidate in used_names or (os.path.exists(os.path.join(current_folder, candidate + ".mp4")) and candidate + ".mp4" != cur_file):
                    candidate = f"{clean} ({dup})"
                    dup += 1
                used_names.add(candidate)

                new_file = candidate + ".mp4"
                new_path = os.path.join(current_folder, new_file)
                try:
                    os.rename(cur_path, new_path)
                    clip["file"] = new_file
                    changed = True

                    cur_thumb = clip.get("thumb")
                    if cur_thumb:
                        cur_thumb_path = os.path.join(current_folder, cur_thumb)
                        if os.path.exists(cur_thumb_path) and re.match(r"^thumb_\d+\.jpg$", cur_thumb):
                            new_thumb = candidate + ".jpg"
                            new_thumb_path = os.path.join(current_folder, new_thumb)
                            if not os.path.exists(new_thumb_path):
                                os.rename(cur_thumb_path, new_thumb_path)
                                clip["thumb"] = new_thumb
                except OSError as e:
                    log.warning("Migration: couldn't rename clip file: %s", e)
            else:
                used_names.add(os.path.splitext(cur_file)[0])

        if changed:
            try:
                write_job(job)
            except OSError as e:
                log.warning("Migration: couldn't save job.json for %s: %s", job.get("id"), e)


migrate_library()
mark_interrupted()
threading.Thread(target=auto_approve, daemon=True).start()
app.mount("/library", StaticFiles(directory=LIBRARY), name="library")


@app.get("/api/logs")
def logs(lines: int = 400):
    """The end of the log file, to see what an API answered when something failed."""
    try:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
            text = "".join(f.readlines()[-max(1, min(lines, 5000)):])
    except OSError:
        text = "Nothing logged yet."
    return PlainTextResponse(text, headers={"Cache-Control": "no-store"})


@app.get("/")
def index(code: str = "", state: str = "", error: str = ""):
    """The home page, which is also where Google's YouTube sign-in comes back to."""
    if state and state == pending_state.get("youtube"):
        pending_state.pop("youtube")
        note = "connected=youtube"
        try:
            if error or not code:
                raise social.SocialError(f"YouTube: sign-in was cancelled ({error or 'no code'})")
            save_settings({"youtube": social.yt_connect(social_cfg("youtube"), code, REDIRECT)})
        except (social.SocialError, httpx.HTTPError) as e:
            note = "social_error=" + urllib.parse.quote(str(e))
        return RedirectResponse(f"/?{note}")
    # no-store: a reload must always run the latest page
    return FileResponse(os.path.join(ROOT, "web", "index.html"), headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT)
