"""Tiny Clipper pipeline: one long video in, captioned 9:16 clips out.

Free hosted APIs plus one tiny on-device face detector, so no GPU is needed:
  transcription   Groq whisper-large-v3-turbo, word-level timestamps
  moment picking  Gemini, falling back to a Groq LLM
  framing         per shot: follow the face (OpenCV), else the subject (Gemini), else centre
  hook voice      Gemini TTS through Puter, recorded by the open web page
  cutting         ffmpeg (system one, else the imageio-ffmpeg bundle)
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

import httpx

log = logging.getLogger("tinyclipper")

GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Google Cloud (Vertex AI, now "Agent Platform"): same Gemini models, billed to
# the Cloud project, so free-trial / promotional credit pays for them.
VERTEX_BASE = "https://aiplatform.googleapis.com/v1/publishers/google/models"
WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
# gemini-3.8-flash exists but its free tier is ~20 requests/day (Oct 2026);
# flash-lite's is far roomier and is plenty for picking moments.
# Models are tried in order: when one is overloaded (503), out of free quota
# (429, counted per model) or retired (404), the next one takes over.
# gemini-3.8-flash is last: its free tier is ~20 requests/day (Oct 2026).
GEMINI_MODELS = list(dict.fromkeys(
    ([os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL") else [])
    + ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash",
       "gemini-3.8-flash"]))  # 2.5 models are closed to new users
GROQ_LLM_MODELS = list(dict.fromkeys(
    ([os.environ["GROQ_LLM_MODEL"]] if os.environ.get("GROQ_LLM_MODEL") else [])
    + ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]))
GEMINI_MODEL = GEMINI_MODELS[0]
LLM_TIMEOUT = 150  # a hung model counts as failed; long transcripts answer in well under a minute
VERTEX_TIMEOUT = 600  # paid Cloud: wait out slow answers rather than fail the job
STREAM_TIMEOUT = 90  # max silence on a streamed answer; thought summaries keep a thinking model talking,
# so silence this long means a hung request (good answers take 3-35 s, hung ones never finish)
# With Google Cloud credit: Flash does everything (3.1 Pro preview kept timing out).
VERTEX_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash"]  # 3.5: when 3.8 hangs
VERTEX_VISION_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash"]
TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]
TTS_VOICE = os.environ.get("TTS_VOICE", "Puck")  # upbeat; 30 voices, e.g. Fenrir (excitable), Charon (informative)
SKIP_TO_NEXT_MODEL = (404, 413, 429, 500, 503)  # gone, too big for its free tier, quota, overloaded

CHUNK_SECONDS = 1200          # 20 min of 32 kbps opus is ~5 MB, under Groq's 25 MB free cap
GOOGLE_CHUNK_SECONDS = 600    # Google: 30 min max with word timestamps; smaller inline requests
GOOGLE_STT_MODEL = "gemini-3.5-transcribe"
# Cloud Speech-to-Text v2 (Chirp 3): needs a Google sign-in on this computer
# (Application Default Credentials), since Speech-to-Text refuses API keys.
CHIRP_LOCATION = os.environ.get("CHIRP_LOCATION", "us")   # "us" or "eu"
CHIRP_LANGUAGE = os.environ.get("CHIRP_LANGUAGE", "auto")  # e.g. "en-US" to skip detection
CHIRP_PIECE = 55  # Recognize takes at most 1 minute of audio per request
STT_ENGINE = os.environ.get("STT_ENGINE", "google")  # "groq" to keep Whisper even with a Gemini key
# Thumbnails: Nano Banana Pro, then Nano Banana 2.1 (no free tier: Google Cloud credit or a paid key).
THUMB_MODELS = ["gemini-3-pro-image", "gemini-nano-banana-2.1"]
MIN_CLIP, MAX_CLIP = 40, 75   # what the AI is asked for; MIN_CLIP is also the hard floor of a finished clip
KEEP_MIN, KEEP_MAX = 10, 90   # what we accept from it
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")


class ClipperError(Exception):
    """A failure whose message is written for the user."""


def safe_filename(name: str, max_len: int = 100) -> str:
    """Sanitize a string into a safe directory or file name on Windows, Linux, and macOS."""
    if not name:
        return ""
    # Replace colons, slashes, pipes with ' - '
    s = re.sub(r'[:/\\|]', ' - ', str(name))
    # Remove remaining invalid characters: < > " ? * and control chars
    s = re.sub(r'[<>"?*\x00-\x1f]', '', s)
    # Collapse multiple whitespace
    s = re.sub(r'\s+', ' ', s)
    # Collapse multiple dashes
    s = re.sub(r'(?:\s*-\s*)+', ' - ', s)
    # Strip spaces, periods, dashes from start and end
    s = s.strip(". -")
    # Windows reserved filenames check
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if s.upper() in reserved:
        s = f"_{s}"
    if len(s) > max_len:
        s = s[:max_len].rstrip(". -")
    return s


# --- ffmpeg -----------------------------------------------------------------

def ffmpeg_exe():
    """System ffmpeg, else the one bundled with imageio-ffmpeg, copied once to
    tools/ffmpeg.exe so yt-dlp (which looks for that exact name) finds it too."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = os.path.join(TOOLS_DIR, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not os.path.exists(local):
        import imageio_ffmpeg
        os.makedirs(TOOLS_DIR, exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), local)
    return local


_encoder_args = None


def video_encoder_args(force_software=False):
    """Return best available H.264 video encoder arguments.
    Probes hardware encoders (NVENC, QSV, VideoToolbox, AMF, MediaFoundation)
    and falls back smoothly to CPU libx264."""
    global _encoder_args
    if force_software or os.environ.get("CLIPPER_FORCE_CPU"):
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]
    if _encoder_args is not None:
        return list(_encoder_args)

    candidates = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22", "-pix_fmt", "yuv420p"]),
        ("h264_qsv", ["-c:v", "h264_qsv", "-global_quality", "23", "-pix_fmt", "yuv420p"]),
        ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-b:v", "5000k", "-pix_fmt", "yuv420p"]),
        ("h264_amf", ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cbr", "-b:v", "5000k", "-pix_fmt", "yuv420p"]),
        ("h264_mf", ["-c:v", "h264_mf", "-b:v", "5000k", "-pix_fmt", "yuv420p"]),
    ]
    for name, args in candidates:
        try:
            cmd = [ffmpeg_exe(), "-hide_banner", "-f", "lavfi", "-i", "color=s=64x64:d=0.1", *args, "-f", "null", "-"]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode == 0:
                log.info("Using hardware video encoder: %s", name)
                _encoder_args = list(args)
                return list(_encoder_args)
        except Exception:
            pass

    log.info("Using software video encoder: libx264")
    _encoder_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]
    return list(_encoder_args)



def run_ffmpeg(args, cwd=None):
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-y", *args], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise ClipperError(f"Video processing failed: {last[:300]}")


def media_info(path):
    """(duration_seconds, has_audio), read from ffmpeg's banner because the
    bundled build ships without ffprobe."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        raise ClipperError("This file isn't a video Tiny Clipper can read.")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s), bool(re.search(r"Stream #.*Audio:", proc.stderr))


# --- HTTP with retries --------------------------------------------------------

def retry_wait(retry_after, attempt):
    """Seconds to wait after a 429/5xx: the server's retry-after when given,
    else exponential backoff; capped at 10 minutes."""
    try:
        wait = float(retry_after)
    except (TypeError, ValueError):
        wait = 10 * 2 ** attempt
    return max(1.0, min(wait, 600.0))


def post(url, retries=4, on_wait=None, timeout=300, wait_on_429=True, **kwargs):
    """POST that waits out rate limits and server hiccups. Groq's free tier caps
    audio-seconds per hour, so a long video can hit 429 midway; waiting beats failing."""
    where = httpx.URL(url).copy_with(query=None)  # keys travel in headers; never log a query string
    for attempt in range(retries + 1):
        started = time.time()
        try:
            resp = httpx.post(url, timeout=timeout, **kwargs)
        except httpx.TransportError as e:
            log.warning("POST %s -> %s after %.1fs (try %d of %d): %s", where, type(e).__name__,
                        time.time() - started, attempt + 1, retries + 1, e)
            if attempt == retries and isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout)):
                raise ClipperError(f"{httpx.URL(url).host} was reached but didn't answer within {timeout:.0f} s "
                                   f"(the AI was too slow, not your internet).")
            if attempt == retries:
                raise ClipperError(f"Couldn't reach {httpx.URL(url).host}. Check your internet connection. ({e})")
            time.sleep(retry_wait(None, attempt))
            continue
        if resp.status_code == 200:
            log.info("POST %s -> 200 in %.1fs", where, time.time() - started)
        else:  # the whole answer: the reason is usually in there
            log.warning("POST %s -> %d in %.1fs (try %d of %d): %s", where, resp.status_code,
                        time.time() - started, attempt + 1, retries + 1, resp.text[:2000])
        if ((resp.status_code == 429 and wait_on_429) or resp.status_code >= 500) and attempt < retries:
            wait = retry_wait(resp.headers.get("retry-after"), attempt)
            log.info("waiting %.0fs before retrying %s", wait, where)
            if on_wait:
                on_wait(wait)
            time.sleep(wait)
            continue
        return resp


def key_list(keys):
    """One key or a list of them (backups) -> the list, empty entries dropped."""
    return [k for k in ([keys] if isinstance(keys, str) else (keys or [])) if k]


def post_rotating(url, keys, make, on_wait=None, rounds=4, **kwargs):
    """POST with the first key; one that's rate limited (429) hands over to the
    next. Only when every key is limited does it wait, like post() does.
    make(key) -> the request's keyword arguments for that key."""
    keys = key_list(keys)
    if not keys:
        raise ClipperError("Add your free Groq key first (button: Keys).")
    for attempt in range(rounds + 1):
        for key in keys:
            resp = post(url, retries=3, wait_on_429=False, **make(key), **kwargs)
            if resp.status_code != 429:
                return resp
        if attempt == rounds:
            return resp
        wait = retry_wait(resp.headers.get("retry-after"), attempt)
        if on_wait:
            on_wait(wait)
        time.sleep(wait)


def api_error(name, resp):
    if resp.status_code in (400, 401, 403) and "key" in resp.text.lower():
        return ClipperError(f"Your {name} key was rejected. Enter it again in Keys.")
    if resp.status_code == 429:
        return ClipperError(f"{name}'s free quota is used up for now. Try again later.")
    return ClipperError(f"{name} error {resp.status_code}: {resp.text[:200]}")


def check_key(service, key):
    """(ok, message) after asking the provider to list its models with this key."""
    url, headers = {
        "groq": (f"{GROQ_BASE}/models", {"Authorization": f"Bearer {key}"}),
        "gemini": (f"{GEMINI_BASE}/models", {"x-goog-api-key": key}),
    }[service]
    try:
        resp = httpx.get(url, headers=headers, timeout=15)
    except httpx.TransportError:
        return False, "Couldn't reach the provider. Check your internet connection."
    if resp.status_code == 200:
        return True, "Key works."
    if resp.status_code == 429:
        return True, "Key works, but its free quota is used up for now."
    if resp.status_code in (400, 401, 403):
        return False, "This key was rejected. Copy it again from the provider's page."
    return False, f"The provider answered {resp.status_code}. Try again in a minute."


# --- download ---------------------------------------------------------------

def download(url, out_dir, progress):
    """Fetch a video link with yt-dlp; returns (path, title)."""
    import yt_dlp

    def hook(d):
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if d.get("status") == "downloading" and total:
            progress("Downloading the video", d.get("downloaded_bytes", 0) / total)

    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "outtmpl": os.path.join(out_dir, "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "concurrent_fragment_downloads": 4,
        "ffmpeg_location": os.path.dirname(ffmpeg_exe()),
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise ClipperError(f"Couldn't download that link. Try downloading the video yourself and "
                           f"uploading the file instead. ({str(e)[:200]})")
    files = [f for f in os.listdir(out_dir) if f.startswith("source.") and not f.endswith(".part")]
    if not files:
        raise ClipperError("The download finished but no video file was saved.")
    return os.path.join(out_dir, files[0]), info.get("title") or url


# --- transcription ----------------------------------------------------------

def parse_transcript(body, offset):
    """Groq verbose_json -> (words, segments), shifted by the chunk's offset."""
    words = [{"w": str(w["word"]).strip(), "s": round(float(w["start"]) + offset, 3),
              "e": round(float(w["end"]) + offset, 3)}
             for w in body.get("words") or [] if str(w.get("word", "")).strip()]
    segments = [{"s": round(float(s["start"]) + offset, 2), "e": round(float(s["end"]) + offset, 2),
                 "text": str(s.get("text", "")).strip()}
                for s in body.get("segments") or [] if str(s.get("text", "")).strip()]
    return words, segments


def punctuate(words, text):
    """Google's timed words come bare; its text has the punctuation and casing.
    Line them up so each word carries its punctuation (sentence cuts need it)."""
    tokens, i, out = text.split(), 0, []
    for w in words:
        for j in range(i, min(i + 4, len(tokens))):
            if _norm(tokens[j]) == _norm(w["w"]):
                w, i = {**w, "w": tokens[j]}, j + 1
                break
        out.append(w)
    return out


def sentence_segments(words):
    """Segments that end where a word ends a sentence, for build_sentences."""
    segments, cur = [], []
    for w in words:
        cur.append(w)
        if w["w"][-1:] in ".?!…":
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)
    return [{"s": c[0]["s"], "e": c[-1]["e"], "text": " ".join(w["w"] for w in c)} for c in segments]


def google_transcribe(audio, keys, offset):
    """One chunk through Gemini 3.5 Transcribe (Google Cloud credit first):
    (words, segments) shifted by offset."""
    import base64
    studio = split_keys(keys)[0]  # not on Google Cloud: that route would only 404
    resp = gemini_request({"contents": [{"role": "user", "parts": [
                               {"inlineData": {"mimeType": "audio/ogg", "data": base64.b64encode(audio).decode()}}]}],
                           "generationConfig": {"audioTranscriptionConfig": {"wordTimestamp": True}}},
                          studio, [GOOGLE_STT_MODEL], [GOOGLE_STT_MODEL], "Google speech-to-text", timeout=VERTEX_TIMEOUT)
    try:
        parts = resp.json()["candidates"][0]["content"]["parts"]
        heard = next(x["audioTranscription"] for x in parts if "audioTranscription" in x)
    except (KeyError, IndexError, StopIteration, ValueError):
        log.warning("Google speech-to-text answer without words: %s", resp.text[:2000])
        raise ClipperError("Google speech-to-text sent back no words.")

    def secs(v):
        return round(float(str(v).rstrip("s")) + offset, 3)

    words = punctuate([{"w": str(w["word"]).strip(), "s": secs(w["startOffset"]), "e": secs(w["endOffset"])}
                       for w in heard.get("words") or [] if str(w.get("word", "")).strip()], heard.get("text", ""))
    return words, sentence_segments(words)


# --- Google sign-in (Application Default Credentials) ---------------------------

class _HttpxRequest:
    """google-auth's transport interface on top of httpx (no 'requests' needed)."""
    class _Response:
        def __init__(self, r):
            self.status, self.headers, self.data = r.status_code, dict(r.headers), r.content

    def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
        import google.auth.exceptions
        try:
            return self._Response(httpx.request(method, url, content=body, headers=headers, timeout=timeout or 60))
        except httpx.HTTPError as e:
            raise google.auth.exceptions.TransportError(e)


_signin = {}


def google_signin():
    """(access token, project) from the Google sign-in on this computer
    (`gcloud auth application-default login`), or None if there isn't one."""
    try:
        import google.auth
        if "creds" not in _signin:
            creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            _signin.update(creds=creds, project=getattr(creds, "quota_project_id", None) or project)
        creds = _signin["creds"]
        if not creds.valid:
            creds.refresh(_HttpxRequest())
        return creds.token, _signin["project"]
    except Exception as e:  # not signed in, expired sign-in, or no project set
        _signin.clear()
        log.info("No Google sign-in for Speech-to-Text: %s", str(e)[:300])
        return None


def speech_pieces(media_path, duration, work):
    """Cut the audio into pieces under a minute, at pauses where possible, so no
    word is split. Returns [(start offset, path)]."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", media_path, "-vn",
                           "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    pauses = [(a + b) / 2 for a, b in zip(starts, ends)]
    cuts, at = [], 0.0
    while duration - at > CHIRP_PIECE:
        ok = [p for p in pauses if at + 20 < p <= at + CHIRP_PIECE]
        at = ok[-1] if ok else at + CHIRP_PIECE  # latest pause in range, else a hard cut
        cuts.append(round(at, 3))
    run_ffmpeg(["-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k",
                "-f", "segment", "-reset_timestamps", "1",  # each piece starts at 0: times are per piece
                *(["-segment_times", ",".join(f"{c:.3f}" for c in cuts)] if cuts else []),
                os.path.join(work, "p%04d.ogg")])
    names = sorted(f for f in os.listdir(work) if f.startswith("p"))
    return list(zip([0.0] + cuts, (os.path.join(work, n) for n in names)))


def chirp_piece(path, offset, signin):
    """One piece through Speech-to-Text v2 Chirp 3: (words, segments)."""
    import base64
    token, project = signin
    loc = CHIRP_LOCATION
    with open(path, "rb") as f:
        audio = base64.b64encode(f.read()).decode()
    resp = post(f"https://{loc}-speech.googleapis.com/v2/projects/{project}/locations/{loc}/recognizers/_:recognize",
                retries=3, timeout=180, headers={"Authorization": f"Bearer {token}", "x-goog-user-project": project},
                json={"config": {"autoDecodingConfig": {}, "model": "chirp_3", "languageCodes": [CHIRP_LANGUAGE],
                                 "features": {"enableWordTimeOffsets": True, "enableAutomaticPunctuation": True}},
                      "content": audio})
    if resp.status_code != 200:
        raise api_error("Google Speech-to-Text", resp)

    def secs(v):
        return round(float(str(v or "0s").rstrip("s")) + offset, 3)

    words = []
    for result in resp.json().get("results") or []:
        alt = (result.get("alternatives") or [{}])[0]
        heard = [{"w": str(w.get("word", "")).strip(), "s": secs(w.get("startOffset")), "e": secs(w.get("endOffset"))}
                 for w in alt.get("words") or [] if str(w.get("word", "")).strip()]
        words += punctuate(heard, alt.get("transcript", ""))
    return words, sentence_segments(words)


def chirp_transcribe(media_path, signin, progress):
    """The whole video through Chirp 3, a few pieces at a time: (words, segments)."""
    from concurrent.futures import ThreadPoolExecutor
    work = tempfile.mkdtemp(prefix="tinyclipper_")
    try:
        pieces = speech_pieces(media_path, media_info(media_path)[0], work)
        done = 0
        results = [None] * len(pieces)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(chirp_piece, path, offset, signin): i for i, (offset, path) in enumerate(pieces)}
            for fut in futures:
                results[futures[fut]] = fut.result()
                done += 1
                progress(f"Turning speech into text with Google ({done} of {len(pieces)})", done / len(pieces))
        words = [w for r in results for w in r[0]]
        return words, sentence_segments(words)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def transcribe(media_path, keys, progress):
    """Word-timestamped transcript; returns (words, segments, language).
    Google (Gemini 3.5 Transcribe) when there's a Gemini key, Groq Whisper as the
    backup or when it's the only key. keys: {"groq", "gemini", "vertex"} (a bare
    list means Groq keys)."""
    keys = keys if isinstance(keys, dict) else {"groq": keys}
    groq = key_list(keys.get("groq"))
    signin = google_signin() if STT_ENGINE != "groq" else None
    if signin:  # Google Cloud Speech-to-Text (Chirp 3), paid from the Cloud project
        try:
            words, segments = chirp_transcribe(media_path, signin, progress)
            return words, segments, None
        except (ClipperError, httpx.HTTPError) as e:
            if not (groq or any(split_keys(keys))):
                raise
            log.warning("Google Speech-to-Text failed (%s); trying the next option", e)
    # gemini-3.5-transcribe only exists on the AI Studio side (not on Google Cloud):
    # free Gemini keys only.
    use_google = STT_ENGINE != "groq" and bool(split_keys(keys)[0])
    if not (use_google or groq):
        if split_keys(keys)[1]:
            raise ClipperError("Speech-to-text on Google Cloud needs the Google sign-in on this computer: install the "
                               "Google Cloud CLI, run 'gcloud auth application-default login', then "
                               "'gcloud auth application-default set-quota-project YOUR_PROJECT_ID', and restart "
                               "Tiny Clipper. (Or add a Groq key and turn off 'Use only Google Cloud'.)")
        raise ClipperError("Add a Gemini or Groq key in Keys first.")
    chunk = GOOGLE_CHUNK_SECONDS if use_google else CHUNK_SECONDS
    work = tempfile.mkdtemp(prefix="tinyclipper_")
    try:
        run_ffmpeg(["-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                    "-b:a", "32k", "-f", "segment", "-segment_time", str(chunk),
                    os.path.join(work, "c%04d.ogg")])
        names = sorted(os.listdir(work))
        words, segments, language = [], [], None
        for i, name in enumerate(names):
            progress(f"Turning speech into text (part {i + 1} of {len(names)})", i / len(names))
            with open(os.path.join(work, name), "rb") as f:
                audio = f.read()
            if use_google:
                try:
                    w, sg = google_transcribe(audio, keys, i * chunk)
                    words, segments = words + w, segments + sg
                    continue
                except (ClipperError, httpx.HTTPError) as e:
                    if not groq:
                        raise
                    log.warning("Google speech-to-text failed (%s); using Groq Whisper", e)
                    use_google = False
            resp = post_rotating(f"{GROQ_BASE}/audio/transcriptions", groq, lambda key: dict(
                        headers={"Authorization": f"Bearer {key}"},
                        files={"file": (name, audio, "audio/ogg")},
                        data={"model": WHISPER_MODEL, "response_format": "verbose_json",
                              "timestamp_granularities[]": ["word", "segment"]}),
                        on_wait=lambda sec: progress(f"Groq's free limit reached, waiting {sec:.0f}s…"))
            if resp.status_code != 200:
                raise api_error("Groq", resp)
            body = resp.json()
            language = language or body.get("language")
            w, sg = parse_transcript(body, i * chunk)
            words, segments = words + w, segments + sg
        return words, segments, language
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- moment picking ---------------------------------------------------------

PROMPT = """You are an expert short-form video editor. Below is the transcript of a video that is {duration:.0f} seconds long, one sentence per line, as: id | start time | sentence.

Make the {n} best TikTok / Instagram Reels / YouTube Shorts clips from it. Every clip is about ONE subject the speaker really talks through, and tells it as a complete little story: someone who sees only the clip understands the point and gets the payoff.

Choosing subjects:
- Pick the subjects the speaker develops in depth and explains clearly: a story with a turn, an argument with its reasons, a lesson with its example, a strong opinion with its why.
- Prefer stretches where the speaker is fluent, confident and on topic. Skip rambling, stuttering, small talk, housekeeping ("subscribe", "next question") and one-liners that would need padding to fill time.
- Every clip has an arc: hook (why care), setup (the context a newcomer needs), development (the substance: details, reasons, example), payoff (the answer, the punchline, the conclusion).

How to build each clip:
1. Decide its subject and message first, in one sentence ("message").
2. Choose the sentences that deliver it, as "parts": runs of consecutive sentences, each [first_id, last_id] (both included).
   - If one continuous stretch says it all, use a single part. That's the best clip when it works; use as few parts as you can.
   - Otherwise join up to {max_parts} parts from anywhere in the video: leave out tangents, filler, repetition and side stories, and keep the context, the key point and the payoff.
   - Parts play in the order you list them. Keep the video's order unless a later sentence makes a much stronger opening and the clip still makes sense that way.
   - Every part is whole sentences. Never keep a sentence that points at something you cut out ("as I said", "that one", "he" when we don't know who).
3. Check it: the first sentence works as an opening (a hook, a question or a bold claim, never "And...", "So...", "But..." leaning on missing context) and the last sentence lands the payoff (the answer, the punchline, the conclusion).

All parts of a clip together last between {lo} and {hi} seconds (judge it from the start times); never under {lo}. If a subject can't fill {lo} seconds with real substance, pick a different subject instead of padding. Two clips never use the same sentence.

Each clip also gets a cold open, a teaser played before it starts:
- "teaser_id": the id of the single most gripping sentence in the clip's parts (a reaction, a punchline, a shocking line), not its first sentence.
- "voiceover": what a narrator says over that teaser, 6 to 12 words, building curiosity without spoiling the ending, true to what actually happens. Examples: "Watch how this man turned 100 dollars into a business", "You won't believe what this dog does next".

Answer with JSON only, exactly in this shape:
{{"clips": [{{"message": "the one point this clip gets across", "parts": [[12, 15], [31, 33]], "title": "catchy title, max 8 words", "hook": "one sentence on why viewers will keep watching", "score": 87, "teaser_id": 32, "voiceover": "Watch how this man ..."}}]}}
"score" is the viral potential from 0 to 100. Write "message", "title", "hook" and "voiceover" in the transcript's language.

TRANSCRIPT:
{transcript}"""

REVIEW_PROMPT = """You are the final editor checking draft short-form clips cut from the transcript below (one sentence per line: id | start time | sentence). For each draft you get its message, its parts and the exact words a viewer will hear, part by part.

Watch each draft as a first-time viewer who never saw the full video. Fix a draft when:
- its message doesn't come across completely (setup missing, payoff missing, cut off mid-thought),
- a sentence points at something that was cut out,
- a part is filler, a tangent or a repeat,
- the opening leans on missing context, or the ending trails into another topic,
- it is shorter than {lo} seconds: extend it with the surrounding sentences that develop the same subject (more context, the example, the reasons), never with filler,
- its subject is thin (a single remark stretched out): replace it with a subject the speaker develops properly.
Fix it by changing its parts: add, remove, extend or shorten runs of whole sentences, from anywhere in the video. Leave drafts that already work exactly as they are.

Same rules as the drafts: at most {max_parts} parts per clip, {lo} to {hi} seconds in total, parts play in the listed order, two clips never use the same sentence, "teaser_id" is a sentence in the clip's parts but not its first.

Answer with JSON only: every clip, fixed or not, in the drafts' shape:
{{"clips": [{{"message": "...", "parts": [[12, 15], [31, 33]], "title": "...", "hook": "...", "score": 87, "teaser_id": 32, "voiceover": "..."}}]}}

DRAFTS:
{drafts}

TRANSCRIPT:
{transcript}"""


def build_sentences(words, segments):
    """Split the transcript into sentences with exact word timings, so a cut can
    only fall between sentences. Uses Whisper's punctuation from the segments;
    a segment whose words don't line up with its text stays one piece."""
    if not segments:  # no punctuation to go on: split at pauses
        segments = []
        for w in words:
            if not segments or w["s"] - segments[-1]["e"] > 0.7:
                segments.append({"s": w["s"], "e": w["e"], "text": ""})
            segments[-1]["e"] = w["e"]
            segments[-1]["text"] += " " + w["w"]
    sentences, wi = [], 0
    for n, seg in enumerate(segments):
        last = n == len(segments) - 1
        seg_words = []
        while wi < len(words) and (last or (words[wi]["s"] + words[wi]["e"]) / 2 < seg["e"]):
            seg_words.append(words[wi])
            wi += 1
        if not seg_words:
            continue
        parts = [x for x in re.split(r"(?<=[.!?…])\s+", seg["text"].strip()) if x]
        counts = [len(x.split()) for x in parts]
        if sum(counts) != len(seg_words):
            parts, counts = [seg["text"].strip() or " ".join(w["w"] for w in seg_words)], [len(seg_words)]
        i = 0
        for text, k in zip(parts, counts):
            chunk = seg_words[i:i + k]
            i += k
            sentences.append({"s": chunk[0]["s"], "e": chunk[-1]["e"], "text": text})
    return sentences


_working = {}  # provider -> (model, key) that answered last, tried first next time


def try_models(provider, models, keys, call):
    """call(model, key) -> response for each model with each key (backups), best
    model first, starting with the pair that worked last. Returns the response
    of the first pair that doesn't ask for 'try the next one', else the last one."""
    keys = key_list(keys)
    pairs = sorted(((m, k) for m in models for k in keys), key=lambda p: p != _working.get(provider))
    if not pairs:
        raise ClipperError(f"Add a {provider} key in Keys first.")
    for i, (model, key) in enumerate(pairs):
        nxt = f"{pairs[i + 1][0]}{' (backup key)' if i + 1 < len(pairs) and pairs[i + 1][1] != keys[0] else ''}" \
            if i + 1 < len(pairs) else ""
        try:
            resp = call(model, key)
        except ClipperError as e:  # unreachable or hung: the next model may still answer
            if i == len(pairs) - 1:
                raise
            log.warning("%s %s didn't answer (%s); trying %s", provider, model, e, nxt)
            continue
        if resp.status_code not in SKIP_TO_NEXT_MODEL or i == len(pairs) - 1:
            if resp.status_code == 200:
                _working[provider] = (model, key)
            return model, resp
        log.warning("%s %s answered %d; trying %s", provider, model, resp.status_code, nxt)


def split_keys(keys):
    """(AI Studio keys, Google Cloud keys) from a key list or {"gemini": [...], "vertex": [...]}.
    A key that works on Google Cloud is used only there: its project has billing on,
    so on AI Studio it would be paid usage that Cloud trial credit doesn't cover."""
    if isinstance(keys, dict):
        cloud = key_list(keys.get("vertex"))
        return [k for k in key_list(keys.get("gemini")) if k not in cloud], cloud
    return key_list(keys), []


last_model = {}  # what answered the last Gemini request: {"model", "cloud"}


def merge_stream(resp):
    """A streamGenerateContent (SSE) answer as one generateContent response:
    every chunk's parts in order, the last chunk's finishReason and usage."""
    parts, last = [], {}
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            last = json.loads(line[5:])
            parts += (((last.get("candidates") or [{}])[0]).get("content") or {}).get("parts") or []
    cand = dict(((last.get("candidates") or [{}])[0]), content={"role": "model", "parts": parts})
    return httpx.Response(200, json={**last, "candidates": [cand]}, request=resp.request)


def gemini_request(body, keys, vertex_models=VERTEX_MODELS, studio_models=None, label="Gemini", timeout=None,
                   stream=False):
    """POST a generateContent body: Google Cloud models first (if a key has it
    enabled), then the AI Studio models. Returns the 200 response.
    stream: use streamGenerateContent for long-thinking text answers. Google Cloud
    gemini-3.8-flash sometimes never answers a request that thinks for ~30 s+
    (seen: 2x300 s timeouts; the same request answered in 35 s on another try).
    STREAM_TIMEOUT bounds the silence; then a retry, then the next model.
    Not for audio/images: their
    data would arrive in pieces."""
    method = "streamGenerateContent?alt=sse" if stream else "generateContent"
    if stream:  # thought summaries stream while the model thinks: silence then means hung, not busy
        body = {**body, "generationConfig": {**body.get("generationConfig", {}),
                                             "thinkingConfig": {"includeThoughts": True}}}
    tries = 1  # retries: 2 attempts, then the next model
    timeout = min(timeout or VERTEX_TIMEOUT, STREAM_TIMEOUT) if stream else timeout
    studio, cloud = split_keys(keys)
    routes = ([("Google Cloud", vertex_models, cloud, VERTEX_BASE, timeout or VERTEX_TIMEOUT)] if cloud else []) + \
             ([("Gemini", studio_models or GEMINI_MODELS, studio, f"{GEMINI_BASE}/models", timeout or LLM_TIMEOUT)]
              if studio else [])
    if not routes:
        raise ClipperError(f"Add a Gemini key in Keys first.")
    for i, (provider, models, ks, base, timeout) in enumerate(routes):
        last = i == len(routes) - 1
        try:
            model, resp = try_models(provider, models, ks, lambda m, k: post(
                f"{base}/{m}:{method}", retries=tries, timeout=timeout, wait_on_429=False,
                headers={"x-goog-api-key": k}, json=body))
        except ClipperError:
            if last:
                raise
            continue
        if resp.status_code == 200:
            last_model.update(model=model, cloud=provider == "Google Cloud")
            return merge_stream(resp) if stream else resp
        if last:
            raise api_error(label, resp)
        log.warning("%s failed (%d); using the free Gemini models", provider, resp.status_code)


def gemini_json(parts, keys, temperature, vertex_models=VERTEX_MODELS):
    """JSON from Gemini for a list of content parts, through every model and key."""
    resp = gemini_request({"contents": [{"role": "user", "parts": parts}],
                           "generationConfig": {"responseMimeType": "application/json", "temperature": temperature}},
                          keys, vertex_models, stream=True)
    try:
        out = resp.json()["candidates"][0]["content"]["parts"]
        return json.loads("".join(p.get("text", "") for p in out if not p.get("thought")))
    except (KeyError, IndexError, ValueError):
        log.warning("Unreadable Gemini answer: %s", resp.text[:2000])
        raise ClipperError("Gemini sent back an empty or unreadable answer.")


def ask_gemini(prompt, keys):
    return gemini_json([{"text": prompt}], keys, 0.4)


def gemini_speak(text, keys, path):
    """The hook voice from Gemini TTS, made here (no browser needed). Only the
    hook text is sent, so nothing else can be read out. Returns the WAV path."""
    import base64
    import wave
    resp = gemini_request({"contents": [{"role": "user", "parts": [{"text": text}]}],
                           "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                               "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": TTS_VOICE}}}}},
                          keys, TTS_MODELS, TTS_MODELS, "Gemini voice")
    try:
        audio = resp.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        data = base64.b64decode(audio["data"])
    except (KeyError, IndexError, ValueError):
        log.warning("Gemini voice answer without audio: %s", resp.text[:2000])
        raise ClipperError("Gemini voice sent back no audio.")
    if "wav" in audio.get("mimeType", ""):
        with open(path, "wb") as f:
            f.write(data)
    else:  # raw 16-bit PCM, e.g. "audio/L16;codec=pcm;rate=24000"
        rate = int(re.search(r"rate=(\d+)", audio.get("mimeType", "")).group(1)) if "rate=" in audio.get("mimeType", "") else 24000
        with wave.open(path, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(data)
    return path


def ask_groq(prompt, keys):
    _, resp = try_models("Groq", GROQ_LLM_MODELS, keys, lambda model, key: post(
        f"{GROQ_BASE}/chat/completions", retries=1, timeout=LLM_TIMEOUT, wait_on_429=False,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0.4, "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": prompt}]}))
    if resp.status_code != 200:
        raise api_error("Groq", resp)
    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except (KeyError, IndexError, ValueError, TypeError):
        raise ClipperError("Groq sent back an empty or unreadable answer.")


MAX_PARTS = 4


def clip_runs(c, count):
    """The [(first_id, last_id)] sentence runs of an AI pick, in play order: its
    "parts" (or an older single start_id/end_id). Runs that continue each other
    are merged; a run reusing sentences already in the clip is dropped."""
    raw = c.get("parts")
    if raw is None and "start_id" in c:
        raw = [[c.get("start_id"), c.get("end_id")]]
    runs = []
    for r in raw if isinstance(raw, list) else []:
        try:
            a, b = (int(r[0]), int(r[1])) if isinstance(r, (list, tuple)) else (int(r["start_id"]), int(r["end_id"]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not 0 <= a <= b < count:
            continue
        if runs and runs[-1][0] <= a <= runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], max(runs[-1][1], b))  # carries straight on: one part
        elif not any(a <= y and b >= x for x, y in runs):
            runs.append((a, b))
    return runs[:MAX_PARTS]


def run_span(sentences, a, b, duration, last):
    """A run's cut points: a breath before it, and after it a short one at a join
    or the full natural pause at the clip's end, never into a neighbouring sentence."""
    before = sentences[a - 1]["e"] if a > 0 else 0.0
    after = sentences[b + 1]["s"] if b + 1 < len(sentences) else float(duration)
    start = max(sentences[a]["s"] - 0.25, (before + sentences[a]["s"]) / 2, 0.0)
    end = min(sentences[b]["e"] + (0.7 if last else 0.3), (sentences[b]["e"] + after) / 2, float(duration))
    return round(start, 2), round(end, 2)


def clean_clips(raw, sentences, duration, n, floor=KEEP_MIN):
    """Turn the AI's picks (sentence-id runs) into cuts: whole sentences only,
    floor..KEEP_MAX seconds, no sentence in two clips, best n by score."""
    def score(c):
        try:
            return max(0, min(100, int(float(c.get("score", 0)))))
        except (TypeError, ValueError):
            return 0

    def spoken(runs):
        return sum(sentences[b]["e"] - sentences[a]["s"] for a, b in runs)

    picks = [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []
    out, used = [], set()
    for c in sorted(picks, key=score, reverse=True):
        runs = clip_runs(c, len(sentences))
        while runs and spoken(runs) > KEEP_MAX:  # too long: drop whole sentences from the end
            a, b = runs[-1]
            runs[-1:] = [(a, b - 1)] if b > a else []
        ids = {i for a, b in runs for i in range(a, b + 1)}
        if not runs or ids & used:
            continue
        parts = [dict(zip(("s", "e"), run_span(sentences, a, b, duration, k == len(runs) - 1)))
                 for k, (a, b) in enumerate(runs)]
        if not floor <= sum(x["e"] - x["s"] for x in parts) <= KEEP_MAX:
            continue
        used |= ids
        out.append({"start": min(x["s"] for x in parts), "end": max(x["e"] for x in parts),
                    "parts": parts, "ids": [list(r) for r in runs],
                    "message": str(c.get("message") or "").strip()[:300],
                    "title": str(c.get("title") or "Untitled clip").strip()[:100],
                    "hook": str(c.get("hook") or "").strip()[:300], "score": score(c),
                    "intro": clean_teaser(c, sentences, runs),
                    "voiceover": str(c.get("voiceover") or "").strip()[:160]})
        if len(out) == n:
            break
    return out


def clean_teaser(c, sentences, runs):
    """The cold open: one whole sentence from the clip's parts (not its first),
    cut short at 5 s if it runs long. None when the AI gave no usable one."""
    try:
        t = int(c["teaser_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if t == runs[0][0] or not any(a <= t <= b for a, b in runs):
        return None
    ts, te = sentences[t]["s"], sentences[t]["e"]
    te = min(te, ts + 5.0)
    if te - ts < 1.0:
        return None
    return {"start": round(max(ts - 0.1, 0.0), 2), "end": round(te + 0.2, 2)}


def replace_words(words, a, b, text):
    """Swap the words spoken between a and b for corrected caption text. Same
    word count keeps each word's timing; otherwise the new words share out the
    same stretch of time by length."""
    idx = [i for i, w in enumerate(words) if w["s"] >= a - 0.05 and w["e"] <= b + 0.05]
    new = text.split()
    if not idx or [words[i]["w"] for i in idx] == new:
        return words
    old = [words[i] for i in idx]
    if len(new) == len(old):
        timed = [{**w, "w": t} for w, t in zip(old, new)]
    else:
        t0, t1 = old[0]["s"], old[-1]["e"]
        total, pos, timed = sum(len(t) + 1 for t in new), t0, []
        for t in new:
            d = (t1 - t0) * (len(t) + 1) / total
            timed.append({"w": t, "s": round(pos, 3), "e": round(pos + d * 0.9, 3)})
            pos += d
    return words[:idx[0]] + timed + words[idx[-1] + 1:]


def answer_list(answer, key):
    """The list an AI answer holds under key. Models sometimes skip the wrapper
    object and send the bare list, so accept that too."""
    if isinstance(answer, dict):
        answer = answer.get(key)
    return answer if isinstance(answer, list) else []


def describe_drafts(clips, sentences):
    """Drafts as the reviewer reads them: settings, then the words heard per part."""
    lines = []
    for k, c in enumerate(clips, 1):
        secs = sum(x["e"] - x["s"] for x in c["parts"])
        teaser = next((i for a, b in c["ids"] for i in range(a, b + 1)
                       if c["intro"] and sentences[i]["s"] <= c["intro"]["start"] + 0.2 <= sentences[i]["e"]), None)
        lines.append(f"{k}. message: {c['message']} | parts: {json.dumps(c['ids'])} ({secs:.0f} s) | "
                     f"title: {c['title']} | hook: {c['hook']} | score: {c['score']} | "
                     f"teaser_id: {teaser} | voiceover: {c['voiceover']}")
        for a, b in c["ids"]:
            lines.append(f"   [{a}-{b}] " + " ".join(sentences[i]["text"] for i in range(a, b + 1)))
    return "\n".join(lines)


def pick_moments(sentences, duration, n, keys, progress):
    """Draft clips, then have the same AI check each one as a first-time viewer
    and fix any that don't deliver a complete message. Gemini first, Groq if
    Gemini is missing or fails. Returns (clips, ai_name)."""
    transcript = "\n".join(f"{i} | {s['s']:.0f}s | {s['text']}" for i, s in enumerate(sentences))
    rules = {"n": n, "lo": MIN_CLIP, "hi": MAX_CLIP, "max_parts": MAX_PARTS, "transcript": transcript}
    errors = []
    gemini = {"gemini": keys.get("gemini"), "vertex": keys.get("vertex")}
    for name, ask, key in (("Gemini", ask_gemini, gemini if any(gemini.values()) else None),
                           ("Groq", ask_groq, keys.get("groq"))):
        if not key:
            continue
        progress(f"{name} is finding the best moments")
        try:
            drafts = clean_clips(answer_list(ask(PROMPT.format(duration=duration, **rules), key), "clips"),
                                 sentences, duration, n)
        except ClipperError as e:
            errors.append(str(e))
            continue
        if not drafts:
            errors.append(f"{name} didn't find a usable moment.")
            continue
        progress(f"{name} is checking that each clip tells a complete story")
        try:  # short drafts get here too: the review grows them to MIN_CLIP
            reviewed = clean_clips(answer_list(ask(REVIEW_PROMPT.format(
                drafts=describe_drafts(drafts, sentences), **rules), key), "clips"),
                sentences, duration, n, floor=MIN_CLIP)
        except ClipperError:
            reviewed = []
        long_drafts = [c for c in drafts if sum(x["e"] - x["s"] for x in c["parts"]) >= MIN_CLIP]
        # A review that lost clips along the way isn't an improvement.
        clips = reviewed if len(reviewed) >= max(1, len(long_drafts) - 1) else long_drafts
        if not clips:
            errors.append(f"{name} didn't find a moment of at least {MIN_CLIP} seconds.")
            continue
        if name == "Gemini" and last_model.get("model"):
            name = f"{last_model['model']}{' (Google Cloud)' if last_model['cloud'] else ''}"
        return clips, name
    raise ClipperError(" ".join(errors) or "Add a Groq or Gemini key in Keys first.")


# --- visual & action moment detection (non-verbal / silent) -------------------

VISUAL_MOMENTS_PROMPT = """You are an expert short-form video editor choosing viral clips from video frames.
Below are sequential frames from candidate moments of an action, sports, gameplay, workout, tutorial or silent video.
For each candidate moment #{idx} (from {start:.1f}s to {end:.1f}s):
1. "title": catchy, viral hook title (max 8 words).
2. "hook_line": curiosity gap question or bold teaser (max 8 words) burned on screen for the first 3 seconds (e.g. "Wait until the very end", "How did they survive this?", "The cleanest play you'll see").
3. "score": action/interest rating (0 to 100) based on visual movement, intensity, and viewer entertainment value.

Answer with JSON only:
{"clips": [
  {"id": 1, "title": "Insane Comeback in Final Seconds", "hook_line": "Watch how close this was", "score": 92}
]}"""


def scan_motion_energy(src, duration, fps=2):
    """Scan video motion intensity at low resolution (160x90). Returns [(t, score)]."""
    import cv2
    import numpy as np
    sw, sh = 160, 90
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
           "-i", src, "-an", "-vf", f"fps={fps},scale={sw}:{sh}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = sw * sh
    prev, scores, t = None, [], 0.0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(sh, sw)
            if prev is not None:
                diff = float(np.mean(cv2.absdiff(prev, frame)))
                scores.append((round(t, 2), diff))
            prev = frame
            t += 1.0 / fps
    finally:
        proc.stdout.close()
        proc.wait()
    return scores


def extract_window_frames(src, start, end, count=4):
    """Sample small JPEG frames across [start, end] for visual evaluation."""
    step = (end - start) / (count + 1)
    frames = []
    for k in range(1, count + 1):
        t = start + k * step
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
               "-i", src, "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "5",
               "-f", "image2", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0 and len(proc.stdout) > 500:
            frames.append(proc.stdout)
    return frames


def find_visual_windows(src, duration, n=4, win_len=45):
    """Find the highest-motion candidate windows snapped to scene cuts."""
    cuts = scene_cuts(src, 0, duration)
    motion_scores = scan_motion_energy(src, duration)
    if not motion_scores:
        step = max(MIN_CLIP, duration / max(1, n))
        return [{"start": round(i * step, 2), "end": round(min(duration, (i + 1) * step), 2), "score": 50}
                for i in range(n)]

    windows = []
    step = 10.0
    t = 0.0
    while t + MIN_CLIP <= duration:
        w_end = min(duration, t + win_len)
        win_motion = [m for ts, m in motion_scores if t <= ts <= w_end]
        win_cuts = [c for c in cuts if t <= c <= w_end]
        avg_motion = sum(win_motion) / max(1, len(win_motion))
        score = avg_motion * (1.0 + 0.15 * min(6, len(win_cuts)))
        windows.append({"start": t, "end": w_end, "score": score, "cuts": win_cuts})
        t += step

    windows.sort(key=lambda w: w["score"], reverse=True)

    picked = []
    for w in windows:
        overlap = any(not (w["end"] <= p["start"] + 15 or w["start"] >= p["end"] - 15) for p in picked)
        if not overlap:
            picked.append(w)
            if len(picked) >= n * 2:
                break

    results = []
    for w in picked:
        s, e = w["start"], w["end"]
        near_s = [c for c in cuts if abs(c - s) <= 5.0]
        if near_s:
            s = near_s[0]
        near_e = [c for c in cuts if abs(c - e) <= 5.0]
        if near_e:
            e = near_e[-1]
        s = max(0.0, s)
        e = min(duration, e)
        if e - s < MIN_CLIP:
            e = min(duration, s + MIN_CLIP)
            if e - s < MIN_CLIP and s > 0:
                s = max(0.0, e - MIN_CLIP)
        if e - s > MAX_CLIP:
            e = s + MAX_CLIP
        results.append({"start": round(s, 2), "end": round(e, 2), "score": round(w["score"], 2)})

    results.sort(key=lambda x: x["start"])
    return results[:n]


def pick_visual_moments(src, duration, n, keys, progress=None):
    """Pick action & visual highlights without requiring speech; returns (clips, model_name)."""
    if progress:
        progress("Scanning video motion & visual scene cuts", 0.1)
    candidates = find_visual_windows(src, duration, n=n)
    if not candidates:
        return [], "none"

    if progress:
        progress("Sampling candidate keyframes for visual evaluation", 0.4)
    for c in candidates:
        c["frames"] = extract_window_frames(src, c["start"], c["end"], count=3)

    clips_meta = {}
    if keys and any(split_keys(keys)):
        if progress:
            progress("Evaluating visual hooks with Gemini Vision", 0.6)
        import base64
        parts = [{"text": VISUAL_MOMENTS_PROMPT}]
        for idx, c in enumerate(candidates, 1):
            parts.append({"text": f"\nCandidate moment #{idx} (from {c['start']:.1f}s to {c['end']:.1f}s):"})
            for img in c.get("frames", []):
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(img).decode()}})
        try:
            resp = gemini_json(parts, keys, 0.2, VERTEX_VISION_MODELS)
            for item in answer_list(resp, "clips"):
                if isinstance(item, dict) and "id" in item:
                    clips_meta[int(item["id"])] = item
        except Exception as e:
            log.warning("Gemini visual evaluation failed (%s); using motion scores", e)

    out_clips = []
    for idx, c in enumerate(candidates, 1):
        meta = clips_meta.get(idx, {})
        title = meta.get("title") or f"Action Highlight {idx}"
        hook_line = meta.get("hook_line") or title
        score = int(meta.get("score") or min(95, max(60, round(c["score"] * 6))))
        start, end = c["start"], c["end"]
        out_clips.append({
            "i": idx,
            "title": title,
            "score": score,
            "start": start,
            "end": end,
            "length": round(end - start),
            "parts": [{"s": start, "e": end}],
            "intro": {"start": start, "end": round(min(start + 3.2, end), 2)},
            "voiceover": hook_line,
            "visual": True,
            "framing": "auto"
        })

    return out_clips, "Gemini Vision" if clips_meta else "Motion Scanner"


# --- hook voice clean-up -------------------------------------------------------

def _norm(word):
    return re.sub(r"[^\w]", "", word.lower())


def trim_spoken_prefix(path, text, groq_keys):
    """Cut anything the voice says before the hook line itself. Puter wraps the
    text for Gemini TTS ("Say the following text aloud. ...") and the model can
    read that wrapper out loud. Returns the (possibly new) audio path; a clean
    voice, or any failure, leaves the file as it was."""
    with open(path, "rb") as f:
        audio = f.read()
    resp = post_rotating(f"{GROQ_BASE}/audio/transcriptions", groq_keys, lambda key: dict(
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (os.path.basename(path), audio, "application/octet-stream")},
                data={"model": WHISPER_MODEL, "response_format": "verbose_json",
                      "timestamp_granularities[]": ["word"]}), rounds=0)
    if resp.status_code != 200:
        return path
    heard = [(_norm(w["word"]), float(w["start"]), float(w["end"])) for w in resp.json().get("words") or []]
    start = spoken_start(heard, text)
    if start is None:
        return path
    out = os.path.splitext(path)[0] + ".trim.wav"
    run_ffmpeg(["-ss", f"{start:.3f}", "-i", path, out])
    final = os.path.splitext(path)[0] + ".wav"
    os.replace(out, final)
    if final != path:
        os.remove(path)
    return final


def spoken_start(heard, text):
    """Where the hook line begins in a transcribed voice ([(word, start, end)]),
    or None when nothing comes before it."""
    target = [w for w in (_norm(x) for x in text.split()) if w][:2]
    for k in range(len(heard)):
        if target and [w for w, _, _ in heard[k:k + len(target)]] == target:
            return max(0.0, heard[k][1] - 0.08) if k > 0 and heard[k][1] > 0.3 else None
    # Line not recognised word for word: fall back to cutting after "...aloud".
    for k, (w, _, end) in enumerate(heard[:10]):
        if w == "aloud" and k + 1 < len(heard):
            return max(end, heard[k + 1][1] - 0.08)
    return None


# --- thumbnails (Nano Banana) --------------------------------------------------

THUMB_PROMPT = """Design a scroll-stopping vertical 9:16 cover image (thumbnail) for a short video.
Build it from these frames of the video: keep the real person or subject recognisable and make them the main visual. Don't invent a different person.
Put this title on it in big, bold, highly readable text: "{title}"
High contrast, clean composition, room for the text, no watermarks, no borders, no other text.
What the video is about: {message}"""


def make_thumbnail(frames, title, message, keys, out_path):
    """A 1080x1920 cover from video frames (JPEG bytes) with Nano Banana; returns out_path."""
    import base64
    parts = [{"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(f).decode()}} for f in frames]
    parts.append({"text": THUMB_PROMPT.format(title=title, message=message or title)})
    config = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "9:16", "imageSize": "1K"}}
    try:
        resp = gemini_request({"contents": [{"role": "user", "parts": parts}], "generationConfig": config},
                              keys, THUMB_MODELS, THUMB_MODELS, "Nano Banana", timeout=VERTEX_TIMEOUT)
    except ClipperError as e:
        if "imageConfig" not in str(e) and "image_config" not in str(e):
            raise
        config.pop("imageConfig")  # older API shape: the prompt already asks for 9:16, cropping fixes the rest
        resp = gemini_request({"contents": [{"role": "user", "parts": parts}], "generationConfig": config},
                              keys, THUMB_MODELS, THUMB_MODELS, "Nano Banana", timeout=VERTEX_TIMEOUT)
    try:
        image = next(x["inlineData"] for x in resp.json()["candidates"][0]["content"]["parts"]
                     if x.get("inlineData", {}).get("mimeType", "").startswith("image/"))
    except (KeyError, IndexError, StopIteration, ValueError):
        raise ClipperError("Nano Banana sent back no image.")
    raw = out_path + ".raw"
    with open(raw, "wb") as f:
        f.write(base64.b64decode(image["data"]))
    try:  # exactly 1080x1920, whatever size came back
        run_ffmpeg(["-i", raw, "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                    "-frames:v", "1", "-q:v", "2", out_path])
    finally:
        os.remove(raw)
    return out_path


# --- captions and rendering -------------------------------------------------

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,2,90,90,560,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
# Big hook line at the top during the cold open, on an accent-coloured box (BGR #FF5A36).
HOOK_STYLE = ("Style: Hook,Arial,76,&H00FFFFFF,&H00FFFFFF,&H00365AFF,&H00000000,"
              "-1,0,0,0,100,100,0,0,3,18,0,8,80,80,300,1\n")
HIGHLIGHT = r"{\fscx112\fscy112\c&H00D7FF&}"  # Punchy kinetic word pop (112% scale) + gold highlight (#FFD700)
NORMAL = r"{\fscx100\fscy100\c&HFFFFFF&}"


def ass_time(t):
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def caption_groups(words, max_words=3, max_gap=0.6):
    """Short on-screen phrases: break after max_words, a pause or punctuation."""
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["s"] - cur[-1]["e"] > max_gap or cur[-1]["w"][-1:] in ".?!,"):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words, start, end):
    """Word-by-word captions for one clip: the phrase stays up, the spoken word lights up."""
    inside = [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.05]
    groups = caption_groups(inside)
    lines = [ASS_HEADER]
    for gi, group in enumerate(groups):
        group_end = min(group[-1]["e"] + 0.4, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end)
        clean = [ass_text(w["w"]) for w in group]
        for i, w in enumerate(group):
            t0 = group[0]["s"] if i == 0 else w["s"]
            t1 = group[i + 1]["s"] if i + 1 < len(group) else group_end
            text = " ".join(f"{HIGHLIGHT}{c}{NORMAL}" if j == i else c for j, c in enumerate(clean))
            lines.append(f"Dialogue: 0,{ass_time(t0 - start)},{ass_time(t1 - start)},Caption,,0,0,0,,{text}\n")
    return "".join(lines)


def ass_text(text):
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def build_hook_ass(text, duration):
    header = ASS_HEADER.replace("\n\n[Events]", "\n" + HOOK_STYLE + "\n[Events]")
    return header + f"Dialogue: 0,{ass_time(0)},{ass_time(duration)},Hook,,0,0,0,,{{\\fad(200,200)}}{ass_text(text)}\n"


# --- framing: keep the person or subject in the vertical frame ---------------
#
# Per shot (scene cut to scene cut) the framing is picked automatically:
#   faces found            -> the crop follows the main face (on-device, OpenCV YuNet)
#   no face, Gemini key    -> Gemini says where the subject is, or that the shot is a
#                             screen / wide scene, which then shows the whole frame
#   otherwise              -> centre crop
# ponytail: follows the biggest, most central face, not whoever is speaking;
# active-speaker detection (mouth movement) is the upgrade if two-person shots look wrong.

LAYOUTS = ("auto", "fill", "fit")
SAMPLE_FPS = 4
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
FILL = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1"


def fit_chain(tag):
    """Whole frame on a blurred copy of itself: nothing cut off (screens, slides, wide shots).
    Downscales the background layer before blurring for ~5x faster rendering."""
    return (f"split[{tag}a][{tag}b];[{tag}a]scale=270:480:force_original_aspect_ratio=increase,"
            f"crop=270:480,boxblur=8:2,scale=1080:1920:flags=bilinear[{tag}bg];[{tag}b]scale=1080:1920:"
            f"force_original_aspect_ratio=decrease[{tag}fg];[{tag}bg][{tag}fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def video_size(path):
    """(width, height) as displayed, so phone videos with a rotation flag come out right."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", proc.stderr)
    if not m:
        raise ClipperError("Couldn't read the video's size.")
    w, h = int(m.group(1)), int(m.group(2))
    return (h, w) if re.search(r"rotation of -?(90|270)", proc.stderr) else (w, h)


def scene_cuts(src, start, end):
    """Absolute times of hard cuts between start and end (ffmpeg scene score)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                           "-i", src, "-an", "-vf", "scale=320:-2,select='gt(scene,0.3)',showinfo",
                           "-f", "null", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    cuts = []
    for t in (float(x) for x in re.findall(r"pts_time:([\d.]+)", proc.stderr)):
        if t > 0.5 and (not cuts or start + t - cuts[-1] > 1.0):  # ignore flicker
            cuts.append(start + t)
    return cuts


_detector = None


def face_detector(width, height):
    """YuNet, downloaded once (~230 KB) to tools/. None if it can't be had:
    framing then falls back to Gemini or the centre."""
    global _detector
    import cv2
    path = os.path.join(TOOLS_DIR, "face_detection_yunet_2023mar.onnx")
    try:
        if not os.path.exists(path):
            os.makedirs(TOOLS_DIR, exist_ok=True)
            data = httpx.get(YUNET_URL, follow_redirects=True, timeout=60).raise_for_status().content
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
        if _detector is None:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide harmless backend notices
            _detector = cv2.FaceDetectorYN.create(path, "", (width, height), 0.7)
        _detector.setInputSize((width, height))
        return _detector
    except Exception as e:
        log.warning("Face detection unavailable: %s", e)
        return None


def main_face(faces, width, previous):
    """Centre x (0-1) of the face to follow: big and central wins, but stay on
    the previous person unless someone clearly bigger appears. None if no usable face."""
    best, best_score = None, 0.0
    for f in faces if faces is not None else []:
        x, w, h = float(f[0]), float(f[2]), float(f[3])
        if w < 0.04 * width:  # crowd / background faces
            continue
        cx = (x + w / 2) / width
        score = w * h * (1 - 0.5 * abs(cx - 0.5))
        if previous is not None and abs(cx - previous) < 0.1:
            score *= 1.6
        if score > best_score:
            best, best_score = cx, score
    return best


def sample_frames(src, start, end, width, height):
    """Small BGR frames at SAMPLE_FPS: yields (absolute_time, frame)."""
    import numpy as np
    sw, sh = 640, max(2, int(round(640 * height / width / 2)) * 2)
    proc = subprocess.Popen([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
                             "-t", f"{end - start:.3f}", "-i", src, "-an",
                             "-vf", f"fps={SAMPLE_FPS},scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    size, k = sw * sh * 3, 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start + k / SAMPLE_FPS, np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


def steady_camera(hits):
    """Turn jittery face positions into calm camera moves: ignore small wobble,
    ease toward real moves. hits: [(t, x)] -> keyframes [(t, x)]."""
    keys, cam = [], None
    for t, x in hits:
        if cam is None:
            cam = x
        elif abs(x - cam) > 0.04:
            cam += (x - cam) * 0.35
        keys.append((t, round(cam, 4)))
    return keys


SUBJECT_PROMPT = """These are frames from consecutive shots of a video that will be cropped to a vertical 9:16 phone format.
For each frame, in order, say what the crop should follow:
- "subject": a clear main subject (an animal, object, product, person seen from behind...). Give "x", its horizontal centre from 0 (left edge) to 1 (right edge).
- "screen": a screen recording, slides, a document or text that must stay readable.
- "wide": a wide scene where the whole frame matters and no single subject stands out.
Answer with JSON only: {"shots": [{"kind": "subject", "x": 0.62}, {"kind": "screen"}]}"""


def ask_gemini_subjects(jpegs, key):
    """One Gemini call for all faceless shots of a clip: [{"kind", "x"}] or None."""
    import base64
    parts = [{"text": SUBJECT_PROMPT}] + [
        {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(j).decode()}} for j in jpegs]
    try:
        shots = answer_list(gemini_json(parts, key, 0, VERTEX_VISION_MODELS), "shots")
    except (ClipperError, httpx.HTTPError):
        return None  # those shots get a centre crop
    return shots if len(shots) == len(jpegs) else None


def plan_framing(src, start, end, gemini_key=None):
    """How to frame each shot between start and end, or None when the source is
    already vertical-ish (a plain fill crop is right then)."""
    import cv2
    width, height = video_size(src)
    scaled_w = int(round(width * 1920 / height / 2)) * 2
    if scaled_w < 1080 + 40:
        return None
    bounds = [start, *scene_cuts(src, start, end), end]
    shots = [{"s": round(a, 3), "e": round(b, 3), "hits": [], "n": 0, "frame": None}
             for a, b in zip(bounds, bounds[1:])]
    detector, previous = None, None
    for t, frame in sample_frames(src, start, end, width, height):
        shot = next((s for s in shots if s["s"] <= t < s["e"]), shots[-1])
        if shot["n"] == 0:
            previous = None  # new shot: free to pick a new person
        shot["n"] += 1
        if shot["frame"] is None and t >= (shot["s"] + shot["e"]) / 2:
            shot["frame"] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        detector = detector or face_detector(frame.shape[1], frame.shape[0])
        if detector is not None:
            x = main_face(detector.detect(frame)[1], frame.shape[1], previous)
            if x is not None:
                shot["hits"].append((t, x))
                previous = x

    faceless = []
    for shot in shots:
        if shot["n"] and len(shot["hits"]) >= 0.4 * shot["n"]:
            shot.update(mode="face", keys=steady_camera(shot["hits"]))
        else:
            shot.update(mode="centre", x=0.5)
            if shot["frame"] is not None:
                faceless.append(shot)
    if faceless and gemini_key:
        answers = ask_gemini_subjects([s["frame"] for s in faceless[:8]], gemini_key) or []
        for shot, a in zip(faceless, answers):
            kind = a.get("kind") if isinstance(a, dict) else None
            if kind in ("screen", "wide"):
                shot["mode"] = "fit"
            elif kind == "subject":
                try:
                    shot.update(mode="subject", x=max(0.0, min(1.0, float(a["x"]))))
                except (KeyError, TypeError, ValueError):
                    pass
    for shot in shots:
        for k in ("hits", "n", "frame"):
            shot.pop(k)
    return {"scaled_w": scaled_w, "shots": shots}


def plan_parts(src, parts, gemini_key=None):
    """plan_framing for each part (they can be far apart), merged into one plan;
    shots keep absolute times, so crop_x/framing work across all of them."""
    merged = None
    for x in parts:
        plan = plan_framing(src, x["s"], x["e"], gemini_key)
        if plan is None:
            return None
        merged = plan if merged is None else {**merged, "shots": merged["shots"] + plan["shots"]}
    return merged


def describe_framing(plan):
    """Short label for the clip card."""
    modes = {s["mode"] for s in plan["shots"]} if plan else set()
    labels = [label for mode, label in (("face", "follows the face"), ("subject", "follows the subject"),
                                        ("fit", "shows the whole frame where needed")) if mode in modes]
    return ", ".join(labels) or "centred"


def crop_x(plan, t):
    """Left edge of the 1080 px crop at absolute time t, in scaled pixels."""
    shot = next((s for s in plan["shots"] if s["s"] <= t < s["e"]), plan["shots"][-1])
    if shot["mode"] == "face":
        keys = shot["keys"]
        if t <= keys[0][0]:
            x = keys[0][1]
        elif t >= keys[-1][0]:
            x = keys[-1][1]
        else:
            i = next(i for i in range(1, len(keys)) if keys[i][0] >= t)
            (t0, x0), (t1, x1) = keys[i - 1], keys[i]
            x = x0 + (x1 - x0) * (t - t0) / (t1 - t0)
    else:
        x = shot["x"] if shot["mode"] == "subject" else 0.5
    return int(max(0, min(plan["scaled_w"] - 1080, round(x * plan["scaled_w"] - 540))))


def framing(inp, layout, plan, t0, dur, out_dir, tag, cmd_name):
    """Filter chain turning input inp into a 1080x1920 frame labelled [tag].
    The crop moves through sendcmd keyframes (cmd_name, written to out_dir)."""
    if layout == "fit":
        return f"{inp}{fit_chain(tag + 'f')}[{tag}]"
    if not plan:
        return f"{inp}{FILL}[{tag}]"
    lines, last = [], None
    for k in range(int(dur * 15) + 1):
        x = crop_x(plan, t0 + k / 15)
        if x != last:
            lines.append(f"{k / 15:.3f} crop@{tag} x {x};")
            last = x
    with open(os.path.join(out_dir, cmd_name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    track = (f"scale={plan['scaled_w']}:1920,sendcmd=f={cmd_name},"
             f"crop@{tag}=1080:1920:{crop_x(plan, t0)}:0,setsar=1")
    fits = [s for s in plan["shots"] if s["mode"] == "fit" and s["e"] > t0 and s["s"] < t0 + dur]
    if layout != "auto" or not fits:
        return f"{inp}{track}[{tag}]"
    enable = "+".join(f"between(t,{max(s['s'] - t0, 0):.3f},{min(s['e'] - t0, dur):.3f})" for s in fits)
    return (f"{inp}split[{tag}1][{tag}2];[{tag}1]{track}[{tag}t];[{tag}2]{fit_chain(tag + 'f')}[{tag}f];"
            f"[{tag}t][{tag}f]overlay=enable='{enable}'[{tag}]")


WHOOSH = 0.6  # seconds of transition sound at the end of the hook
SAME = "aresample=48000,aformat=channel_layouts=stereo"  # concat and amix need matching audio
CAPTION_STYLE = ("Style: Caption,Arial,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
                 "-1,0,0,0,100,100,0,0,1,7,3,2,90,90,560,1")
CAPTION_Y = round(1 - 560 / 1920, 4)  # where captions sit by default (their bottom edge, 0-1 from the top)
TEXT_STYLES = ("Style: Text,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
               "-1,0,0,0,100,100,0,0,1,6,2,5,40,40,0,1\n"
               "Style: TextBox,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00365AFF,&H00000000,"
               "-1,0,0,0,100,100,0,0,3,16,0,5,40,40,0,1")

# Built-in sound effects, made by ffmpeg the first time they're used: (source, filters).
SFX = {
    "whoosh": ("anoisesrc=r=48000:c=pink:a=0.6:d=0.7",
               "highpass=f=300,lowpass=f=7000,afade=t=in:d=0.5:curve=exp,afade=t=out:st=0.5:d=0.2,volume=2"),
    "pop": ("sine=f=520:r=48000:d=0.15", "afade=t=in:d=0.005,afade=t=out:st=0.02:d=0.13:curve=exp,volume=1.5"),
    "ding": ("sine=f=1318:r=48000:d=1.2", "afade=t=out:st=0.02:d=1.15:curve=exp,volume=0.8"),
    "boom": ("sine=f=55:r=48000:d=0.9", "afade=t=out:st=0.05:d=0.85:curve=exp,volume=3"),
    "click": ("anoisesrc=r=48000:c=white:a=0.8:d=0.04", "highpass=f=2500,afade=t=out:d=0.04"),
}


def sfx_path(name):
    path = os.path.join(TOOLS_DIR, "sfx", f"{name}.wav")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        source, filters = SFX[name]
        run_ffmpeg(["-f", "lavfi", "-i", source, "-af", filters, "-ac", "2", path])
    return path


def sound_path(file, out_dir):
    """'sfx:<name>' for a built-in effect, else a file uploaded into the job folder."""
    return sfx_path(file[4:]) if file.startswith("sfx:") else os.path.join(out_dir, file)


def build_edit_ass(words, total, captions, texts):
    """Captions (word by word, styled and placed as chosen) plus free text boxes."""
    size = int(captions.get("size", 86))
    margin = int(round((1 - captions.get("y", CAPTION_Y)) * 1920))
    styles = (f"Style: Caption,Arial,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
              f"-1,0,0,0,100,100,0,0,1,{max(3, size // 12)},3,2,90,90,{margin},1\n{TEXT_STYLES}")
    ass = build_ass(words if captions.get("on", True) else [], 0, total).replace(CAPTION_STYLE, styles)
    for t in texts:
        rgb = t.get("color", "#ffffff").lstrip("#")
        body = ass_text(t["text"]).replace("\n", "\\N")
        ass += (f"Dialogue: 1,{ass_time(t['t0'])},{ass_time(t['t1'])},{'TextBox' if t.get('box') else 'Text'},,0,0,0,,"
                f"{{\\pos({round(t['x'] * 1080)},{round(t['y'] * 1920)})\\fs{int(t['size'])}"
                f"\\c&H{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}&}}{body}\n")
    return ass


def cover_chain(size, frame):
    """Fill the 1080x1920 frame (zoomed in by frame['zoom']) and crop around the
    chosen point; x/y say where the frame's centre sits, 0-1 across the picture."""
    w, h = size
    zoom = min(3.0, max(1.0, float(frame.get("zoom", 1))))
    scale = max(1080 / w, 1920 / h) * zoom
    cw, ch = int(round(w * scale / 2)) * 2, int(round(h * scale / 2)) * 2
    px = int(max(0, min(cw - 1080, round(frame.get("x", 0.5) * cw - 540))))
    py = int(max(0, min(ch - 1920, round(frame.get("y", 0.5) * ch - 960))))
    return f"scale={cw}:{ch},crop=1080:1920:{px}:{py},setsar=1"


def part_video(inp, frame, size, plan, t0, dur, out_dir, tag, cmd_name):
    """One part's picture as a 1080x1920 stream labelled [tag]: auto (follows the
    person, needs plan), fit (whole frame), fill/manual (fixed crop)."""
    mode = frame.get("mode", "fill")
    if mode == "fit":
        return f"{inp}{fit_chain(tag + 'f')}[{tag}]"
    if mode == "auto" and plan:
        return framing(inp, "auto", plan, t0, dur, out_dir, tag, cmd_name)
    return f"{inp}{cover_chain(size, frame if mode == 'manual' else {})}[{tag}]"


def render_edit(src, edl, words, out_dir, name, intro=None, voice=None, plan=None, out_name=None):
    """Render an edit to <out_name or name>.mp4 in out_dir and return the file name.

    edl: {"segments": [{"s", "e", "frame", "volume"}], "texts": [...], "sounds": [...],
          "captions": {"on", "y", "size"}, "hook": {"line"}}
    The parts play back to back, each with its own framing and volume; captions
    and text boxes go on top, sound effects are mixed in. With intro (a teaser
    span of the source) the clip opens cold: that moment muted, the hook line on
    screen and the narrator (voice, any audio file) over it, then a whoosh."""
    src = os.path.abspath(src)
    size = video_size(src)
    parts = edl["segments"]
    args, graph = [], []
    out_file = (out_name or name) + ".mp4"

    def add_input(*a):
        args.extend(a)
        return sum(1 for x in args if x == "-i") - 1

    total, mapped = 0.0, []
    has_src_audio = media_info(src)[1]
    for k, seg in enumerate(parts):
        d = seg["e"] - seg["s"]
        i = add_input("-ss", f"{seg['s']:.3f}", "-t", f"{d:.3f}", "-i", src)
        graph.append(part_video(f"[{i}:v]", seg["frame"], size, plan, seg["s"], d, out_dir, f"p{k}x", f"{name}_p{k}.cmd"))
        if has_src_audio:
            graph.append(f"[{i}:a]volume={float(seg.get('volume', 1)):.2f},{SAME}[a{k}]")
        else:
            sil_idx = add_input("-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo")
            graph.append(f"[{sil_idx}:a]{SAME}[a{k}]")
        mapped += [{**w, "s": w["s"] - seg["s"] + total, "e": w["e"] - seg["s"] + total}
                   for w in words if w["s"] >= seg["s"] - 0.05 and w["e"] <= seg["e"] + 0.05]
        total += d
    graph.append("".join(f"[p{k}x][a{k}]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=1[vc][ac]")
    with open(os.path.join(out_dir, name + ".ass"), "w", encoding="utf-8") as f:
        f.write(build_edit_ass(mapped, total, edl.get("captions") or {}, edl.get("texts") or []))
    graph.append(f"[vc]subtitles={name}.ass[mv]")  # relative: cwd=out_dir dodges Windows path escaping

    duck_streams, sfx_streams = [], []
    for j, snd in enumerate(edl.get("sounds") or []):
        i = add_input("-i", sound_path(snd["file"], out_dir))
        tag = f"s{j}"
        graph.append(f"[{i}:a]{SAME},volume={float(snd.get('volume', 1)):.2f},adelay={snd['t0'] * 1000:.0f}:all=1[{tag}]")
        file_name = str(snd.get("file", ""))
        dur = float(snd.get("dur") or 0.0)
        # Background music / ambient audio (> 5s and not built-in SFX) ducks under the speaker's voice
        should_duck = snd.get("duck") is True or (snd.get("duck") is None and dur >= 5.0 and not file_name.startswith("sfx:"))
        if should_duck:
            duck_streams.append(f"[{tag}]")
        else:
            sfx_streams.append(f"[{tag}]")

    if duck_streams:
        if len(duck_streams) > 1:
            graph.append(f"{''.join(duck_streams)}amix=inputs={len(duck_streams)}:duration=first:normalize=0[mpre]")
            music_in = "[mpre]"
        else:
            music_in = duck_streams[0]
        # Duck the background music under the spoken voice [ac]
        graph.append(f"{music_in}[ac]sidechaincompress=threshold=0.10:ratio=4:attack=50:release=450[ducked]")
        final_inputs = ["[ac]", "[ducked]", *sfx_streams]
        graph.append(f"{''.join(final_inputs)}amix=inputs={len(final_inputs)}:duration=first:normalize=0[ma]")
    elif sfx_streams:
        final_inputs = ["[ac]", *sfx_streams]
        graph.append(f"{''.join(final_inputs)}amix=inputs={len(final_inputs)}:duration=first:normalize=0[ma]")
    else:
        graph.append("[ac]anull[ma]")

    if not intro:
        maps = ["-map", "[mv]", "-map", "[ma]"]
    else:
        length = intro["end"] - intro["start"]
        if voice:
            length = max(length, media_info(voice)[0] + 0.5)  # let the narrator finish
        i = add_input("-ss", f"{intro['start']:.3f}", "-t", f"{length:.3f}", "-i", src)
        frame = next((s["frame"] for s in parts if s["s"] <= intro["start"] < s["e"]), parts[0]["frame"])
        with open(os.path.join(out_dir, name + "_hook.ass"), "w", encoding="utf-8") as f:
            f.write(build_hook_ass((edl.get("hook") or {}).get("line") or edl.get("title") or "", length))
        graph.append(part_video(f"[{i}:v]", frame, size, plan, intro["start"], length, out_dir, "hx", f"{name}_intro.cmd")
                     + f";[hx]subtitles={name}_hook.ass[iv]")
        # The teaser's own sound is dropped: only the narrator, or silence.
        if voice:
            i = add_input("-i", voice)
            graph.append(f"[{i}:a]{SAME},adelay=150:all=1,volume=1.4,apad,atrim=0:{length:.3f}[bed]")
        else:
            i = add_input("-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo")
            graph.append(f"[{i}:a]{SAME}[bed]")
        # Whoosh: filtered noise swelling into the cut, the cue that the clip starts.
        i = add_input("-f", "lavfi", "-t", f"{WHOOSH:.2f}", "-i", "anoisesrc=r=48000:c=pink:a=0.5")
        graph.append(f"[{i}:a]highpass=f=300,lowpass=f=7000,afade=t=in:d={WHOOSH - 0.15:.2f}:curve=exp,"
                     f"afade=t=out:st={WHOOSH - 0.15:.2f}:d=0.15,volume=2.0,{SAME},"
                     f"adelay={max(length - WHOOSH, 0) * 1000:.0f}:all=1[wh]")
        graph.append("[bed][wh]amix=inputs=2:duration=first:normalize=0[ia]")
        graph.append("[iv][ia][mv][ma]concat=n=2:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]"]

    enc_args = video_encoder_args()
    try:
        run_ffmpeg([*args, "-filter_complex", ";".join(graph), *maps,
                    *enc_args,
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out_file],
                   cwd=out_dir)
    except ClipperError:
        # Fall back to software CPU libx264 if hardware encoder encounters any issue
        if "-c:v" in enc_args and "libx264" not in enc_args:
            log.warning("Hardware video encoder failed; falling back to libx264 CPU encoder")
            run_ffmpeg([*args, "-filter_complex", ";".join(graph), *maps,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out_file],
                       cwd=out_dir)
        else:
            raise
    return out_file


def clip_edl(clip, layout, captions=True):
    """The edit a freshly made clip starts from: its parts, as the AI cut them."""
    return {"title": clip["title"],
            "segments": [{"s": x["s"], "e": x["e"], "volume": 1.0,
                          "frame": {"mode": layout if layout in ("auto", "fit") else "fill", "x": 0.5, "y": 0.5, "zoom": 1.0}}
                         for x in clip.get("parts") or [{"s": clip["start"], "e": clip["end"]}]],
            "texts": [], "sounds": [],
            "captions": {"on": bool(captions), "y": CAPTION_Y, "size": 86},
            "hook": {"on": bool(clip.get("intro")), "line": clip.get("voiceover") or clip["title"]}}


def render_clip(src, clip, words, out_dir, index, layout, captions=True, voice=None, plan=None):
    """Cut one 1080x1920 H.264 clip with burned-in captions; returns its file name.
    plan (from plan_framing) keeps the person or subject in frame; clip["intro"]
    adds the hook (see render_edit)."""
    if clip.get("file"):
        out_name = os.path.splitext(clip["file"])[0]
    else:
        clean = safe_filename(clip.get("title") or "") or f"clip_{index:02d}"
        candidate = clean
        dup = 2
        while os.path.exists(os.path.join(out_dir, candidate + ".mp4")):
            candidate = f"{clean} ({dup})"
            dup += 1
        out_name = candidate
    return render_edit(src, clip_edl(clip, layout, captions), words, out_dir, f"clip_{index:02d}",
                       clip.get("intro"), voice, plan, out_name=out_name)
