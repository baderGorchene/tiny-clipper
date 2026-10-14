"""Post finished clips to Instagram, YouTube and TikTok through your own free developer apps.

What each platform allows until it reviews your app:
  Instagram  posts publicly to your own Business/Creator account (no review needed)
  YouTube    uploads stay private until Google audits your project
  TikTok     clips land in your TikTok inbox as drafts; you finish posting in the app
"""
import os
import time
import urllib.parse

import httpx


class SocialError(Exception):
    """A posting failure whose message is written for the user."""


def _error(name, resp):
    try:
        body = resp.json()
        err = body.get("error") if isinstance(body, dict) else None
        msg = (err.get("message") or err.get("description") or err.get("code")) if isinstance(err, dict) else err
        msg = msg or body.get("error_description") or body.get("message")
    except ValueError:
        msg = resp.text[:200]
    return SocialError(f"{name}: {msg or f'error {resp.status_code}'}")


# --- Instagram (Instagram API with Instagram Login) --------------------------

IG = "https://graph.instagram.com"
TEMP_HOST = "https://litterbox.catbox.moe/resources/internals/api.php"


def ig_check(token):
    """{"user_id", "username"} for a token, or SocialError."""
    resp = httpx.get(f"{IG}/me", params={"fields": "user_id,username", "access_token": token}, timeout=20)
    if resp.status_code != 200:
        raise _error("Instagram", resp)
    me = resp.json()
    return {"user_id": str(me["user_id"]), "username": me.get("username", "")}


def ig_refresh(cfg):
    """Long-lived tokens last 60 days; renewing weekly keeps them alive forever."""
    if time.time() - cfg.get("refreshed", 0) < 7 * 86400:
        return cfg
    resp = httpx.get(f"{IG}/refresh_access_token",
                     params={"grant_type": "ig_refresh_token", "access_token": cfg["token"]}, timeout=20)
    if resp.status_code == 200:
        cfg = {**cfg, "token": resp.json()["access_token"], "refreshed": time.time()}
    return cfg


def temp_link(path, mime="video/mp4"):
    """Public link to a file for 1 hour, because Instagram fetches media by URL."""
    with open(path, "rb") as f:
        resp = httpx.post(TEMP_HOST, data={"reqtype": "fileupload", "time": "1h"},
                          files={"fileToUpload": (os.path.basename(path), f, mime)}, timeout=600)
    link = resp.text.strip()
    if resp.status_code != 200 or not link.startswith("https://"):
        raise SocialError(f"Instagram: couldn't put the clip online for Instagram to fetch ({link[:120]})")
    return link


def ig_post(cfg, path, caption, cover=None):
    """Publish a Reel (cover: an optional thumbnail image); returns its link."""
    token, uid = cfg["token"], cfg["user_id"]
    link = temp_link(path)
    extra = {"cover_url": temp_link(cover, "image/jpeg")} if cover else {}
    resp = httpx.post(f"{IG}/{uid}/media", timeout=60, data={
        "media_type": "REELS", "video_url": link, "caption": caption, "share_to_feed": "true",
        "access_token": token, **extra})
    if resp.status_code != 200:
        raise _error("Instagram", resp)
    container = resp.json()["id"]
    for _ in range(60):  # Instagram processes the video: up to ~5 minutes
        time.sleep(5)
        status = httpx.get(f"{IG}/{container}", timeout=20,
                           params={"fields": "status_code,status", "access_token": token}).json()
        if status.get("status_code") == "FINISHED":
            break
        if status.get("status_code") in ("ERROR", "EXPIRED"):
            raise SocialError(f"Instagram couldn't process the video: {status.get('status', 'unknown reason')}")
    else:
        raise SocialError("Instagram took too long to process the video. Try again later.")
    resp = httpx.post(f"{IG}/{uid}/media_publish", data={"creation_id": container, "access_token": token}, timeout=60)
    if resp.status_code != 200:
        raise _error("Instagram", resp)
    media = httpx.get(f"{IG}/{resp.json()['id']}", params={"fields": "permalink", "access_token": token},
                      timeout=20).json()
    return media.get("permalink") or f"https://www.instagram.com/{cfg.get('username', '')}"


# --- YouTube (Data API v3, installed-app OAuth on the loopback address) -------

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
YT_SCOPE = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"


def yt_auth_url(cfg, redirect_uri, state):
    return GOOGLE_AUTH + "?" + urllib.parse.urlencode({
        "client_id": cfg["client_id"], "redirect_uri": redirect_uri, "response_type": "code",
        "scope": YT_SCOPE, "access_type": "offline", "prompt": "consent", "state": state})


def yt_connect(cfg, code, redirect_uri):
    """Trade the sign-in code for a lasting refresh token; returns the updated config."""
    resp = httpx.post(GOOGLE_TOKEN, timeout=20, data={
        "client_id": cfg["client_id"], "client_secret": cfg["client_secret"], "code": code,
        "grant_type": "authorization_code", "redirect_uri": redirect_uri})
    if resp.status_code != 200:
        raise _error("YouTube", resp)
    tokens = resp.json()
    channel = httpx.get("https://www.googleapis.com/youtube/v3/channels", timeout=20,
                        params={"part": "snippet", "mine": "true"},
                        headers={"Authorization": f"Bearer {tokens['access_token']}"}).json()
    name = ((channel.get("items") or [{}])[0].get("snippet") or {}).get("title", "your channel")
    return {**cfg, "refresh_token": tokens["refresh_token"], "channel": name}


def yt_post(cfg, path, title, description):
    """Upload a Short; returns its link."""
    resp = httpx.post(GOOGLE_TOKEN, timeout=20, data={
        "client_id": cfg["client_id"], "client_secret": cfg["client_secret"],
        "refresh_token": cfg["refresh_token"], "grant_type": "refresh_token"})
    if resp.status_code != 200:
        raise SocialError("YouTube: sign-in expired. Connect YouTube again in Keys.")
    auth = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    size = os.path.getsize(path)
    init = httpx.post("https://www.googleapis.com/upload/youtube/v3/videos", timeout=60,
                      params={"uploadType": "resumable", "part": "snippet,status"},
                      headers={**auth, "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                      json={"snippet": {"title": f"{title[:90]} #Shorts", "description": description,
                                        "categoryId": "22"},
                            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False}})
    if init.status_code != 200:
        raise _error("YouTube", init)
    with open(path, "rb") as f:
        resp = httpx.put(init.headers["Location"], content=f.read(), timeout=900,
                         headers={**auth, "Content-Type": "video/mp4"})
    if resp.status_code not in (200, 201):
        raise _error("YouTube", resp)
    return f"https://youtube.com/shorts/{resp.json()['id']}"


# --- TikTok (Content Posting API, upload to inbox) ----------------------------
# TikTok only redirects sign-ins to an https address, never to this computer,
# so the user pastes back the address they land on (it carries the code).

TT_AUTH = "https://www.tiktok.com/v2/auth/authorize/"
TT_TOKEN = "https://open.tiktokapis.com/v2/oauth/token/"
TT_INIT = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"


def tt_auth_url(cfg, state):
    return TT_AUTH + "?" + urllib.parse.urlencode({
        "client_key": cfg["client_key"], "response_type": "code", "scope": "video.upload",
        "redirect_uri": cfg["redirect_uri"], "state": state})


def _tt_tokens(cfg, data):
    resp = httpx.post(TT_TOKEN, timeout=20, data={
        "client_key": cfg["client_key"], "client_secret": cfg["client_secret"], **data})
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code != 200 or "access_token" not in body:
        raise SocialError(f"TikTok: {body.get('error_description') or body.get('error') or resp.text[:150]}")
    return {**cfg, "access_token": body["access_token"], "refresh_token": body["refresh_token"],
            "expires": time.time() + int(body.get("expires_in", 86400)), "open_id": body.get("open_id", "")}


def tt_connect(cfg, pasted, expected_state):
    """Finish sign-in from the pasted address (or bare code)."""
    query = urllib.parse.parse_qs(urllib.parse.urlparse(pasted.strip()).query)
    if query.get("error"):
        raise SocialError(f"TikTok: {query.get('error_description', query['error'])[0]}")
    if "code" in query:
        if query.get("state", [""])[0] != expected_state:
            raise SocialError("TikTok: that address is from an older sign-in. Click Connect TikTok again.")
        code = query["code"][0]
    else:
        code = pasted.strip()
    if not code:
        raise SocialError("TikTok: paste the full address of the page TikTok sent you to.")
    return _tt_tokens(cfg, {"code": code, "grant_type": "authorization_code", "redirect_uri": cfg["redirect_uri"]})


def tt_fresh(cfg):
    """Access tokens last 24 h; the refresh token renews them for a year."""
    if cfg.get("expires", 0) - 120 > time.time():
        return cfg
    return _tt_tokens(cfg, {"grant_type": "refresh_token", "refresh_token": cfg["refresh_token"]})


def tt_post(cfg, path):
    """Send the clip to the TikTok inbox as a draft."""
    size = os.path.getsize(path)
    # One chunk up to 64 MB; above that 10 MB chunks, the last one taking the remainder.
    chunk = size if size <= 64 * 1024 * 1024 else 10 * 1024 * 1024
    count = max(1, size // chunk)
    resp = httpx.post(TT_INIT, timeout=60, headers={"Authorization": f"Bearer {cfg['access_token']}",
                                                    "Content-Type": "application/json; charset=UTF-8"},
                      json={"source_info": {"source": "FILE_UPLOAD", "video_size": size,
                                            "chunk_size": chunk, "total_chunk_count": count}})
    body = resp.json()
    if resp.status_code != 200 or (body.get("error") or {}).get("code") != "ok":
        raise _error("TikTok", resp)
    upload_url = body["data"]["upload_url"]
    with open(path, "rb") as f:
        for i in range(count):
            first = i * chunk
            last = size - 1 if i == count - 1 else first + chunk - 1
            f.seek(first)
            put = httpx.put(upload_url, content=f.read(last - first + 1), timeout=600, headers={
                "Content-Type": "video/mp4", "Content-Range": f"bytes {first}-{last}/{size}"})
            if put.status_code not in (200, 201, 206):
                raise SocialError(f"TikTok upload failed ({put.status_code}). Try again.")
    return "Sent to your TikTok inbox. Open TikTok to finish and post it."
