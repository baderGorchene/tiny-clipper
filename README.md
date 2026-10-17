# Tiny Clipper

Turns a long video into short vertical clips with captions, ready for TikTok, Reels and Shorts.
It uses free online AI (Groq and Google Gemini), so it runs on any computer: no graphics card needed.

## Start

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/) (tick "Add to PATH").
2. Double-click **`start.bat`**. The first start takes a few minutes; after that it opens in seconds.
3. Your browser opens Tiny Clipper. Add your free Gemini key (the app shows you how), then drop in a video.

Keep the black window open while you use it. Closing it stops Tiny Clipper.

## Where things are

- **Your clips:** the `library` folder. They stay until you delete them in the app.
- **Your keys:** `settings.json`, on this computer only.
- **Originals:** each finished video's original is kept in its `library` folder so you can **Edit** its clips. The editor has a live preview and a timeline: split and trim parts (even past the original cut), reframe by dragging the picture, add text and sound effects or your own music, move the captions. Deleting a video in the app removes its original too.

## How it works

1. **Speech to text:** Google Cloud Speech-to-Text (**Chirp 3**) when this computer has the Google sign-in (below), else Google `gemini-3.5-transcribe` with a free Gemini key, else Groq `whisper-large-v3-turbo` (free). All give a timestamp for every word.
2. **Best moments:** Gemini (`gemini-3.1-flash-lite`, then `gemini-3.5-flash-lite`, `gemini-flash-lite-latest`, `gemini-3.5-flash` or `gemini-3.8-flash` when one is overloaded or out of quota). If Gemini isn't set up or every model fails, Groq takes over (`openai/gpt-oss-120b`, then `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`). A second pass checks that every clip tells a complete story.
   - **Your review** (on by default, "Let me review the picks first"): the job stops here and nothing is voiced or cut until you click **Review picks**. Fix titles and hook lines, hear the voice, add or remove sentences, drop clips. Left alone for 30 minutes it goes ahead with the clips scoring 85+ (`"hitl_auto_approve_minutes"` in `settings.json`; 0 waits for you). Other videos keep processing meanwhile.
3. **Framing (Auto):** for each shot, a small face detector on your computer keeps the main person in frame. If there's no face, Gemini points out the subject. Screens and wide shots show the whole frame.
4. **Hook:** each clip can open on its best moment, muted, with an AI narrator ("Watch how this man…") and a whoosh into the clip. The voice is made by Tiny Clipper itself, no browser needed: Gemini 3.8 Flash TTS with your Gemini key (Google Cloud credit first), then Google Cloud Text-to-Speech if this computer has the Google sign-in, then the free [edge-tts](https://github.com/rany2/edge-tts) voice (no key).
5. **Cutting:** ffmpeg makes a 1080×1920 clip with captions where each spoken word lights up.

## Posting to Instagram, YouTube and TikTok

Optional, in **Keys → Post directly**. Each platform needs a free developer app of your own; the app shows the steps. Then every clip gets a **Post** button.

- **Instagram Reels:** posts publicly to your Business/Creator account. Instagram fetches videos from a link, so the clip is put on a temporary file host ([Litterbox](https://litterbox.catbox.moe), deleted after 1 hour) just long enough for that.
- **YouTube Shorts:** uploads stay private until Google audits your project ([free audit form](https://support.google.com/youtube/contact/yt_api_form)).
- **TikTok:** clips go to your TikTok inbox as drafts; you finish posting in the TikTok app. Until TikTok reviews your app, it works for the accounts you add as testers.

## Using Google Cloud credit

If your Gemini key's Google Cloud project has the **Agent Platform API** (Vertex AI) turned on and a billing account with credit, Tiny Clipper uses it automatically (Keys shows "Google Cloud credit: on"):

- **Gemini 3.1 Pro** picks the moments and checks each clip's story (Gemini 3.8 Flash if Pro is busy).
- **Gemini 3.8 Flash** spots the subject for framing.
- **Gemini 3.8 Flash TTS** makes the hook voices on the server, so the page doesn't need to stay open.
- **Nano Banana Pro** designs a 9:16 thumbnail for every clip from its real frames (Nano Banana 2.1 as backup). It's also used as the Instagram cover. Without the credit (Nano Banana has no free tier), Tiny Clipper makes a free cover itself: the sharpest frame, cropped to the face, with the title on it.

The free Gemini models and Groq remain as fallbacks.

**Speech-to-text on Google Cloud** (Chirp 3) needs a Google sign-in on this computer, because Speech-to-Text doesn't accept API keys. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then in a new terminal run `gcloud auth application-default login` and `gcloud auth application-default set-quota-project YOUR_PROJECT_ID`, and restart Tiny Clipper. Keys shows "signed in" when it's found.

The switch at the top of the page picks which services run:

- **Auto: all keys** (default): Google Cloud credit first, the free services as fallbacks.
- **100% free**: only free tiers (free Gemini key, Groq, edge-tts voice, free covers); the credit and the Google sign-in aren't used.
- **100% Google Cloud**: the credit alone, with no fallbacks; your other keys stay saved but unused. Switch back when the credit ends.

To use different models, set the environment variables `GEMINI_MODEL`, `GROQ_LLM_MODEL` or `GROQ_WHISPER_MODEL`.

Run the checks with `.venv\Scripts\python test_clipper.py`.
