"""Checks for the pure logic in clipper.py. Run: python test_clipper.py (or pytest)."""
import clipper


def words_at(*spans):
    return [{"w": f"w{i}", "s": s, "e": e} for i, (s, e) in enumerate(spans)]


def test_parse_transcript_shifts_by_chunk_offset():
    body = {"words": [{"word": " hi", "start": 1.0, "end": 1.2}, {"word": " ", "start": 2, "end": 2}],
            "segments": [{"start": 1.0, "end": 1.2, "text": " hi "}]}
    words, segments = clipper.parse_transcript(body, 1200)
    assert words == [{"w": "hi", "s": 1201.0, "e": 1201.2}]
    assert segments == [{"s": 1201.0, "e": 1201.2, "text": "hi"}]


def test_build_sentences_splits_on_punctuation_with_exact_word_times():
    words = [{"w": w, "s": i * 0.5, "e": i * 0.5 + 0.4} for i, w in
             enumerate("so I quit my job. Then I sold everything I had and moved".split())]
    segments = [{"s": 0, "e": 4.4, "text": "So I quit my job. Then I sold everything"},
                {"s": 4.5, "e": 6.4, "text": "I had and moved"}]  # text/word count agree per segment
    sents = clipper.build_sentences(words, segments)
    assert [x["text"] for x in sents] == ["So I quit my job.", "Then I sold everything", "I had and moved"]
    assert (sents[0]["s"], sents[0]["e"]) == (0.0, 2.4) and sents[1]["s"] == 2.5
    # Word count doesn't match the text: the segment stays one sentence instead of guessing.
    odd = clipper.build_sentences(words[:3], [{"s": 0, "e": 2, "text": "So, one thousand dollars. Ok"}])
    assert len(odd) == 1 and (odd[0]["s"], odd[0]["e"]) == (0.0, 1.4)
    # No segments at all: split where the speaker pauses.
    paused = clipper.build_sentences([{"w": "a", "s": 0, "e": 0.3}, {"w": "b", "s": 2, "e": 2.3}], [])
    assert len(paused) == 2


def sentences_every(seconds, count):
    return [{"s": i * seconds, "e": i * seconds + seconds - 0.5, "text": f"s{i}."} for i in range(count)]


def test_clean_clips_cut_only_between_whole_sentences():
    sents = sentences_every(5, 40)  # sentence i runs from 5i to 5i+4.5, then 0.5 s of silence
    raw = [
        {"start_id": 2, "end_id": 7, "title": "best", "score": 90, "teaser_id": 5},
        {"start_id": 6, "end_id": 9, "title": "overlaps best", "score": 80},
        {"start_id": 20, "end_id": 20, "title": "too short", "score": 99},
        {"start_id": 10, "end_id": 39, "score": "n/a"},        # 150 s: trimmed to whole sentences under 90 s
        {"start_id": 5, "end_id": 99}, {"start_id": "x"}, "junk",
    ]
    clips = clipper.clean_clips(raw, sents, 200, 5)
    assert [c["title"] for c in clips] == ["best", "Untitled clip"]
    best, long_one = clips
    assert (best["start"], best["end"]) == (9.75, 39.75)    # 0.25 s into each silence, never into a word
    assert best["intro"] == {"start": 24.9, "end": 29.7}    # whole sentence 5
    assert long_one["end"] - long_one["start"] <= clipper.KEEP_MAX
    assert long_one["end"] == round(sents[27]["e"] + 0.25, 2)  # ends exactly after a sentence
    assert clipper.clean_clips(None, sents, 200, 5) == []


def test_clips_can_join_parts_from_anywhere_in_the_video():
    sents = sentences_every(5, 60)  # sentence i: 5i .. 5i+4.5, then 0.5 s of silence
    raw = [
        {"message": "m", "parts": [[40, 42], [3, 4], [10, 11]], "score": 90, "teaser_id": 10},  # later opener first
        {"parts": [[5, 6], [6, 8], [7, 9]], "score": 80},       # runs that continue each other: one part
        {"parts": [[20, 22], [41, 41]], "score": 70},           # reuses sentence 41 from clip 1: rejected
        {"parts": [[30, 31], [30, 31]], "score": 60},           # the same run twice: kept once
    ]
    first, second, third = clipper.clean_clips(raw, sents, 300, 5)
    assert first["ids"] == [[40, 42], [3, 4], [10, 11]] and first["message"] == "m"
    # parts keep the AI's order; a join gets a short breath (0.25 s here), the clip's end the full pause
    assert first["parts"] == [{"s": 199.75, "e": 214.75}, {"s": 14.75, "e": 24.75}, {"s": 49.75, "e": 59.75}]
    assert (first["start"], first["end"]) == (14.75, 214.75)
    assert first["intro"] == {"start": 49.9, "end": 54.7}      # teaser from a later part
    assert second["ids"] == [[5, 9]] and len(second["parts"]) == 1
    assert third["ids"] == [[30, 31]]
    assert clipper.clean_teaser({"teaser_id": 40}, sents, [(40, 42), (3, 4)]) is None  # the clip's opening line


def test_streamed_gemini_answer_is_merged_into_one_response():
    import httpx
    import json
    chunks = [{"candidates": [{"content": {"parts": [{"text": "think", "thought": True}]}}]},
              {"candidates": [{"content": {"parts": [{"text": '{"a": '}]}}]},
              {"candidates": [{"content": {"parts": [{"text": "1}"}]}, "finishReason": "STOP"}],
               "usageMetadata": {"totalTokenCount": 9}}]
    body = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
    merged = clipper.merge_stream(httpx.Response(200, text=body, request=httpx.Request("POST", "https://x"))).json()
    parts = merged["candidates"][0]["content"]["parts"]
    assert "".join(p["text"] for p in parts if not p.get("thought")) == '{"a": 1}'
    assert merged["candidates"][0]["finishReason"] == "STOP" and merged["usageMetadata"]["totalTokenCount"] == 9


def test_finished_clips_are_at_least_min_clip_long():
    sents = sentences_every(5, 40)
    raw = [{"parts": [[0, 3]], "score": 90}, {"parts": [[10, 19]], "score": 80}]  # ~20 s and ~50 s
    assert [c["ids"] for c in clipper.clean_clips(raw, sents, 200, 5, floor=clipper.MIN_CLIP)] == [[[10, 19]]]


def test_caption_groups_break_on_size_pause_and_punctuation():
    words = [{"w": "one", "s": 0, "e": 0.2}, {"w": "two,", "s": 0.3, "e": 0.5},
             {"w": "three", "s": 0.6, "e": 0.8}, {"w": "four", "s": 2.0, "e": 2.2},
             {"w": "five", "s": 2.3, "e": 2.5}, {"w": "six", "s": 2.6, "e": 2.8}, {"w": "seven", "s": 2.9, "e": 3.0}]
    assert [[w["w"] for w in g] for g in clipper.caption_groups(words)] == \
        [["one", "two,"], ["three"], ["four", "five", "six"], ["seven"]]


def test_ass_times_and_highlighting():
    assert clipper.ass_time(3661.256) == "1:01:01.26"
    assert clipper.ass_time(59.999) == "0:01:00.00"
    ass = clipper.build_ass([{"w": "a{b}", "s": 10.0, "e": 10.4}, {"w": "c", "s": 10.5, "e": 10.9}], 10.0, 20.0)
    events = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert len(events) == 2
    assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.50,")
    assert clipper.HIGHLIGHT + "a(b)" in events[0] and "{" not in events[0].split(clipper.NORMAL)[-1]


def test_teaser_is_a_whole_sentence_inside_the_clip_but_not_its_first():
    sents = sentences_every(5, 20)
    sents[4] = {"s": 20, "e": 29, "text": "a very long sentence."}
    assert clipper.clean_teaser({"teaser_id": 3}, sents, [(2, 8)]) == {"start": 14.9, "end": 19.7}
    assert clipper.clean_teaser({"teaser_id": 2}, sents, [(2, 8)]) is None   # the clip's opening line
    assert clipper.clean_teaser({"teaser_id": 9}, sents, [(2, 8)]) is None   # outside the clip
    assert clipper.clean_teaser({"teaser_id": 4}, sents, [(2, 8)])["end"] == 25.2  # capped at 5 s
    assert clipper.clean_teaser({}, sents, [(2, 8)]) is None
    hook = clipper.build_hook_ass("Watch {this}", 3.2)
    assert "Style: Hook," in hook and hook.rstrip().endswith(r"0:00:03.20,Hook,,0,0,0,,{\fad(200,200)}Watch (this)")


def test_main_face_prefers_big_central_and_sticks_with_the_same_person():
    faces = [[500, 50, 120, 120], [900, 50, 110, 110], [600, 50, 10, 10]]  # x, y, w, h (last: background)
    assert round(clipper.main_face(faces, 1280, None), 3) == round((500 + 60) / 1280, 3)  # bigger, central
    assert round(clipper.main_face(faces, 1280, 0.75), 3) == round((900 + 55) / 1280, 3)  # stays on the same person
    assert clipper.main_face([], 1280, None) is None


def test_steady_camera_ignores_wobble_and_eases_into_moves():
    keys = clipper.steady_camera([(0, 0.5), (1, 0.52), (2, 0.8), (3, 0.8)])
    assert keys[1][1] == 0.5                       # 2% wobble: camera holds
    assert 0.5 < keys[2][1] < 0.8 and keys[3][1] > keys[2][1]  # real move: eases toward it


def test_crop_follows_the_plan_and_stays_inside_the_frame():
    plan = {"scaled_w": 3414, "shots": [
        {"s": 0, "e": 10, "mode": "face", "keys": [(0, 0.0), (10, 1.0)]},
        {"s": 10, "e": 20, "mode": "subject", "x": 0.5},
        {"s": 20, "e": 30, "mode": "fit"}]}
    assert clipper.crop_x(plan, 0) == 0                       # clamped at the left edge
    assert clipper.crop_x(plan, 5) == round(0.5 * 3414 - 540)  # interpolated
    assert clipper.crop_x(plan, 9.99) == 3414 - 1080          # clamped at the right edge
    assert clipper.crop_x(plan, 15) == round(0.5 * 3414 - 540)
    graph = clipper.framing("[0:v]", "auto", plan, 18, 6, ".", "m", "test_m.cmd")
    assert "sendcmd=f=test_m.cmd" in graph and "enable='between(t,2.000,6.000)'" in graph
    assert "overlay" not in clipper.framing("[0:v]", "fill", plan, 18, 6, ".", "m", "test_m.cmd")
    import os
    os.remove("test_m.cmd")


def test_caption_fixes_keep_word_timing():
    words = [{"w": "helo", "s": 1.0, "e": 1.4}, {"w": "wrld", "s": 1.5, "e": 2.0}, {"w": "bye", "s": 3.0, "e": 3.3}]
    fixed = clipper.replace_words(words, 0.9, 2.1, "hello world")
    assert [(w["w"], w["s"], w["e"]) for w in fixed[:2]] == [("hello", 1.0, 1.4), ("world", 1.5, 2.0)]
    longer = clipper.replace_words(words, 0.9, 2.1, "hello big world")
    assert [w["w"] for w in longer] == ["hello", "big", "world", "bye"]
    assert longer[0]["s"] == 1.0 and longer[2]["e"] <= 2.0 and longer[3] == words[2]
    assert clipper.replace_words(words, 0.9, 2.1, "") == [words[2]]  # captions removed there
    assert clipper.replace_words(words, 5, 6, "x") is words          # nothing spoken there


def test_ai_answers_with_or_without_the_wrapper_object():
    clips = [{"start_id": 0, "end_id": 1}]
    assert clipper.answer_list({"clips": clips}, "clips") == clips
    assert clipper.answer_list(clips, "clips") == clips          # bare list, as Gemini sometimes sends
    assert clipper.answer_list({"other": 1}, "clips") == []
    assert clipper.answer_list(None, "clips") == [] and clipper.answer_list("text", "clips") == []


def test_spoken_prefix_is_found_before_the_hook_line():
    heard = [("say", 0.0, 0.2), ("the", 0.2, 0.3), ("following", 0.3, 0.7), ("text", 0.7, 0.9), ("aloud", 0.9, 1.3),
             ("watch", 1.6, 1.8), ("how", 1.8, 2.0), ("this", 2.0, 2.2)]
    assert clipper.spoken_start(heard, "Watch how this man...") == 1.52
    assert clipper.spoken_start(heard[5:], "Watch how this man...") is None       # clean voice: untouched
    assert clipper.spoken_start(heard, "Wach hwo this") == 1.52                   # misheard line: cut after "aloud"
    assert clipper.spoken_start([("hello", 0, 0.3)], "Watch how") is None


def test_google_words_get_their_punctuation_and_sentences():
    words = [{"w": w, "s": i, "e": i + 0.5} for i, w in enumerate(["but", "that", "was", "2018", "these", "days", "15", "%"])]
    fixed = clipper.punctuate(words, "But, that was 2018. These days, 15 %")
    assert [w["w"] for w in fixed] == ["But,", "that", "was", "2018.", "These", "days,", "15", "%"]
    assert fixed[3]["s"] == 3 and fixed[3]["e"] == 3.5                    # timing untouched
    assert [s["text"] for s in clipper.sentence_segments(fixed)] == ["But, that was 2018.", "These days, 15 %"]
    assert [w["w"] for w in clipper.punctuate(words[:2], "Totally different text")] == ["but", "that"]  # no match: left as is


def test_chirp_words_are_shifted_to_their_piece_and_punctuated():
    class R:
        status_code = 200
        def json(self):
            return {"results": [{"alternatives": [{"transcript": "But, that was 2018.",
                     "words": [{"word": "but", "startOffset": "0.2s", "endOffset": "0.5s"},
                               {"word": "that", "startOffset": "0.6s", "endOffset": "0.8s"},
                               {"word": "was", "endOffset": "1s"},
                               {"word": "2018", "startOffset": "1.1s", "endOffset": "1.9s"}]}]}]}
    sent = {}
    real = clipper.post
    clipper.post = lambda url, **kw: sent.update(url=url, auth=kw["headers"]["Authorization"]) or R()
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "p.ogg")
    open(path, "wb").write(b"OggS")
    try:
        words, segments = clipper.chirp_piece(path, 100.0, ("tok", "my-project"))
    finally:
        clipper.post = real
    assert [w["w"] for w in words] == ["But,", "that", "was", "2018."]
    assert (words[0]["s"], words[-1]["e"]) == (100.2, 101.9) and words[2]["s"] == 100.0  # missing start = piece start
    assert segments == [{"s": 100.2, "e": 101.9, "text": "But, that was 2018."}]
    assert "/projects/my-project/locations/" in sent["url"] and sent["auth"] == "Bearer tok"


def test_retry_wait():
    assert clipper.retry_wait("42", 0) == 42
    assert clipper.retry_wait(None, 2) == 40
    assert clipper.retry_wait("99999", 0) == 600


def test_safe_filename():
    assert clipper.safe_filename("I Tried EVERY Way To Build Muscle: What Worked?") == "I Tried EVERY Way To Build Muscle - What Worked"
    assert clipper.safe_filename("Test / Slash \\ Back | Pipe ? Question * Star \" Quote < Less > More") == "Test - Slash - Back - Pipe Question Star Quote Less More"
    assert clipper.safe_filename("AUX") == "_AUX"
    assert clipper.safe_filename("   ...clean name...   ") == "clean name"
    assert len(clipper.safe_filename("A" * 150)) <= 100


def test_video_encoder_args():
    enc = clipper.video_encoder_args()
    assert "-c:v" in enc
    assert "-pix_fmt" in enc and "yuv420p" in enc
    soft = clipper.video_encoder_args(force_software=True)
    assert soft == ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]


def test_downscaled_fit_chain():
    chain = clipper.fit_chain("tag")
    assert "scale=270:480" in chain
    assert "boxblur=8:2" in chain
    assert "scale=1080:1920:flags=bilinear" in chain


def test_kinetic_word_pop_in_ass():
    assert r"\fscx112\fscy112" in clipper.HIGHLIGHT
    assert r"\fscx100\fscy100" in clipper.NORMAL
    ass = clipper.build_ass([{"w": "Pop", "s": 0.0, "e": 0.5}], 0.0, 1.0)
    assert r"\fscx112\fscy112" in ass and r"\fscx100\fscy100" in ass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

