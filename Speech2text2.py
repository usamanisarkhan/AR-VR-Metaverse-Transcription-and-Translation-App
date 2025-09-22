import os
import cv2
import ffmpeg
import whisper
import numpy as np
from deep_translator import GoogleTranslator
from PIL import ImageFont, ImageDraw, Image

# -------- CONFIG --------
VIDEO_PATH = "input_video.mp4"      # your video
AUDIO_PATH = "temp_audio.wav"
TEMP_VIDEO = "temp_subs.mp4"
OUTPUT_PATH = "output_with_subs.mp4"

FONT_PATH = "arial.ttf"             # font supporting your language
FONT_SIZE = 32
TEXT_POSITION = (50, 400)           # x,y position of subtitles
MAX_WIDTH_RATIO = 0.8               # wrap if text too wide
# ------------------------

def extract_audio(video_path, audio_path):
    """Extract mono 16kHz audio from video."""
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    print("✅ Audio extracted:", audio_path)

def transcribe_and_translate(audio_path, lang="en"):
    """Transcribe audio with Whisper and translate to English."""
    model = whisper.load_model("small")
    print(f"⏳ Transcribing {lang}...")
    result = model.transcribe(audio_path, language=lang)

    subtitles = []
    for seg in result["segments"]:
        start, end = seg["start"], seg["end"]
        src_text = seg["text"].strip()
        en_text = GoogleTranslator(source=lang, target="fi").translate(src_text)
        print(f"[{start:.2f}-{end:.2f}] {src_text} → {en_text}")
        subtitles.append((start, end, en_text))
    return subtitles

def draw_subtitle(frame, text, position=(50, 400), max_width_ratio=0.8):
    """Draw wrapped subtitle on a frame, red text with black outline."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()

    x, y = position
    max_width = int(frame.shape[1] * max_width_ratio)

    words = text.split()
    lines, current_line = [], ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0,0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for i, line in enumerate(lines):
        line_y = y + i * (FONT_SIZE + 10)
        outline_range = 2
        for dx in range(-outline_range, outline_range+1):
            for dy in range(-outline_range, outline_range+1):
                draw.text((x+dx, line_y+dy), line, font=font, fill=(0,0,0))
        draw.text((x, line_y), line, font=font, fill=(255,0,0))  # red text

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def add_subtitles(video_path, output_path, subtitles):
    """Add progressive word-by-word subtitles to video."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps

        current_text = ""
        for (start, end, text) in subtitles:
            if start <= time_sec <= end:
                words = text.split()
                duration = end - start
                if duration <= 0:
                    duration = 0.1
                words_per_second = len(words) / duration
                elapsed = time_sec - start
                words_to_show = int(elapsed * words_per_second)
                words_to_show = min(words_to_show, len(words))
                current_text = " ".join(words[:words_to_show])
                break

        if current_text:
            frame = draw_subtitle(frame, current_text, TEXT_POSITION, MAX_WIDTH_RATIO)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("🎬 Subtitled silent video saved:", output_path)

def merge_audio(video_path, original_video, output_path):
    """Merge silent subtitled video with original audio."""
    input1 = ffmpeg.input(video_path)
    input2 = ffmpeg.input(original_video)
    (
        ffmpeg
        .output(input1.video, input2.audio, output_path, vcodec="copy", acodec="aac")
        .overwrite_output()
        .run(quiet=True)
    )
    print("🔊 Audio merged back:", output_path)

def main():
    extract_audio(VIDEO_PATH, AUDIO_PATH)
    subtitles = transcribe_and_translate(AUDIO_PATH, lang="en")
    add_subtitles(VIDEO_PATH, TEMP_VIDEO, subtitles)
    merge_audio(TEMP_VIDEO, VIDEO_PATH, OUTPUT_PATH)

    if os.path.exists(TEMP_VIDEO):
        os.remove(TEMP_VIDEO)

if __name__ == "__main__":
    main()
