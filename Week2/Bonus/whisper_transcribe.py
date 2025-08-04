import os
import json
import whisper
from yt_dlp import YoutubeDL

# Paths
ffmpeg_path = r"asd/ffmpeg/bin"
audio_dir = "talks_audio"
transcript_path = os.path.join("talks_transcripts", "talks_transcripts.jsonl")

# YouTube URLs
video_urls = [
    "https://www.youtube.com/watch?v=Xe3H2R_2Ta4",
    "https://www.youtube.com/watch?v=4f2073Kx4KA",
    "https://www.youtube.com/watch?v=P3icxlpgPhE"
]

# GPU-aware Whisper model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

# Make folders
os.makedirs(audio_dir, exist_ok=True)
os.makedirs("talks_transcripts", exist_ok=True)

def get_audio_filename(url):
    vid_id = url.split("v=")[-1]
    return os.path.join(audio_dir, f"{vid_id}.mp3"), vid_id

def download_audio(url):
    filepath, vid_id = get_audio_filename(url)
    if os.path.exists(filepath):
        print(f"[~] Skipping download, found: {filepath}")
        return filepath, vid_id
    ydl_opts = {
        'ffmpeg_location': ffmpeg_path,
        'format': 'bestaudio/best',
        'outtmpl': filepath.replace(".mp3", "") + ".%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return filepath, vid_id

with open(transcript_path, "w", encoding="utf-8") as outfile:
    for url in video_urls:
        try:
            audio_path, vid_id = download_audio(url)
            print(f"[🎙️] Transcribing {vid_id} ...")
            result = model.transcribe(audio_path)
            if "segments" in result:
                data = {
                    "video_id": vid_id,
                    "url": url,
                    "segments": result["segments"],
                    "text": result["text"]
                }
                json.dump(data, outfile)
                outfile.write("\n")
                print(f"[✓] Done: {vid_id}")
            else:
                print(f"[!] No segments for {vid_id}")
        except Exception as e:
            print(f"[!] Error with {url}: {e}")
