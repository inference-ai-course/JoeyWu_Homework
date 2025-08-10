from fastapi import FastAPI, UploadFile, File, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from collections import deque, defaultdict
from openai import OpenAI
import whisper, torch, pyttsx3
import tempfile, os, sys, uuid

# =========================
# Config
# =========================
ASR_MODEL = "small"                       # use "tiny" for faster CPU testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_WINDOWS = (os.name == "nt")
HISTORY_MAX_MESSAGES = 10                 # ~5 user/assistant pairs kept in memory

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. "
    "Use prior context from the conversation when relevant."
)

# =========================
# Init
# =========================
app = FastAPI()
asr = whisper.load_model(ASR_MODEL, device=DEVICE)

# In-memory only (resets on restart): conversation_id -> deque of messages
# Each message: {"role": "user"|"assistant"|"system", "content": "..."}
conversations = defaultdict(lambda: deque(maxlen=HISTORY_MAX_MESSAGES))
CURRENT_CONVO_ID: str | None = None  # single default conversation per run (if caller doesn't pass one)

# =========================
# Helpers
# =========================
def transcribe_audio(audio_bytes: bytes, filename_hint: str) -> str:
    """Write bytes to temp file, let Whisper transcribe, cleanup temp."""
    _, ext = os.path.splitext(filename_hint or "")
    suffix = ext if ext else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        out = asr.transcribe(tmp_path, fp16=False)
        return (out.get("text") or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def generate_response_with_history(messages: list[dict]) -> str:
    """Call OpenAI Chat Completions with full message history."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # ensure your account has access to this model
        messages=messages,
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()

def synthesize_speech(text: str) -> str:
    """pyttsx3 → write to temp .wav and return its path."""
    eng = pyttsx3.init()
    eng.setProperty("rate", 165)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    eng.save_to_file(text, path)
    eng.runAndWait()
    return path

def play_local(path: str):
    """Play WAV on local Windows machine (non-blocking)."""
    if IS_WINDOWS:
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"[winsound] playback failed: {e}", file=sys.stderr)

# =========================
# API: /chat
# =========================
@app.post("/chat/")
async def chat_endpoint(
    response: Response,
    file: UploadFile = File(...),
    debug: bool = Query(True, description="Return JSON (no audio file response)"),
    play: bool  = Query(True,  description="Play TTS locally on this machine"),
    speak_when_debug: bool = Query(True, description="Speak locally even when debug=true"),
    conversation_id: str | None = Query(None, description="Sticky ID for multi-turn memory"),
    reset: bool = Query(False, description="Reset this conversation's history"),
):
    global CURRENT_CONVO_ID

    # 0) Use a single auto-generated conversation per run if none provided
    if conversation_id is None:
        if CURRENT_CONVO_ID is None:
            CURRENT_CONVO_ID = str(uuid.uuid4())
        conversation_id = CURRENT_CONVO_ID

    # Expose the ID to clients (useful in Swagger/curl)
    response.headers["X-Conversation-Id"] = conversation_id

    # Optional reset of just this conversation
    if reset:
        conversations[conversation_id].clear()

    # 1) ASR
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes, file.filename)

    # 2) Build messages from prior context + new user message
    history = conversations[conversation_id]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *list(history), {"role": "user", "content": user_text}]

    # 3) LLM
    bot_text = generate_response_with_history(messages)

    # 4) Update in-memory conversation (RAM only; resets on restart)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": bot_text})

    # --- Debug JSON path (Option A): still allow local speech if requested ---
    if debug:
        if speak_when_debug:
            audio_path_dbg = synthesize_speech(bot_text)
            if play:
                play_local(audio_path_dbg)
            # cleanup the temp wav created only for local playback
            try:
                os.remove(audio_path_dbg)
            except OSError:
                pass

        return JSONResponse({
            "conversation_id": conversation_id,
            "history_len": len(history),
            "device": DEVICE,
            "asr_model": ASR_MODEL,
            "text": user_text,
            "reply": bot_text
        })

    # 5) TTS → WAV file (normal non-debug response)
    audio_path = synthesize_speech(bot_text)

    # Optional local playback
    if play:
        play_local(audio_path)

    # 6) Return WAV and clean up after sending
    def cleanup(p: str):
        try:
            os.remove(p)
        except OSError:
            pass

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename="reply.wav",
        background=BackgroundTask(cleanup, audio_path),
    )