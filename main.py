from fastapi import FastAPI, UploadFile, File, Form
from models.messages import AIRequest, AIResponse
from config import ConfigLoader
from openai import OpenAI

app = FastAPI(title="Speech Recognition API")
config_loader = ConfigLoader()
client = OpenAI(
    api_key=config_loader.get_str("OPENAI_API_KEY")
)

@app.post("/transcribe", response_model=AIResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form(None)
):
    """
    Endpoint to transcribe audio files.
    The audio file is expected to be sent as multipart/form-data.
    """

    ai_request = AIRequest(language=language)
    audio_file = await file.read()

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language=ai_request.language if ai_request.language else None
    )

    return AIResponse(text=transcription.text)