from fastapi import FastAPI, UploadFile, File
from app.speech_to_text import transcribe_and_diarize
from app.nlp_processing import process_text, summarize_conversation, extract_problem_and_resolution
from app.prediction import predict_satisfaction, predict_behavior
from app.utils import extract_features

app = FastAPI()

@app.post("/analyze/")
async def analyze_conversation(file: UploadFile = File(...)):
    audio_file = file.file
    transcript = transcribe_and_diarize(audio_file)

    # Join transcript into a single text conversation
    full_conversation = ' '.join(transcript)
    
    # Get entities and sentiment from NLP
    entities, sentiment = process_text(full_conversation)

    # Predict satisfaction and behavior
    features = extract_features(transcript)
    satisfaction = predict_satisfaction(features)
    behavior = predict_behavior(full_conversation)

    # Summarize the conversation
    conversation_summary = summarize_conversation(full_conversation)

    # Extract problem and resolution from the conversation
    problem, resolution = extract_problem_and_resolution(full_conversation)

    return {
        "transcript": transcript,
        "entities": entities,
        "sentiment": sentiment,
        "satisfaction": satisfaction,
        "behavior": behavior,
        "conversation_summary": conversation_summary,
        "problem": problem,
        "resolution": resolution
    }
