from pyannote.audio import Pipeline

def transcribe_and_diarize(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(audio_file)
    
    transcript = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        transcript.append(f"Speaker {speaker}: start={turn.start}, end={turn.end}")
    return transcript
