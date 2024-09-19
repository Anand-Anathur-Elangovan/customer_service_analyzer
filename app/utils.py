def extract_features(transcript):
    # Extract features from the transcript
    return [len(transcript), transcript.count('!'), transcript.count('?')]
