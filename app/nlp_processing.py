import spacy
from transformers import pipeline

# Load SpaCy for NER
nlp = spacy.load("en_core_web_sm")

# Summarization and Sentiment analysis from transformers pipeline
summarization_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    revision="a4f8f3e"
)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="af0f99b"
)

def process_text(conversation):
    doc = nlp(conversation)

    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Perform sentiment analysis
    sentiment = sentiment_pipeline(conversation)

    return entities, sentiment

def summarize_conversation(conversation):
    # Use a summarization model to summarize the conversation
    summary = summarization_pipeline(conversation, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extract_problem_and_resolution(conversation):
    # Simple heuristic for problem and resolution extraction (you can enhance this with custom models)
    if "problem" in conversation.lower():
        problem_start = conversation.lower().index("problem")
        problem = conversation[problem_start:]
    else:
        problem = "Not explicitly mentioned."

    if "resolution" in conversation.lower():
        resolution_start = conversation.lower().index("resolution")
        resolution = conversation[resolution_start:]
    else:
        resolution = "Not explicitly mentioned."

    return problem, resolution
