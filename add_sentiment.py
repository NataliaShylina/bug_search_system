import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

INPUT_CSV = "GFG_powerbi.csv"
OUTPUT_CSV = "bugs_with_sentiment.csv"
TEXT_COLUMNS = ["Summary", "Description"]
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

df = pd.read_csv(INPUT_CSV)

for col in TEXT_COLUMNS:
    if col not in df.columns:
        raise ValueError(f"Brak kolumny: {col}")

df["text_for_sentiment"] = (
    df["Summary"].fillna("") + ". " + df["Description"].fillna("")
)

#  LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

#  SENTIMENT FUNCTION
LABEL_MAP = {
    0: -1,  # negative
    1: 0,   # neutral
    2: 1    # positive
}

def compute_sentiment(text):
    if not text.strip():
        return 0.0

    inputs = tokenizer(
        text[:512],  # limit tokenów
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    score = 0.0
    for i, p in enumerate(probs):
        score += LABEL_MAP[i] * p.item()

    return round(score, 4)

# APPLY sentiment_score
tqdm.pandas()
df["sentiment_score"] = df["text_for_sentiment"].progress_apply(compute_sentiment)

df.drop(columns=["text_for_sentiment"], inplace=True)
df.to_csv(OUTPUT_CSV, index=False)

print("sentiment_score dodany")
print(df["sentiment_score"].describe())
