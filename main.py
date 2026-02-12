from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse 
from schemas import ReviewRequest, SentimentResponse
from model import classifier

app = FastAPI(title="Sentiment Analysis API", version="1.0")

# 1. CHANGED: The root path now redirects straight to /docs
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: ReviewRequest):
    try:
        result = classifier.predict(request.text)
        return result
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error.")