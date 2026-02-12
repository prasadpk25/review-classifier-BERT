import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class SentimentModel:
    def __init__(self):
        # Force CPU usage to prevent CUDA errors on free tier
        self.device = torch.device("cpu")
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        print(f"Loading model: {self.model_name}...") # Debug log
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
            raise e

    def predict(self, text: str):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device) # Move inputs to CPU
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        confidence, predicted_class = torch.max(probs, dim=-1)
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        
        return {
            "label": label_map[predicted_class.item()],
            "confidence": round(float(confidence.item()), 4)
        }

classifier = SentimentModel()