from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The product review text to analyze")
    
    # Example for Swagger UI (Documentation)
    class Config:
        schema_extra = {
            "example": {
                "text": "I bought this yesterday and the battery life is amazing!"
            }
        }

class SentimentResponse(BaseModel):
    label: str
    confidence: float