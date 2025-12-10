from pydantic import BaseModel, Field

class SentimentAnalysisModel(BaseModel):
    headline: str = Field(description="financial news headline")
    sentiment: str = Field(description="sentiment (0=bearish, 1=bullish, 2=neutral)")