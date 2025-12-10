from pydantic import BaseModel, Field

class ScoreModel(BaseModel):
    judgement: str = Field(description="How likely is the given tweet comes from dataset B according to your judgement. The judgement should be one of 'very unlikely', 'unlikely', 'unsure', 'likely', and 'very likely'.")
