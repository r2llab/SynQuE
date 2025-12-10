from pydantic import BaseModel, Field

class Text2SQLModel(BaseModel):
    question: str = Field(description="natural language question that asks about a database")
    SQL: str = Field(description="SQLite query that answers the question")