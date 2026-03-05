from pydantic import BaseModel

class ChatSchema(BaseModel):
    query: str
    message_history: list