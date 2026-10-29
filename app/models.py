from pydantic import BaseModel


class ChatTurn(BaseModel):
    role: str
    content: str


class Source(BaseModel):
    id: str
    doc_id: str
    title: str
    doc_type: str
    date: str
    section: str
    score: float
    text: str
