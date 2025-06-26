from pydantic import BaseModel

# --- Models and endpoint for speaker recognition ---
class SpeakerResult(BaseModel):
    speaker_id: str
    score: float

# --- Models and endpoint for NL to Cypher conversion ---
class NLQuery(BaseModel):
    nl_query: str