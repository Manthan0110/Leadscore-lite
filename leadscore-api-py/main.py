# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os, json, logging, re
from google import genai   # google-genai SDK

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# init client — uses ADC in Cloud Run, or API key if set locally
client = genai.Client()

class Lead(BaseModel):
    name: str | None = None
    email: str | None = None
    company: str | None = None
    pitch: str

@app.post("/score")
async def score(lead: Lead):
    # Make a short, clear prompt that asks for JSON only
    prompt = f"""
You are an assistant that scores sales leads from 0 to 100 (100 best).
Return ONLY a single JSON object and nothing else with:
{{"score": <integer 0-100>, "reasons": [<short strings>]}}
Lead information:
Name: {lead.name}
Email: {lead.email}
Company: {lead.company}
Pitch: {lead.pitch}
"""
    logging.info("Sending prompt to Vertex AI / Gemini")
    resp = client.models.generate_content(
        model="gemini-2.5-flash",    # example model name; you can change later
        contents=prompt
    )

    text = resp.text
    logging.info("Raw model output: %s", text[:400])

    # Try to parse JSON strictly; if model adds text, find the JSON blob
    try:
        result = json.loads(text)
    except Exception:
        m = re.search(r"\{(.|\n)*\}", text)
        if m:
            result = json.loads(m.group(0))
        else:
            # If parsing fails, return raw text so you can inspect
            return {"score": None, "reasons": [], "raw": text}

    return {"score": result.get("score"), "reasons": result.get("reasons", [])}
