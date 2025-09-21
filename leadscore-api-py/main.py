# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os, json, logging, re
from google import genai
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

logging.basicConfig(level=logging.INFO)
app = FastAPI()
client = genai.Client()

# Read these from env (we'll set them when deploying)
INSTANCE_CONNECTION_NAME = os.environ.get("INSTANCE_CONNECTION_NAME")  # project:region:instance
DB_USER = os.environ.get("DB_USER", "leaduser")
DB_PASS = os.environ.get("DB_PASS", "StrongPassword123!")
DB_NAME = os.environ.get("DB_NAME", "leadscore_db")

# Create a SQLAlchemy engine using the Cloud SQL Python Connector
def get_engine():
    connector = Connector()  # creates connector instance

    def getconn():
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
            ip_type=IPTypes.PUBLIC  # connector will use secure connection; PUBLIC is fine because IAM + connector secures it
        )
        return conn

    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_size=5,
        max_overflow=2,
    )
    return pool

engine = get_engine()
metadata = sqlalchemy.MetaData()
leads_table = sqlalchemy.Table(
    "leads",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("payload", sqlalchemy.JSON),
    sqlalchemy.Column("score", sqlalchemy.Integer),
    sqlalchemy.Column("reasons", sqlalchemy.ARRAY(sqlalchemy.String)),
    sqlalchemy.Column("created_at", sqlalchemy.TIMESTAMP(timezone=True), server_default=sqlalchemy.func.now()),
)

# (Optional) ensure table exists (safe to call)
metadata.create_all(engine)

class Lead(BaseModel):
    name: str | None = None
    email: str | None = None
    company: str | None = None
    pitch: str

@app.post("/score")
async def score(lead: Lead):
    prompt = f"""
You are an assistant that scores sales leads from 0 to 100 (100 best).
Return ONLY a single JSON object and nothing else with:
{{"score": <integer 0-100>, "reasons": [<short strings>]}}
Lead:
Name: {lead.name}
Email: {lead.email}
Company: {lead.company}
Pitch: {lead.pitch}
"""
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = resp.text

    try:
        result = json.loads(text)
    except Exception:
        m = re.search(r"\{(.|\n)*\}", text)
        if m:
            result = json.loads(m.group(0))
        else:
            return {"score": None, "reasons": [], "raw": text}

    score_val = result.get("score")
    reasons_val = result.get("reasons", [])

    # Insert into DB
    with engine.begin() as conn:
        conn.execute(
            leads_table.insert().values(
                payload={"name": lead.name, "email": lead.email, "company": lead.company, "pitch": lead.pitch},
                score=score_val,
                reasons=reasons_val
            )
        )

    return {"score": score_val, "reasons": reasons_val}
