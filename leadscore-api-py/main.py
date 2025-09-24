# main.py
# LeadScore Lite - main FastAPI app
# Features:
# - / health endpoint
# - /score endpoint: calls GenAI (Vertex or API key), parses JSON {score,reasons}
# - Saves to Cloud SQL Postgres using cloud-sql-python-connector + SQLAlchemy
# - Structured JSON logging and Error Reporting integration
# - Lazy GenAI client to avoid startup crashes

import os
import time
import json
import re
import logging
import sys
from typing import Optional, List, Any

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -- Logging setup (structured JSON)
try:
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None  # if missing, fallback to plain logging

logger = logging.getLogger("leadscore-api-py")
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    if jsonlogger:
        # produce JSON logs for Cloud Logging to parse easily
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
        json_formatter = jsonlogger.JsonFormatter(fmt)
        stream_handler.setFormatter(json_formatter)
    else:
        stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream_handler)

# Error Reporting client (optional)
_error_client = None
try:
    from google.cloud import errorreporting

    _error_client = errorreporting.Client()
except Exception:
    _error_client = None

# -- GenAI lazy client (avoid calling at import time)
_genai_client = None


def get_genai_client():
    """
    Lazy create a genai.Client().
    The google-genai SDK will auto-detect:
     - GEMINI_API_KEY env var (api-key flow), OR
     - Vertex flow when envs GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set and ADC is available.
    """
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai

            # Create the client (this may raise if not properly configured)
            _genai_client = genai.Client()
            logger.info("genai client initialized", extra={"mode": "genai.Client()"})
        except Exception as e:
            # Do not crash at import; raise later when the endpoint is used.
            logger.exception("genai_client_init_failed")
            raise
    return _genai_client


# -- Database (Cloud SQL) setup using cloud-sql-python-connector + SQLAlchemy
# Environment variables expected:
# INSTANCE_CONNECTION_NAME (project:region:instance)
# DB_USER, DB_PASS, DB_NAME

INSTANCE_CONNECTION_NAME = os.environ.get("INSTANCE_CONNECTION_NAME", "")
DB_USER = os.environ.get("DB_USER", "leaduser")
DB_PASS = os.environ.get("DB_PASS", "")
DB_NAME = os.environ.get("DB_NAME", "leadscore_db")

# Using lazy engine creation to avoid startup network calls
_engine = None


def get_engine():
    """
    Returns a SQLAlchemy engine that uses the Cloud SQL Python Connector.
    """
    global _engine
    if _engine is not None:
        return _engine

    # If instance connection name not provided, we won't attempt DB operations
    if not INSTANCE_CONNECTION_NAME:
        logger.warning("INSTANCE_CONNECTION_NAME not set; database operations will be disabled")
        return None

    try:
        from google.cloud.sql.connector import Connector, IPTypes
        import sqlalchemy
    except Exception:
        logger.exception("missing_db_dependencies")
        raise

    connector = Connector()

    def getconn():
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
            ip_type=IPTypes.PUBLIC,  # connector secures the connection even over public IP
        )
        return conn

    # SQLAlchemy engine using the connector's creator
    engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn, pool_size=5, max_overflow=2)
    _engine = engine

    # ensure table exists (safe to call)
    try:
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
        metadata.create_all(engine)
        logger.info("ensured leads table exists")
    except Exception:
        logger.exception("failed_create_table")

    return _engine


# Pydantic model for incoming lead
class Lead(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    pitch: str


# FastAPI app
app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://your-production-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS","PUT","DELETE"],
    allow_headers=["*"],
)


@app.get("/")
def hello():
    logger.info("health_check", extra={"service": os.environ.get("K_SERVICE", "leadscore-api-py")})
    return {"msg": "Hello — logs are JSON now!"}


# Add root POST endpoint so frontend POSTs to '/' work and preflight to '/' succeeds.
# This reuses the same score() handler logic by delegating to it.
@app.post("/")
async def submit_root(lead: Lead):
    """Accept POST / with same body as /score and delegate to the score handler.
    This ensures OPTIONS preflight to '/' is handled (CORS middleware + route exists) and avoids 404 for /.
    """
    return await score(lead)


# Middleware for structured request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        latency_s = time.time() - start
        logger.info(
            "request_finished",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": int(latency_s * 1000),
                "service": os.environ.get("K_SERVICE", "leadscore-api-py"),
            },
        )
        return response
    except Exception:
        logger.exception("request_failed", extra={"method": request.method, "path": request.url.path})
        if _error_client:
            try:
                _error_client.report_exception()
            except Exception:
                logger.exception("error_reporting_failed")
        raise


def parse_json_from_model_output(text: str) -> Any:
    """
    Tries to extract a JSON object from model output.
    Returns parsed object or raises ValueError.
    """
    # Try direct JSON
    try:
        return json.loads(text)
    except Exception:
        # try to find the first {...} block
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # fallback: raise so caller can handle
    raise ValueError("Could not parse JSON from model output")


@app.post("/score")
async def score(lead: Lead):
    """
    Endpoint: POST /score
    Body: { name, email, company, pitch }
    Returns: {"score": int, "reasons": [str, ...]}
    Also saves to Cloud SQL leads table if DB configured.
    """
    logger.info("score_request_received", extra={"lead_preview": {"name": lead.name, "company": lead.company}})

    # Build a strict prompt (ask model to return JSON only)
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

    # Call the genai client lazily (this may raise if not configured)
    try:
        client = get_genai_client()
    except Exception as e:
        # Provide helpful guidance without exposing secrets
        logger.exception("genai_client_not_ready")
        return JSONResponse(status_code=500, content={"error": "AI client not configured on server. Check envs or credentials."})

    # Call model - note: API differs slightly across SDK versions; use generate_content if available
    try:
        # prefer "models.generate_content" if available in SDK
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            model_text = resp.text
        else:
            # fallback to older method names
            resp = client.generate(prompt)  # may vary by SDK
            model_text = getattr(resp, "text", str(resp))
    except Exception:
        logger.exception("model_call_failed")
        return JSONResponse(status_code=502, content={"error": "Model request failed"})

    logger.info("model_raw_output", extra={"preview": model_text[:500]})

    # Parse JSON out of model output
    try:
        parsed = parse_json_from_model_output(model_text)
    except ValueError:
        logger.exception("model_output_parse_failed")
        # return raw for debugging in dev; in prod you may want to hide raw
        return JSONResponse(status_code=500, content={"error": "Model returned unparsable output", "raw": model_text})

    score_val = parsed.get("score") if isinstance(parsed, dict) else None
    reasons_val = parsed.get("reasons", []) if isinstance(parsed, dict) else []

    # Safeguard types
    try:
        if score_val is not None:
            score_val = int(score_val)
    except Exception:
        logger.warning("score_not_int", extra={"score_val": score_val})
        score_val = None

    if not isinstance(reasons_val, list):
        reasons_val = [str(reasons_val)]

    # Persist to DB if engine available
    try:
        engine = get_engine()
        if engine:
            import sqlalchemy

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
            with engine.begin() as conn:
                conn.execute(
                    leads_table.insert().values(
                        payload={
                            "name": lead.name,
                            "email": lead.email,
                            "company": lead.company,
                            "pitch": lead.pitch,
                        },
                        score=score_val,
                        reasons=reasons_val,
                    )
                )
            logger.info("saved_score", extra={"score": score_val})
        else:
            logger.info("db_not_configured", extra={"note": "INSTANCE_CONNECTION_NAME missing or engine not created"})
    except Exception:
        logger.exception("db_insert_failed")
        # do not fail the entire request if DB write fails; return the score to caller
        # optionally notify error reporting
        if _error_client:
            try:
                _error_client.report_exception()
            except Exception:
                logger.exception("error_report_failed")

    # Return cleaned response
    return {"score": score_val, "reasons": reasons_val}


# Optional: temporary endpoint to test Error Reporting (remove after testing)
@app.post("/__test_error_report")
def test_error_report():
    try:
        raise RuntimeError("test error for error reporting")
    except Exception:
        logger.exception("test_error_report_triggered")
        if _error_client:
            _error_client.report_exception()
        return {"ok": True, "message": "reported test error"}


# uvicorn entrypoint is handled by Dockerfile command: uvicorn main:app --host 0.0.0.0 --port $PORT
