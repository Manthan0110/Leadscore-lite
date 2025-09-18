from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def hello():
    return {"msg": "Hello Cloud Run! Kay CHallay Jevla Ka?"}