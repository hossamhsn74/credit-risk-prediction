from fastapi import FastAPI
from app.api.predict import router as predict_router

app = FastAPI()
app.include_router(predict_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello"}
