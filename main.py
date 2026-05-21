import uvicorn
from routers.server import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    uvicorn.run("routers.server:app", host="127.0.0.1", port=8011)
