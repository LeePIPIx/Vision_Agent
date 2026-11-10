"""Single-origin web UI and durable medical imaging task API."""
import os
from medical_app.api import create_app
app = create_app()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)
