# backend.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uvicorn
import json
import numpy as np

# Import your existing LangGraph app
from main import app as langgraph_app

# Create FastAPI app
app = FastAPI(title="CSV Analyzer API")

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

@app.post("/analyze")
async def analyze_csv(
        csv_file: UploadFile = File(...),
        user_query: str = Form(...)
):
    try:
        # Validate file type
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")

        # Save uploaded file temporarily
        file_path = f"temp_{csv_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(csv_file.file, buffer)

        print(f"📁 Processing file: {csv_file.filename}")
        print(f"❓ User query: {user_query}")

        # Run your LangGraph pipeline
        state = {
            "csv_path": file_path,
            "user_query": user_query,
        }

        final_state = langgraph_app.invoke(state)

        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Prepare response data and convert numpy types
        response_data = {
            "formatted_response": final_state.get("formatted_response", ""),
            "llm_response": final_state.get("llm_response", ""),
            "retrieved_context": convert_numpy_types(final_state.get("retrieved_context", [])),
            "error": final_state.get("error", "")
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Clean up temp file if it exists
        file_path = f"temp_{csv_file.filename}"
        if os.path.exists(file_path):
            os.remove(file_path)

        error_msg = f"Analysis error: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "CSV Analyzer API is running"}

@app.get("/")
async def root():
    return {"message": "CSV Analyzer API - Use /analyze endpoint"}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)