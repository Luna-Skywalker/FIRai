# Description: This file is the main file for the FastAPI server. It loads the FAISS index and initializes the FastAPI server for FIR analysis.
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# generator = EmbeddingGenerator()
# generator.generate_embeddings('./Datasets/testing1.csv', './FIRfaiss_db', 'myFIRIndex')

# Load embeddings database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"}, encode_kwargs={'normalize_embeddings': False})
db = FAISS.load_local(folder_path="./FIRfaiss_db",embeddings=embeddings, index_name="myFIRIndex")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define FastAPI endpoints
@app.post("/FIR")
async def predict(text: str):
    similar_response = db.similarity_search(text, k=6)
    page_contents_array = [doc.page_content for doc in similar_response]
    sections = [s[-8:].replace('_', ' ').strip() for s in page_contents_array]
    return JSONResponse(content=jsonable_encoder(sections), status_code=200)
    # return {"responses": sections}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #uvicorn FIR_IPC:app --reload