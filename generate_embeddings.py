# Description: This file is used to generate embeddings using the sentence-transformers library.
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {"device": "cpu"}
        self.model_name = model_name
        self.model_kwargs = model_kwargs

    def generate_embeddings(self, data_path, save_folder, index_name):

        # Load preprocessed data
        loader = CSVLoader(file_path=data_path)
        documents = loader.load()

        # Generate embeddings
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=self.model_kwargs, encode_kwargs={'normalize_embeddings': False})

        # Build vector store and save
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(folder_path=save_folder, index_name=index_name)

# if __name__ == "__main__":
#     data_path = './Datasets/testing1.csv'
#     save_folder = './FIRfaiss_db'
#     index_name = "myFIRIndex"        
#     generator = EmbeddingGenerator()
#     generator.generate_embeddings(data_path, save_folder, index_name)