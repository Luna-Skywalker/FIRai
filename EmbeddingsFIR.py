from generate_embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()
generator.generate_embeddings('./Datasets/testing1.csv', './FIRfaiss_db', 'myFIRIndex')