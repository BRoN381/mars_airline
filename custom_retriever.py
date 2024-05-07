from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class CustomRetriever(BaseRetriever):
    embeddings: Embeddings
    db: Chroma
    def get_relevant_documents(self, question: str):
        emb = self.embeddings.embed_query(question)
        results = self.db.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.5
        )
        relevant_docs = []
        for result in results:
            relevant_docs.append((result.metadata['source'].split('\\')[-1], result.page_content))
        
        return relevant_docs
    