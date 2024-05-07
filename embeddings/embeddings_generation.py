from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from create_pinecone import vector_store
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings()

filepath = r'C:\Users\iamwi\OneDrive\Desktop\intern\mars_airline\documents\regulations.txt'

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=35,
    chunk_overlap=0
)

loader = TextLoader(
    file_path=filepath, 
    encoding='utf-8'
)

docs = loader.load_and_split(
    text_splitter=text_splitter
)

embeddings_dict = {}
filename = filepath.split('/')[-1].split('.')[0]
for doc_id, doc in docs.items():
    embeddings_dict[f"{doc_id}_{filepath}"] = embeddings.embed(doc).tolist()

vector_store.upsert(embeddings_dict)
# db = Chroma.from_documents(
#     docs,
#     embedding=embeddings,
#     persist_directory='emb'
# )