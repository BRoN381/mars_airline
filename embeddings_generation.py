from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# read the file paths from the documents folder
filepaths = os.listdir('./documents')

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator='。',
    chunk_size=35,
    chunk_overlap=0
)

docs = []

for filepath in filepaths:
    loader = TextLoader(
        file_path='./documents/'+filepath, 
        encoding='utf-8',
    )
    docs.extend(loader.load_and_split(text_splitter=text_splitter))
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory='./emb_with_filename'
)