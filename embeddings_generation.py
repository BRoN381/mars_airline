from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

filepaths = ['regulations.txt', 'schedule.txt', 'faq.txt']

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator='。',
    chunk_size=35,
    chunk_overlap=0
)

# 創建一個空的列表來存儲文檔和文件名的元組
docs = []

# 加載和分割每個文檔
for filepath in filepaths:
    loader = TextLoader(
        file_path=r'C:\Users\iamwi\OneDrive\Desktop\intern\mars_airline\documents\\'+filepath, 
        encoding='utf-8',
    )
    # 加載文檔和文件名元組
    docs.extend(loader.load_and_split(text_splitter=text_splitter))
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory=r'C:\Users\iamwi\OneDrive\Desktop\intern\mars_airline\emb_with_filename'
)