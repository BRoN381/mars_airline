from langchain.chat_models import ChatOpenAI
from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from custom_retriever import CustomRetriever
from dotenv import load_dotenv
import json

#===============================
# setup
#===============================
app = FastAPI()
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
chat = ChatOpenAI()
memory = ConversationBufferMemory(memory_key='history', return_messages=True)
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory='./emb_with_filename',
    embedding_function=embeddings
)


#===============================
# api
#===============================
@app.post("/new_session")
def clear_session():
    memory.clear()
    return {"message": "Conversation buffer memory cleared."}

@app.post("/get_response")
def read_str(input: dict):
    try:
        question = input['question']
        retriever = CustomRetriever(
            embeddings=embeddings,
            db=db
        )
        # first time asking question
        if memory.buffer_as_str == '':
            summary = question
        else:
            pre_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template("以下是你和客人之前的對話紀錄，請你根據這些對話紀錄以及客人最新的問題生成一個客人想問的完整的問句，請勿自行回答問題，只要統整出客人想問的完整的問句就好。另外請勿重複提到先前的對話內容，以利後續使用RAG的功能提取跟最新問句相關的資訊。"),
                    MessagesPlaceholder(variable_name='history'),
                    HumanMessagePromptTemplate.from_template(f'以下是客人最新詢問的問題：\n {question}')
                ]
            )
            pre_chain = LLMChain(
                llm=chat,
                prompt=pre_prompt,
                memory=memory
            )

            reply = pre_chain({
                'question': question
            })
            summary = reply['text']
        relativeInfo = retriever.get_relevant_documents(summary)
        textList = [text[1] for text in relativeInfo]
        sourceList = [text[0] for text in relativeInfo]
        humanprompt = ''
        documentTranslate = {'regulations.txt': '公司規章專區', 'schedule.txt': '航班時刻表', 'faq.txt': '常見問題專區'}
        for i in range(len(textList)):
            source = documentTranslate[sourceList[i].split('/')[-1]]
            humanprompt += f'{textList[i]}。\n以上資訊的來源是{source}\n'
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("你是一個在火星航空工作的客服Henry，請利用以下提供的資訊以及資訊的來源檔案名稱回答客人的問題，並且告知客人你是由哪一篇文章獲得之資訊，如果客人想要了解更多可以去查閱相關文件。注意，請判斷客人的問題是否是通用的一般性問題例如打招呼，或是涉及火星航空的規定或航班資訊需要額外的資料提供才能回答。如果是通用的問題請忽略公司規章按照客服的身分回答，如果是需要額外資料才能回答的請參考提供的公司規章回答。另外如果問題需要額外的資料才能回答，問題的內容卻和以下規章資訊相關性不高，一定要聲明\"我沒找到相關資訊\"，如果有找到訊息可以回答則忽略。"),
                MessagesPlaceholder(variable_name='history'),
                HumanMessagePromptTemplate.from_template(f'以下是公司規章：\n{humanprompt} \n以下是客人詢問的問題：\n {question}')
            ]
        )

        chain = LLMChain(
            llm=chat,
            prompt=prompt,
            memory=memory
        )

        reply = chain({
            'question': question
        })

        return {"reply": reply['text']}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")