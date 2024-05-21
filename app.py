from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from PyPDF2 import PdfWriter
import gradio as gr
import os
from dotenv import load_dotenv
import openai

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ['my_secret']
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
loader = PyPDFLoader("/home/user/app/docs.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

#vector embedding
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template="""Your name is Sayno. You act like a successful rich man. 
Never reveal that it is an AI. Use the following pieces of context to answer the users question shortly.
Given the following summaries of a long document and a question, create a final answer with references.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}
You MUST answer in Korean and in Markdown format:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

chain_type_kwargs = {"prompt": prompt}

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    reduce_k_below_max_tokens=True,
    verbose=True,
)

query = "행복한 인생이란?"
result = chain(query)


for doc in result['source_documents']:
    print('내용 : ' + doc.page_content[0:100].replace('\n', ' '))
    print('파일 : ' + doc.metadata['source'])
    print('페이지 : ' + str(doc.metadata['page']))


def respond(message, chat_history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다.

    result = chain(message)

    bot_message = result['answer']

    for i, doc in enumerate(result['source_documents']):
        bot_message += '[' + str(i+1) + '] ' + doc.metadata['source'] + '(' + str(doc.metadata['page']) + ') '

    chat_history.append((message, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.

    return "", chat_history  # 수정된 채팅 기록을 반환합니다.

with gr.Blocks(theme='gstaff/sketch') as demo:  # gr.Blocks()를 사용하여 인터페이스를 생성합니다.
    gr.Markdown("# 안녕하세요. 세이노와 대화해보세요. \n 답변 생성에 조금 시간이 소요될 수 있습니다.")
    chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
    msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
    clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

    msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
    clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.
demo.launch(debug=True)  # 인터페이스를 실행합니다. 실행하면 사용자는 '입력' 텍스트박스에 메시지를 작성하고 제출할 수 있으며, '초기화' 버튼을 통해 채팅 기록을 초기화 할 수 있습니다.
