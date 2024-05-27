#所有跟AI大模型交互的代码
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def qa_agent(openai_api_key,memory,uploaded_file,question):
    model=ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=openai_api_key)
    #用户上传的文件会直接存储在内存中，把在内存中的内容写进入本地文件，然后把路径传给加载器
    #对用户上传的内容进行读取，read返回bytes,即内容的二进制数据
    file_content = uploaded_file.read()
    #存储二进制数据的临时路径
    temp_file_path = "temp.pdf"
    #python写入读取到的二进制数据，模式对应的是wb
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)
    #调用pdfloader,得到加载器实例
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    #对文档进行分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    #向量嵌入
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts,embeddings_model)
    #检索器
    retriever = db.as_retriever()
    #检索增强对话链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history":memory,"question":question})
    return response
