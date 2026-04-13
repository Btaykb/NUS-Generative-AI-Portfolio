import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from config import HUGGING_FACE_TOKEN


class RAGAgent:
    def __init__(self, pdf_path):
        self.hf_token = HUGGING_FACE_TOKEN
        self.chat_history = []

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100)
        self.llm = self._setup_llm()

        self.vector_db = self._ingest_document(pdf_path)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        self.chain = self._build_chain()

    def _setup_llm(self):
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=self.hf_token,
            max_new_tokens=512
        )
        return ChatHuggingFace(llm=llm_endpoint)

    def _ingest_document(self, pdf_path):
        """Loads and indexes the PDF."""
        loader = PyMuPDFLoader(pdf_path)
        data = loader.load()
        chunks = self.text_splitter.split_documents(data)
        return FAISS.from_documents(chunks, self.embeddings)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Constructs the LCEL RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a research assistant. Use the following context from the AlexNet paper to answer: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        setup_retrieval = RunnableParallel({
            "context": (lambda x: x["question"]) | self.retriever | self._format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        })

        return setup_retrieval | prompt | self.llm | StrOutputParser()

    def ask(self, query):
        """The primary method called by the Controller."""
        try:
            response = self.chain.invoke({
                "question": query,
                "chat_history": self.chat_history
            })

            # Update internal memory
            self.chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=response)
            ])

            return response
        except Exception as e:
            return f"RAG Error: {str(e)}"

    def clear_memory(self):
        self.chat_history = []
