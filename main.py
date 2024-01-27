import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
import dotenv

# load env
dotenv.load_dotenv()
# set os env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


loader = WebBaseLoader("https://docs.smith.langchain.com/overview")

docs = loader.load()
embeddings = OpenAIEmbeddings()


# LANGCHAIN CORE CHAIN

# init llm
llm = ChatOpenAI()

# print(llm.invoke("how can langsmith help with testing?"))
output_parser = StrOutputParser()
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are world class technical documentation writer?"),
#         ("user", "{input}"),
#     ]
# )

# chain = prompt | llm | output_parser

# query = input("Hello! How can i assist you today?: ")
# print(f"Getting Response for the query: {query}\n\n...")
# print(chain.invoke({"input": query}))


# RETRIEVAL CHAIN
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vectors = DocArrayInMemorySearch(documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
)


document_chain = create_stuff_documents_chain(llm, prompt)

# message = document_chain.invoke(
#     {
#         "input": "how can langsmith help with testing?",
#         "context": [
#             Document(page_content="langsmith can let you visualize test results")
#         ],
#     }
# )


retrieval = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retrieval, document_chain)

response = retrieval_chain.invoke({"input": "How can langchain help with testing?"})

print(response["answer"])
# Output
# print(message)
