from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import os
import sys
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import Language
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma

# pip install langchain, llama-cpp, sentence-transformers, gitpython, chromadb required

git_repo_url= "https://github.com/KnudRonau/design-pattern-showcasing"
model_path="D:/LmStudio/Models/TheBloke/dolphin-2.6-mistral-7B-GGUF/dolphin-2.6-mistral-7b.Q6_K.gguf"
#query = "In less than 100 words, describe what happens in the MainFrame class"

loader = GitLoader(
    clone_url=git_repo_url,
    repo_path="./example_data/test_repo2/",
    file_filter=lambda file_path: file_path.endswith(".java"),
    branch="master",
)
java_repo = loader.load()
len(java_repo)

java_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=1000, chunk_overlap=200
)
texts = java_splitter.split_documents(java_repo)
len(texts)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(texts, embedding_function)

from langchain.chains.question_answering import load_qa_chain

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    n_batch=4096,
    n_ctx=8192,
    callback_manager=callback_manager,
    verbose=True,
)

chain = load_qa_chain(llm, chain_type="stuff")

chain.llm_chain.prompt.template

#docs = db.similarity_search(query)

while True:
    query = input("Enter a question regarding the repository: ")
    if query == "exit":
        break
    else:
        docs = db.similarity_search(query) 
        chain.invoke({'question': query, 'input_documents': docs})
