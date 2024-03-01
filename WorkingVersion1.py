from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import shutil
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
from langchain.chains.question_answering import load_qa_chain
import re
from langchain_core.output_parsers import StrOutputParser

# pip install langchain, llama-cpp, sentence-transformers, gitpython, chromadb required

"""
git_repo_url= "https://github.com/KnudRonau/design-pattern-showcasing"
model_path="D:/LmStudio/Models/TheBloke/dolphin-2.6-mistral-7B-GGUF/dolphin-2.6-mistral-7b.Q6_K.gguf"
query = "In less than 100 words, describe what happens in the MainFrame class"

"""

#method to load the repository
def load_repo(_git_repo: str):

    pattern = r'^https'
    if re.match(pattern, _git_repo):
        java_repo = load_repo_helper(_git_repo, 1)
    else:
        java_repo = load_local_repo(_git_repo)
    #java_repo = load_repo_helper(git_repo, 1)

    print(len(java_repo))

    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=1000, chunk_overlap=200
    )
    texts = java_splitter.split_documents(java_repo)
    len(texts)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(texts, embedding_function)
    return db


#helper method to load the repository of repo is an url
def load_repo_helper(_git_repo_url: str, _repo_counter: int):
    repo_path="./example_data/test_repo" + str(_repo_counter) + "/"
    java_repo = None
    try:
        loader = GitLoader(
            clone_url=_git_repo_url,
            repo_path=repo_path,
            file_filter=lambda file_path: file_path.endswith(".java"),
            branch="master",
        )
        java_repo = loader.load()
    except ValueError as e:
        print("Another repository is already cloned at this path. Trying another path.")
        return load_repo_helper(_git_repo_url, _repo_counter + 1)
    return java_repo


#helper method to load the local repository
def load_local_repo(_repo_path: str):
    loader = GitLoader(
        repo_path=_repo_path,
        #file_filter=lambda file_path: file_path.endswith(".java"),
        branch="master",
    )
    java_repo = loader.load()
    return java_repo


#method to load the model
def load_llm(_model_path: str, _callback_manager: CallbackManager):
    formatted_model_path = _model_path.replace("\\", "/")
    formatted_model_path = formatted_model_path.strip('"')
    print(formatted_model_path)
    print(formatted_model_path)

    llm = LlamaCpp(
        model_path=formatted_model_path,
        n_gpu_layers=-1,
        n_batch=4096,
        n_ctx=8192,
        callback_manager=_callback_manager,
        verbose=True,
    )
    return llm

#method to setup the LLM and DB
def setup(_db: str, _llm: str):
    try:
        db = load_repo(_db)
    except Exception as e:
        print("Error loading the repository")
        return    
    local_llm = load_llm(_llm, CallbackManager([StreamingStdOutCallbackHandler()]))
    global vector_database
    vector_database = db
    global llm
    llm = local_llm

#return the response from the LLM
def llm_reponse(_query: str):
    if(vector_database != None and llm != None):
        docs = vector_database.similarity_search(_query)
        parser = StrOutputParser()

        prompt = PromptTemplate(
            template="You are a software engineering expert. Use the following pieces of context to answer the question at the end. Give an in depth and thorough answer to the question. If you don't know the answer, state that you don't know." + 
                "Context: {context_str} Question: {question} Answer: ", 
            input_variables=["context_str", "question"]
        )

        prompt_and_model = prompt | llm
        response = prompt_and_model.invoke({'context_str': docs, 'question': _query})
        return parser.invoke(response)
    else:
        return "The LLM and DB are not setup yet"

"""
    #main method to run the program
def main():
        #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        #db = load_repo(input("Enter the git repository url: "))

        #llm = load_llm(input("Enter the model path: "), callback_manager) 

        """"""
        chain = load_qa_chain(llm, chain_type="stuff", verbose=True, callback_manager=callback_manager)

        chain.llm_chain.prompt.template

        while True:
            query = input("Enter a question regarding the repository: ")
            if query == "exit":
                break
            else:
                docs = db.similarity_search(query) 
                chain.invoke({'question': query, 'input_documents': docs})
                """"""
        parser = StrOutputParser()

        prompt = PromptTemplate(
            template="You are a software engineering expert. Use the following pieces of context to answer the question at the end. Give an in depth and thorough answer to the question. If you don't know the answer, state that you don't know." + 
                "Context: {context_str} Question: {question} Answer: ", 
            input_variables=["context_str", "question"]
        )

        prompt_and_model = prompt | llm
        
        while True:
            query = input("Enter a question regarding the repository: ")
            if query == "exit":
                break
            else:
                docs = db.similarity_search(query) 
                response = prompt_and_model.invoke({'context_str': docs, 'question': query})
                print(parser.invoke(response))
                print('')

        


if __name__ == "__main__":
        main()

        
"""