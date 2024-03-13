from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp

from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)
from langchain_community.vectorstores.chroma import Chroma
import re
from langchain_core.output_parsers import StrOutputParser

# pip install langchain, llama-cpp, sentence-transformers, gitpython, chromadb required

"""
Repository for example usage:
git_repo_url= "https://github.com/KnudRonau/design-pattern-showcasing"
"""

#method to load the repository
def load_repo(_git_repo: str):

    pattern = r'^https'
    if re.match(pattern, _git_repo):
        java_repo = load_online_repo(_git_repo, 1)
    else:
        java_repo = load_local_repo(_git_repo)
    
    coding_separators = [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                # Split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                "\nobject ",
                "\nstruct ",
                "\nabstract ",
                # Split along function definitions
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nfunction ",
                "\nlet ",
                "\nfn ",
                "\nreturn ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\ndo ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\nwhen ",
                "\nelse ",
                "\ndefault ",
                "\nbegin ",
                "\nrescue ",
                "\nunless ",
                "\nloop ",
                "\nmatch ",
                "\ncontinue ",
                "\nforeach ",
                "\nbreak ",
                "\ndo while ",
                "\nassembly ",
                # Split by exceptions
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
                # PROTO
                "\nmessage ",
                "\nimport ",
                "\nservice ",
                "\nsyntax ",
                "\noption ",
                # RST
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                "\n\n.. *\n\n",
                # markdown
                "\n#{1,6} ",
                "```\n",
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # html
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
                # sol
                "\npragma ",
                "\nusing ",
                "\ncontract ",
                "\nlibrary ",
                "\nconstructor ",
                #cobol
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                "\nINPUT-OUTPUT SECTION.",
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",

    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=coding_separators
    )
    texts = text_splitter.split_documents(java_repo)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(texts, embedding_function)
    return db


#helper method to load the repository of repo is an url
def load_online_repo(_git_repo_url: str, _repo_counter: int):
    repo_path="./example_data/test_repo" + str(_repo_counter) + "/"
    java_repo = None
    try:
        loader = GitLoader(
            clone_url=_git_repo_url,
            repo_path=repo_path,
            branch="master",
        )
        java_repo = loader.load()
    except ValueError:
        print("Another repository is already cloned at this path. Trying another path.")
        return load_online_repo(_git_repo_url, _repo_counter + 1)
    return java_repo


#helper method to load the local repository
def load_local_repo(_repo_path: str):
    loader = GitLoader(
        repo_path=_repo_path,
        branch="master",
    )
    java_repo = loader.load()
    return java_repo


#method to load the model
def load_llm(_model_path: str, _callback_manager: CallbackManager, _temperature: float):
    formatted_model_path = _model_path.replace("\\", "/")
    formatted_model_path = formatted_model_path.strip('"')
    print(formatted_model_path)
    print(formatted_model_path)

    llm = LlamaCpp(
        model_path=formatted_model_path,
        n_gpu_layers=-1,
        # larger n_batch and n_ctx provide better results but are slower
        n_batch=2048,
        n_ctx=4096,
        callback_manager=_callback_manager,
        verbose=True,
        temperature=_temperature,
    )
    return llm


#method to setup the LLM and DB
def setup(_db: str, _llm: str, _temperature: float):
    try:
        db = load_repo(_db)
    except:
        error_message = "The repository could not be loaded, please try again."
        print(error_message)
        db = None
        return error_message
    try:
        local_llm = load_llm(_llm, CallbackManager([StreamingStdOutCallbackHandler()]), _temperature)
    except:
        error_message = "The LLM could not be loaded, please try again."
        print(error_message)
        local_llm = None
        return error_message

    global vector_database
    vector_database = db
    global llm
    llm = local_llm
    return "The repository and LLM have been setup successfully"


#return the response from the LLM
def llm_reponse(_query: str):
    if(vector_database != None and llm != None):
        docs = vector_database.similarity_search(_query, k=4)
        parser = StrOutputParser()

        prompt = PromptTemplate(
            template="You are a software engineering expert. Use the following pieces of context to answer the question at the end. Give an in depth and thorough answer to the question. If you don't know the answer, state that you don't know." + 
                "Context: {context_str} Question: {question} Answer: ", 
            input_variables=["context_str", "question"]
        )
      
        prompt_and_model = prompt | llm | parser
        response = prompt_and_model.invoke({'context_str': docs, 'question': _query})
        return response
    else:
        return "The LLM and DB are not setup yet"
