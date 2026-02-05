#. 
initial steps to set up the environment
import all libraries
from langchain_community import document_loaders # chunks
from langchain.embeddings import OpenAIEmbeddings # embedding model
from langchain.vectorstores import FAISS # vector database
from os import path
from dotenv import load_dotenv # environment variables
from langchain.text_splitter import CharacterTextSplitter # text chunking (recursive/non-recursive/semantic)





