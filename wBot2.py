#!/usr/bin/env python
# coding: utf-8

# #**Chat with Multiple PDF's PaLM 2 + Pinecone + LangChain**
# 

# #**Step 01: Install All the Required Packages**
from IPython import get_ipython
# In[1]:


# get_ipython().system('pip install langchain')
# get_ipython().system('pip install pinecone-client')
# get_ipython().system('pip install pypdf')


# In[16]:


#get_ipython().system('pip install -q google-generativeai')


# #**Step 02: Import All the Required Libraries**

# In[2]:


from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
import os
import sys


# #**Step 03: Load the PDF Files**

# In[3]:


# get_ipython().system('mkdir pdfs')


# In[4]:


#get_ipython().system('gdown 1hPQlXrX8FbaYaLypxTmeVOFNitbBMlEE -O pdfs/yolov7paper.pdf')
#get_ipython().system('gdown 1vILwiv6nS2wI3chxNabMgry3qnV67TxM -O pdfs/rachelgreecv.pdf')


# #**Step 04: Extract the Text from the PDF's**

# In[5]:
loader = PyPDFDirectoryLoader("pdfs")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBHINqTvLyNBmt3VCSBHYaqxsxfBjBlHDk'
embeddings=GooglePalmEmbeddings()
query_result = embeddings.embed_query("Hello World")
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '9e40adbc-2c2e-4f66-a1b5-85ad2a1f716a')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')
# index_name = "my-index" # put in the name of your pinecone index here
pinecone.init(
      api_key=PINECONE_API_KEY,  # find at app.pinecone.io
      environment=PINECONE_API_ENV  # next to api key in console
  )
index_name = "my-index" # put in the name of your pinecone index here

docsearch = Pinecone.from_existing_index(index_name, embeddings)
query = "Rachel Green Qualification"
docs = docsearch.similarity_search(query, k=3)
docs
llm = GooglePalm(temperature=0.1)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
prompt_template  = """
  you are Q a high school revesion asistant,
  Use the following piece of context to answer the question. Please provide a detailed response for each of the question.

  {context}

  Question: {question}

  """
prompt = PromptTemplate(template = prompt_template , input_variables=["context", "question"])


def init_f():
  print("waiting....", flush=True)
  # loader = PyPDFDirectoryLoader("pdfs")
  # data = loader.load()


  # In[8]:


  print(data)


  # #**Step 05: Split the Extracted Data into Text Chunks**

  # In[9]:


  # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)


  # In[10]:


  # text_chunks = text_splitter.split_documents(data)


  # In[13]:


  print("Length of chunk === ",len(text_chunks))


  # In[11]:


  # text_chunks[2]


  # # In[12]:


  # text_chunks[3]


  # In[14]:


  # os.environ['GOOGLE_API_KEY'] = 'AIzaSyBHINqTvLyNBmt3VCSBHYaqxsxfBjBlHDk'


  # #**Step 06:Downlaod the Embeddings**

  # In[17]:


  # embeddings=GooglePalmEmbeddings()


  # In[18]:


  # query_result = embeddings.embed_query("Hello World")


  # In[19]:


  # print("Length", len(query_result))


  # #**Step 07: Initializing the Pinecone**

  # In[20]:


  # PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '9e40adbc-2c2e-4f66-a1b5-85ad2a1f716a')
  # PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')


  # In[21]:


  # initialize pinecone
  # pinecone.init(
  #     api_key=PINECONE_API_KEY,  # find at app.pinecone.io
  #     environment=PINECONE_API_ENV  # next to api key in console
  # )
  # index_name = "my-index" # put in the name of your pinecone index here


  # #**Step 08: Create Embeddings for each of the Text Chunk**

  # In[22]:


  # docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)


  # #If you already have an index, you can load it like this
  # 

  # In[23]:


  # docsearch = Pinecone.from_existing_index(index_name, embeddings)


  # #**Step 09: Similarity Search**

  # In[24]:


  query = "YOLOv7 outperforms which models"


  # In[47]:


  query = "Rachel Green Qualification"


  # In[27]:


  docs = docsearch.similarity_search(query, k=3)


  # In[28]:


  docs


  # #**Step 10: Creating a Google PaLM Model Wrapper**

  # In[29]:


  # llm = GooglePalm(temperature=0.1)


  # In[30]:


  # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


  # #Custom Prompts

  # In[38]:


  # prompt_template  = """
  # Use the following piece of context to answer the question. Please provide a detailed response for each of the question.

  # {context}

  # Question: {question}

  # """


  # In[39]:


  # prompt = PromptTemplate(template = prompt_template , input_variables=["context", "question"])


  # #**Step 11: Q/A**

  # In[31]:


  query = "give a summary of the text"


  # In[32]:

  # print("I was here 1 +++++++++", qa.run(query))
  


  # In[33]:


  query = "who is the writer of the text"


  # In[34]:


  # print("Passed through this poin -----------",qa.run(query))


  # In[35]:


  query = "what are some of the application areas of what is mentioned in the text"


  # In[36]:


  # print('Finished @@@@@@@@@',qa.run(query))


  # In[37]:
chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)


# In[41]:


query = "YOLOv7 outperforms which models"



  # while True:
def ask(question):
  # user_input = input(f"Input Prompt: ")
  # docsearch = Pinecone.from_existing_index(index_name, embeddings)

  # llm = GooglePalm(temperature=0.1)
  # chain_type_kwargs = {"prompt": prompt}
  # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)

  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

  if question == 'exit':
    print('Exiting')
    sys.exit()
  if question == '':
    return "ask a question"
  
  result = qa({'query': question})
  if result:
    print(f"Answer: {result['result']}")
    return f"Answer: {result['result']}"
  else:
    return("Result might be empty")
  
    # return "Their was a problem"
  

# #**Step 12: Q/ A with Custom Prompt**

# In[40]:



# In[42]:


# qa.run(query)


# In[ ]:




