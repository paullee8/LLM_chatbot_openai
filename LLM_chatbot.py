import streamlit as st
from streamlit_chat import message
import os
import re
import copy
import time
import pandas as pd
import base64
import asyncio

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,PipelinePromptTemplate)
from langchain.chains.router import MultiRetrievalQAChain
from langchain.chains import LLMChain, ConversationChain

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA

from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
openai_api_base_value = os.environ.get('openai_api_base_value')
openai_api_key_value = os.environ.get('openai_api_key_value')
openai_api_version_value = os.environ.get('openai_api_version_value')
deployment_name_value = os.environ.get('deployment_name_value')
openai_api_type_value = os.environ.get('openai_api_type_value')


if 'pdf_name' not in st.session_state:
    st.session_state['pdf_name'] = []
if 'pdf_display' not in st.session_state:
    st.session_state['pdf_display'] = []
if 'pdf_path' not in st.session_state:
    st.session_state['pdf_path'] = []
if 'pdf_path_df' not in st.session_state:
    st.session_state['pdf_path_df'] = []


def display_pdf_1(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    return pdf_display 

def display_pdf_2(pdf):
    st.markdown(pdf, unsafe_allow_html=True)

async def testing():
    return 'the result is test'

async def embedding_pdf_path(file): 
    file_path = os.path.join(os.getcwd(), file.name)
    loader = UnstructuredPDFLoader(file_path)
    loaded_pdf = loader.load()
    pdf_info = Document(page_content=str(file.name), metadata={'path':file_path})
    st.session_state.pdf_path.append(pdf_info)
    db = FAISS.from_documents(st.session_state.pdf_path, HuggingFaceEmbeddings())
    st.session_state.pdf_path_df = db
    st.session_state.doc_yn = True
    return db

def clear_text():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['history'] = ''


class embedding_pdf():
    def __init__(self, file_path, user_input):
        self.file_path = file_path
        self.user_input = user_input
    def embed_pdf(self):
        print(self.file_path)
        self.loader = PyPDFLoader(self.file_path)
        self.loaded_pdf = self.loader.load()

        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = self.text_splitter.create_documents(self.loaded_pdf)

        self.embeddings = HuggingFaceEmbeddings()

        self.db = Chroma.from_documents(self.texts, self.embeddings)

        self.retriever = self.db.as_retriever()

        self.qa = self.RetrievalQA.from_chain_type(llm=llm_model, chain_type='stuff', retriever=self.retriever)
        return self.qa.run(self.user_input)


def model_seletion(temperature_level):
    global llm_model
    llm_model = AzureChatOpenAI(openai_api_base=openai_api_base_value, 
                    openai_api_version=openai_api_version_value, deployment_name=deployment_name_value, 
                    openai_api_key=openai_api_key_value, openai_api_type = openai_api_type_value,
                    temperature=temperature_level) 
    
    return llm_model, temperature_level

class douc_llm_result():
    def __init__(self, llm_model, user_input):
        self.llm_model = llm_model
        self.user_input = user_input
    def find_docu_name(self):
        self.docu_template = """
                    Help me see if '{search_input}', contains similar word in list:{docu_list}.
                    Return your answer in following template:
                    Yes: return the word in the list
                    """

        self.docu_prompt = PromptTemplate(input_variables=["search_input", "docu_list"], template=self.docu_template)

        self.llm_chain = LLMChain(
                            llm=self.llm_model,
                            prompt=self.docu_prompt,
                            verbose=True,
        )
        self.llm_answer = self.llm_chain.predict(search_input= self.user_input, docu_list = st.session_state.pdf_name)
        print('self.llm_answer',self.llm_answer)
        return(self.llm_answer)
    def calculate_similarity(self):
        self.docu_name = self.find_docu_name()
        self.results_with_scores = st.session_state.pdf_path_df.similarity_search_with_score(self.docu_name)
        for self.doc ,self.score in self.results_with_scores:
            if self.score<=1:
                self.answer = [self.doc.metadata,self.score]
                print(self.answer)
                break
            else: self.answer = ['no',100]
        return self.answer

def main_llm_result(llm_model,lang_mode,bot_charactor, prompt_mode,customized_prompt_input,user_input, history, df_lt=None):
    if bot_charactor == "General": roles = 'Experienced manager'
    elif bot_charactor == "Business": roles = 'Business analyst'
    elif bot_charactor == "Tech": roles = 'Programmer and data scientist'
#============================================================================================================
    full_template = """{introduction}, {example}, {input_template}"""
    full_prompt = PromptTemplate.from_template(full_template)

    introduction_template = """You are chatbot specificly for {people}. Response the answer in {lang} and provide example."""
    introduction_prompt = PromptTemplate.from_template(introduction_template)

    example_template = """{customized_prompt_input}"""
    example_prompt = PromptTemplate.from_template(example_template)

    df_template = """If there is information in {df_prompt_input}, prepare to use there as database to answer."""
    df_prompt = PromptTemplate.from_template(df_template)

    start_template = """# Chat history 
                        {chat_history}

                        Human: {human_input}
                        Chatbot: 
                        """
    start_prompt = PromptTemplate.from_template(start_template)

    input_prompts = [("introduction", introduction_prompt),("example", example_prompt),("input_template", start_prompt)]


    pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    llm_chain = LLMChain(
        llm=llm_model,
        prompt=pipeline_prompt,
        verbose=True,
    )

    llm_answer = (llm_chain.predict(people=roles, lang=lang_mode ,human_input=user_input, 
                                    customized_prompt_input=customized_prompt_input,
                                    df_prompt_input = df_lt, chat_history=history
                                    ) 
                    )
    return llm_answer

#=====================================
async def main():
    with st.sidebar:
        st.sidebar.button("Clear Text", on_click=clear_text)

        lang_mode = st.radio(
            ":earth_asia:",
            ("English", "Cantonese", "Other")
        )
        if lang_mode == 'Other':
            lang_mode = st.sidebar.text_input("Enter your preference")

        bot_charactor = st.sidebar.selectbox(
            "Type of chatbot",
            ("Tech", "General", "Business")
        )
        prompt_mode = st.radio(
            "Choose a Prompt style",
            ("Standard", "Customize")
        )
        if prompt_mode == 'Customize':
            customized_prompt_input = st.sidebar.text_input("Enter your customized prompt")
        else:
            customized_prompt_input = ''

        temperature_levels = st.slider(
            "Temperature level",  0.0, 1.0, (0.1), 0.01
            )

        df_lt = {}
        temp_pdf_lt = []
        uploaded_files = st.file_uploader("Upload a file", type=None , accept_multiple_files=True) 

        if len(uploaded_files) >=1:
            st.session_state.file_yn = True
            for file in uploaded_files:
                if 'csv' in file.type:
                    file_name = re.sub(r'\W+', '_', file.name)+'_df'
                    df = pd.read_csv(file)
                    df_lt[file_name] = df

                if '.xlsx' in file.name:
                    file_name = re.sub(r'\W+', '_', file.name)+'_df'
                    df = pd.read_excel(file)
                    df_lt[file_name] = df

                if 'pdf' in file.type:
                    file_name = re.sub(r'\W+', '_', file.name)
                    temp_pdf_lt.append(file_name)
                    if file_name not in st.session_state.pdf_name:
                        pdf = display_pdf_1(file)
                        st.session_state.pdf_name.append(file_name)
                        st.session_state.pdf_display.append(pdf)
                        embedd_db_result = asyncio.create_task(embedding_pdf_path(file))
                        print('1',type(embedd_db_result))
                
        else:
            st.session_state.file_yn = False
            st.session_state.doc_yn = False

    st.session_state.pdf_name = list(set(st.session_state.pdf_name) & set(temp_pdf_lt))
    df = pd.DataFrame(st.session_state.pdf_name, columns=['File stored'])

    st.sidebar.dataframe(df,hide_index=True,width=500)

    llm_model, temperature_levels = model_seletion(temperature_levels)
    #======================================================================================================
    if bot_charactor == 'General':
        st.title(f'Bot mode: :green[{bot_charactor}]' )
    if bot_charactor == 'Tech':
        st.title('Bot mode: :grey[IT :dog:]' )
    if bot_charactor == 'Business':
        st.title(f'Bot mode: :rainbow[{bot_charactor}]' )
    #======================================================================================================
    if st.session_state.doc_yn:
        if len(df_lt)>0:
            for df_key,df_value in df_lt.items():
                print(df_key)
                st.dataframe(df_value)

    user_input = st.chat_input("Say something")
    message('Hello! How can I assist you today?', is_user=False)   

    if user_input:
        if st.session_state.doc_yn:
            # await testing_result
            # print(testing_result.result())
            try:
                results_with_scores = st.session_state.pdf_path_df.similarity_search_with_score(user_input)
                user_query_check = douc_llm_result(llm_model,user_input)
                if user_query_check.calculate_similarity()[1]<1:
                    print('user_query_check',user_query_check.calculate_similarity()[0]['path'])
                    embedded_docu = embedding_pdf(user_query_check.calculate_similarity()[0]['path'], user_input)
                    print('the embedded docu',embedded_docu.embed_pdf())
            except Exception as e: 
                print('error:', str(e))

            pdf_summary = embedded_docu.embed_pdf()
            user_input = f"Human:{user_input}"+"\n"+f"Chatbot:{pdf_summary}"+"\n"

        output = main_llm_result(llm_model,lang_mode,bot_charactor, prompt_mode,customized_prompt_input,user_input,st.session_state.history,df_lt)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        st.session_state.history += (f"Human:{user_input}"+"\n"+f"Chatbot:{output}"+"\n")

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            

if __name__ == '__main__':

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state.past = []
    if 'history' not in st.session_state:
        st.session_state['history'] = ''
    if 'file_yn' not in st.session_state:
        st.session_state['file_yn'] = False
    if 'doc_yn' not in st.session_state:
        st.session_state['doc_yn'] = False


    asyncio.run(main())