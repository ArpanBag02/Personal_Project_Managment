import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    The input question is given in the form of a job description. Perform the following steps:  
    1. Extract the name from the uploaded CV and greet the person formally according to the time of day (e.g., "Good Morning," "Good Afternoon," "Good Evening").  
    2. Calculate the similarity score between the CV and the job description and display the score with the following markers:  
        - **"🤝"** for scores below 60%,  
        - **"👍"** for scores between 60% to 80%,  
        - **"🔥"** for scores between 81% to 99%,  
        - **"👏"** for a score of 100%.  
        Present the score in **bold** format along with the respective marker.  
    3. Identify the skills that are missing in the CV but are required according to the job description. Provide clear and specific suggestions for improving those skills. For each lacking skill, recommend at least 2 relevant YouTube videos with their titles and thumbnails.  
    4. Suggest hands-on experiences project ideas and opportunities like internships and workshops for relevant skills.
    5. End the response with a randomly selected motivational quote related to success and hard work.  
    6. If the input is not a valid job description, respond politely by suggesting the user provide a valid job description, without giving an irrelevant answer.  

    Answer the question while adhering to these guidelines. If the required information is not available in the provided context, state: "Answer is not available in the context."
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Allow dangerous deserialization for trusted FAISS index files
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    print(response)
    st.write("Reply: ", response["output_text"])





def main():
    st.set_page_config("Personal Project Management")
    st.header("Personal Project Management by Resume")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Text area for job description
    user_question = st.text_area("Provide Job Description")

    # Get Result button
    if st.button("Get Result"):
        if user_question.strip():
            user_input(user_question)  # Process the input
        else:
            st.warning("Please provide a job description before clicking 'Get Result'.")


if __name__ == "__main__":
    main()
