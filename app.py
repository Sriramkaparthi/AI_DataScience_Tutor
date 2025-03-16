import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import easyocr
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Load API key securely
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 API key is missing! Please add it to .env file or Streamlit secrets.")
    st.stop()

st.title("🧠 AI Data Science Tutor")

# Sidebar - Input previous session ID or start new
st.sidebar.subheader("🔑 Session Management")
session_id_input = st.sidebar.text_input("Enter Session ID (or leave empty for new):")

if session_id_input:
    st.session_state.session_id = session_id_input
else:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

st.sidebar.write(f"**Current Session ID:** `{st.session_state.session_id}`")

# Initialize memory for conversation history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

memory = st.session_state.memory

# Initialize AI model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

# Sidebar - Conversation History
st.sidebar.subheader("🕒 Conversation History")
for msg in memory.chat_memory.messages:
    st.sidebar.write(f"**{msg.type.capitalize()}:** {msg.content}")

# User input for text-based questions
user_input = st.text_input("❓ Ask your Data Science question:")

# File uploader for image-based questions (OCR)
uploaded_file = st.file_uploader("📤 Upload an image with your question:", type=["jpg", "jpeg", "png"])

# If an image is uploaded, extract text using OCR
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        reader = easyocr.Reader(['en'])
        image_np = np.array(image)
        extracted_text = reader.readtext(image_np, detail=0)

        if extracted_text:
            user_input = " ".join(extracted_text)
            st.success("✅ Text extracted from image successfully.")
        else:
            st.warning("⚠️ No readable text found in the image.")
    except Exception as e:
        st.error(f"❌ OCR Error: {e}")

# Define AI prompt template
prompt = PromptTemplate(
    input_variables=["user_input", "history"],
    template="""You are an AI-powered Data Science tutor.
    Provide a detailed and well-structured response to the user's question.
    If relevant, include examples, definitions, and practical applications.
    If the user question is unclear, ask follow-up questions for clarification.
    
    **Conversation History:**  
    {history}  
    
    **User Question:**  
    {user_input}  
    
    **AI Response:**  
    """
)

# Corrected method: Replace LLMChain with RunnablePassthrough + LLM pipeline
chain = prompt | llm | RunnablePassthrough()

# Submit button to process user input
if st.button("Submit"):
    if user_input:
        try:
            history = "\n".join([
                f"User: {msg.content}" if msg.type == "human" else f"Tutor: {msg.content}"
                for msg in memory.chat_memory.messages
            ])

            # Generate response from AI
            response = chain.invoke({
                "user_input": user_input,
                "history": history
            })

            # Ensure response is a string before displaying
            response_text = response if isinstance(response, str) else response.get("text", "Sorry, I couldn't process that.")

            # Display response
            st.write("**Tutor:**")
            st.write(response_text)

            # Add to memory
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response_text)

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter a question or upload an image.")
