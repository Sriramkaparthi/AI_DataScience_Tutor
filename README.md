# 🧠 AI Data Science Tutor  

An AI-powered Data Science tutor built using **Streamlit, Google Gemini API, LangChain, and EasyOCR**. This application allows users to ask text-based questions or upload images containing questions, which are then processed using OCR.

## ✨ Features  
- **Conversational AI:** Uses Google Gemini API to generate responses.  
- **Session Management:** Unique session IDs to track conversation history.  
- **OCR Support:** Extracts text from images using EasyOCR.  
- **Memory Storage:** Stores conversation history using LangChain's `ConversationBufferMemory`.  
- **User-Friendly UI:** Built with Streamlit for a clean and interactive experience.  

## 🛠 Tech Stack  
- **Python 3.11**  
- **Streamlit** (Frontend)  
- **Google Gemini API** (AI Model)  
- **LangChain** (Conversational AI)  
- **EasyOCR** (Image-based question extraction)  
- **PIL, NumPy** (Image Processing)

📝 Usage
Ask a Question: Enter your data science question in the text box and click "Submit".
Upload an Image: Upload an image with a question, and OCR will extract the text.
View Responses: The AI tutor provides structured responses.
Check Conversation History: View previous messages in the sidebar.
