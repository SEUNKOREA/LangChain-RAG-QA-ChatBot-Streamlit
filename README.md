# 🔍📚 RAG-based QA ChatBot with Google Generative AI

Build Google's Generative AI-based ChatBot that can be queried based on user's documents using the Langchain🦜 framework.

![Demo image 2024-01-05]("https://github.com/SEUNKOREA/LangChain-RAG-QA-ChatBot-Streamlit/blob/main/Demo%20image%202024-01-05.png)

## Update Logs
- 2023.01.15
    - You can use Gemini (beta) model.
    - The Gemini (beta) model allows general chat, but it is not possible to answer based on user documents. 
    - User document-based QA function will be soon updated.
    - The PaLM-2 model is capable of user document-based QA



## How to start
0. Clone this repository.
    ```
    git clone https://github.com/SEUNKOREA/LangChain-RAG-QA-ChatBot-Streamlit.git
    ```


1. Install necessary packages and libraries

    ```
    pip install -r requirements.txt
    ```


2. Google Cloud Project ID and Application Default Credentials (ADC) are required to invoke and use Google's Generative AI model with the API. Refer to this link and revise app.py to complete the preparation.


3. Start demo web page with streamlit framework.
    ```
    streamlit run app.py
    ```
