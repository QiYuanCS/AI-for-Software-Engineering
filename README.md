
AI Software Engineering Project: Programming Assistant Development - Python, Streamlit, OpenAI API, Docker

• Developed the Ducky programming assistant based on GPT and LLM, providing code suggestions, code review, and debugging support, with natural language code modification capabilities.


• Adopted modular development, incrementally integrating features such as natural language code modification and Quick Chat semantic search enhancements.


• Integrated semantic search with RAG (Retrieval-Augmented Generation) based on The Pragmatic Programmer, optimizing Quick Chat with improved UI interaction, providing relevant page numbers and screenshots as references, using embedding techniques to extract text semantics, and nearest-neighbor search to associate context and return relevant code/knowledge.


# ducky-ui
An example streamlit Python client using LLM technology

# start ducky docker
docker build -t cs5740-project6 .

docker run -p 8501:8501 cs5740-project6

docker run -p 8501:8501 -v .:/app cs5740-project6

# start ducky
python3.12 -mvenv .venv

source .venv/bin/activate

pip install -r requirements.txt

streamlit run 🏠_Ducky.py

# url
URL: http://0.0.0.0:8501
