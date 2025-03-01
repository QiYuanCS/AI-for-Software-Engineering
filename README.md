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
