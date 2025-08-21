backend e giya

python -m venv venv  
venv\Scripts\activate
pip install -r .\requirements.txt
python -m app.core.init_db  
uvicorn app.main:app --reload