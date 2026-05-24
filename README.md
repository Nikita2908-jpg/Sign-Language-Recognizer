cd C:\Users\nikita\Downloads\sign_language_recognizer

.\venv\Scripts\activate
pip install -r requirements.txt

download ASL alphabet datadet from kaggle 
run the app for training using "python train.py"

python desktop_app.py

python manage.py makemigrations
python manage.py migrate
python manage.py runserver  # http://127.0.0.1:8000/
