import webview
import threading
import os

def start_server():
    # Start Django server
    os.system("python manage.py runserver 127.0.0.1:8000")

if __name__ == '__main__':
    # Run Django in background
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()

    # Launch desktop window
    webview.create_window("Sign Language Recognizer", "http://127.0.0.1:8000")
    webview.start(gui="qt")