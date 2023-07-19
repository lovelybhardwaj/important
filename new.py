import gdown
import threading
def download_file():
    url = "https://drive.google.com/file/d/1bieH_2uOLgLEIqEypqLpJxDUZf7qLdLN/view?usp=sharing" 
    output = "frozen.pb"
    gdown.download(url, output, quiet=False)

# Start the download in a separate thread
download_thread = threading.Thread(target=download_file)
download_thread.start()
def download_file():
    url = "https://drive.google.com/file/d/1X5Am-bqdWyXPFCw4kUw2X1-WAWYqgBrO/view?usp=sharing" 
    output = "model.pth"
    gdown.download(url, output, quiet=False)

# Start the download in a separate thread
download_thread = threading.Thread(target=download_file)
download_thread.start()