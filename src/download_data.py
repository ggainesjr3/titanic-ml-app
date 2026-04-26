import os
import requests

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return
    
    print(f"Downloading {filename}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} successfully.")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    urls = {
        "data/train.csv": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "data/test.csv": "https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/test.csv"
    }
    
    for filename, url in urls.items():
        download_file(url, filename)
