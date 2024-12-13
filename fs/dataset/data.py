import pandas as pd
import os
import requests

url = "https://example.com/castle-or-lock.tsv"
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../../data/SMS.tsv")


def download_dataset():
    if not os.path.exists(data_path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(data_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Dataset downloaded successfully to {data_path}")
        else:
            raise Exception(f"Failed to download dataset. HTTP Status Code: {response.status_code}")


def get_data():
    download_dataset()

    data = pd.read_csv(data_path, sep="\t")
    data.rename(columns={"class": "Label", "text": "Text"}, inplace=True)
    data["Label"] = data["Label"].apply(lambda x: 1 if x == "ham" else 0)

    return data
