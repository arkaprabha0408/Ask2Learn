import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

jsons = os.listdir("jsons")

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
        print(content)
        print(f"Creating Embeddings for {json_file}")
        embeddings = create_embedding([c['text'] for c in content['chunks']])