# vector embedding of the chunks using ollama bge-m3 model
# cosine similarity-> value more -> vectors are similar ; value less -> not similar

import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Vector Embeddings for text in {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    # content['chunks'] is a list of dictionaries.
    # For each dictionary c inside content['chunks'], extract c['text'] and make a new list.

    # enumerate() is a built-in function in Python that lets you loop through an iterable and keep track of the index.      
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id  # adds a chunk_id to each chunk
        chunk['embedding'] = embeddings[i]
        # All the embeddings are made by create_embedding function,then here each chunk is assigned its embedding based on the index. 
        chunk_id += 1
        chunk_id += 1
        my_dicts.append(chunk) 
        # Finally, the modified chunk (with chunk_id and embedding) is added to a list called my_dicts.
# my_dicts is a list that contains all chunk info then chunk_id & embeddings added for each chunk.

df = pd.DataFrame.from_records(my_dicts)
# This persists the data(all embeddings) so that we don't have to create embeddings again and again.
joblib.dump(df, 'embeddings.joblib')

# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"]) 
# this is how we can send multiple texts together to create embeddings.The func will return a list with all the embeddings.

 #! ollama pull bge-m3
#Write this is the command prompt to download the model.

