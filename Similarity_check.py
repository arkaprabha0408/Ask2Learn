# Here, we load the saved embeddings and check similarity of a query we ask with the saved embeddings.

import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response


df = joblib.load('embeddings.joblib')
# df contains all the chunks with their embeddings.

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# question_embedding returns a list of lists.The outer list only contains 1 element ie.the inner list.So to take the inner list we do [0].

# df['embedding']=gives the entire 'embedding' column from the dataframe df.
# df['embedding'].values= converts the series to a numpy array.
# vstack= It takes a sequence of arrays and stacks them on top of each other (row-wise), producing a single 2D NumPy array (matrix).

#print(np.vstack(df['embedding'].values))
# As cosine_similarity function takes 2D arrays as input.

#print(np.vstack(df['embedding']).shape)
#(174, 1024)-> 174 chunks, each having an embedding of size 1024(as bge-m3 model produces embeddings of dimension 1024.)

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# the o/p of cosine_similarity is a 2D array with shape (174,1).
# he o/p is a column vector, where each row is the cosine similarity score between that chunk and the query.

# flatten() converts this 2D array to a 1D array,where each element is the cosine similarity score between that chunk and the query.

top_results = 5
# similarities.argsort()=after sorting the values of array in ascending order,it returns the indices
max_indx = similarities.argsort()[::-1][0:top_results]
# Reverses the array to get descending order and takes the top 30 indices.
new_df = df.loc[max_indx] 
# new_df contains the top 30 chunks which are most similar to the query.

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

prompt = f'''I am an engineer learning web development . Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
I asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide me to go to that particular video. If I ask any unrelated questions, tell me that I can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)


