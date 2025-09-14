# break the entire text into chunks with start and end time,which helps us give time-stamps while searching

import whisper
import json
import os

model = whisper.load_model("medium")

audios = os.listdir("audios")

for audio in audios: 
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number, title)
        
        result = model.transcribe(audio = f"audios/{audio}", 
                              language="hi",
                              task="translate",
                              word_timestamps=False )
        
        chunks = []
        for segment in result["segments"]:
             chunks.append({"number": number, "title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
             json.dump(chunks_with_metadata,f)