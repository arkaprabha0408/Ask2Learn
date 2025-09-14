#mp3 to text converter code

import whisper
model = whisper.load_model("medium")

result=model.transcribe(audio="audios/6_Python_Day6.mp3",language="hi",task="translate")
print(result["text"])
