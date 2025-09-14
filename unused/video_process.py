#mp4 to mp3 converter code

import os
import subprocess
files=os.listdir("videos")
for file in files:
    tutorial_number=file.split(".")[0].split("Day")[1]
    file_name=file.split(".")[0]
    print(tutorial_number,file_name)

    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{tutorial_number}_{file_name}.mp3"])    