import subprocess
import re

libraries = [
    "pyannote.audio",
    "librosa",
    "torch",
    "textblob",
    "resemblyzer",
    "spectralcluster",
    "numpy",
    "pydub"
]

def get_version(library):
    try:
        result = subprocess.run(["pip", "show", library], stdout=subprocess.PIPE)
        output = result.stdout.decode('ISO-8859-1')
        version = re.search(r"Version: (.+)", output).group(1)
        return f"{library}=={version}"
    except Exception as e:
        print(f"Couldn't find version for {library}. Command output:\n{output}")
        return None

with open("requirements.txt", "w") as file:
    for library in libraries:
        version = get_version(library)
        if version:
            file.write(version + "\n")
