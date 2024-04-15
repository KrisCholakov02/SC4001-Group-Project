import whisper
import os
import jiwer

# Load the Whisper model
model = whisper.load_model("large")

# Define the directory where the audio files are stored
audio_dir = "./LibriSpeech/test-other"

# Initialize a dictionary to store transcriptions
results = {}

# Initialize a dictionary to store correct transcriptions
correct_transcriptions = {}

# Walk through root directory
for dirpath, dirnames_level1, _ in os.walk(audio_dir):
    for dirname_level1 in dirnames_level1:
        print(dirname_level1)
        for dirpath_level2, _, filenames in os.walk(os.path.join(dirpath, dirname_level1)):
            for filename in filenames:
                # Check if the file is a .flac file
                if filename.endswith(".flac"):
                    # Create the full path to the audio file
                    audio_file = os.path.join(dirpath_level2, filename)
                    # Transcribe the audio file
                    result = model.transcribe(audio_file)
                    # Get the audio file name without the extension
                    audio_file = os.path.splitext(os.path.basename(audio_file))[0]
                    # Store the transcription in the dictionary
                    results[audio_file] = result
                # Check if the file is a .txt file
                if filename.endswith(".txt"):
                    # Iterate through the lines of the text file
                    with open(os.path.join(dirpath_level2, filename)) as file:
                        for line in file:
                            # The first word is the audio file identifier
                            audio_file = line.split()[0]
                            # The rest of the line is the transcription
                            transcription = " ".join(line.split()[1:])
                            # Store the transcription in the dictionary
                            correct_transcriptions[audio_file] = transcription

# Save the transcriptions to a text file in the format key \t value
# Create the file results.txt
with open("results_lg.txt", "w") as file:
    # Iterate through the keys of the dictionary
    for key in results.keys():
        # Write the key and the transcription to the file
        file.write(key + "\t" + results[key]['text'] + "\n")

# Do the same for the correct transcriptions
with open("correct_transcriptions_lg.txt", "w") as file:
    for key in correct_transcriptions.keys():
        file.write(key + "\t" + correct_transcriptions[key] + "\n")