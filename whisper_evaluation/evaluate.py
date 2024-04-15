import os
import jiwer

with open('results_base.txt', 'r') as r, open('correct_transcriptions_base.txt', 'r') as c:
    # Initialize lists
    results = {}
    correct_transcriptions = {}

    # Read lines from the first file
    for line in r:
        # get id and sentance strip by tab
        id, sentence = line.strip().split('\t', 1)
        results[id] = sentence

    # Read lines from the second file
    for line in c:
        id, sentence = line.strip().split('\t', 1)
        correct_transcriptions[id] = sentence

# Compare the ID lists
if results.keys() == correct_transcriptions.keys():
    print("The IDs in both files are the same.")

    # Sort the by key
    results = dict(sorted(results.items()))
    correct_transcriptions = dict(sorted(correct_transcriptions.items()))

    results_list = list(results.values())
    correct_transcriptions_list = list(correct_transcriptions.values())

    transforms = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

    # Calculate the Word Error Rate
    wer = jiwer.wer(correct_transcriptions_list, results_list, truth_transform=transforms, hypothesis_transform=transforms)
    print("Word Error Rate: ", wer)
else:
    print("The IDs in the files are not the same.")