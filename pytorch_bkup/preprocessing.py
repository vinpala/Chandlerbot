import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pandas as pd
import numpy as np

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)
# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadMovieConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

def loadSitcomConversations(path):
    sitcom_transcripts = [s for s in os.listdir(path) if s.endswith('.txt')]
    conversations = []
    for transcript in sitcom_transcripts:
        print(transcript)
        df = pd.read_csv(os.path.join(path,transcript), header=None, sep='+')
        print(" No. of lines in " + transcript + " => " + str(len(df)))
        count = 0
        for _,scene in df.groupby(2):
            conv =[] 
            for row in scene.values:
                conv.append(row[1])
            conversations.append(conv) 
            count += 1
        print(" No. of scenes written from  " + transcript + " => "+str(count)) 
    return conversations
    
    
# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

# Extracts pairs of sentences from sitcom conversations
def extractSentencePairs1(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation[i]
            targetLine = conversation[i+1]
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs
