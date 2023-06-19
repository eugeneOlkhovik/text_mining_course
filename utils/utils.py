import pandas as pd
import numpy as np
import glob
import config
import re
import string
from nltk.corpus import stopwords
import spacy
from nltk.tokenize import word_tokenize
import torch 
import time
import datetime


nlp = spacy.load('en_core_web_sm')


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def read_data(folder_path, is_truncated=True):
    df = pd.DataFrame(columns=['review'])
    file_list = glob.glob(f"{folder_path}/*.txt")
    if is_truncated:
        file_list = file_list[:500]
    for file in file_list:
        with open(file) as f:
            review_string = f.readlines()
            review = pd.DataFrame({'review': review_string})
            df = pd.concat([df, review], ignore_index=True)
    return df


def clean_html(text):
    return re.sub(config.REG_HASHTAG, '', text)


def clean_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def clean_stop_words(text):
    return [word for word in text.split() if
            word.lower() not in stopwords.words('english')]


def get_full_text_process(text):
    processed_text = clean_html(text)
    processed_text = clean_punctuation(processed_text)
    processed_text = clean_stop_words(processed_text)
    return processed_text


def get_lemmatization(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    filtered_sentence = [
        w for w in word_tokenize(text) if w not in stop_words
    ]
    return " ".join(list(filtered_sentence))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
