#!/usr/bin/env python3

import re
from string import punctuation
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stop_words

class SimpleSummarize:
    def __init__(self, filename=None, k=None):
        self.txt = None
        self.word_tokens = None
        self.sent_tokens = None
        self.word_freq = None
        self.freq_dist = {}
        self.sent_scores = {}
        self.top_sents = None
        self.max_len = 40
        self.summary = ''
        self.scores = []
        self.english_stopwords = set(stopwords.words('english')) | stop_words
        if filename and k:
            self.load_file_from_disk(filename)
            self.tokenize()
            self.word_freq_dist()
            self.score_sentences()
            self.summarize(k)
    
    def load_file_from_disk(self, filename):
        with open(filename, "r") as file:
            self.txt = file.read().replace("\n", " ")
            self.txt = self.txt.replace("\'","")
    
    def tokenize(self):
        self.word_tokens = self.tokenizer(self.txt)
        self.sent_tokens = sent_tokenize(self.txt)

    def tokenizer(self,txt):
        txt = txt.lower()
        word_tokens = word_tokenize(txt.lower())
        word_tokens = [w for w in word_tokens if w not in self.english_stopwords and re.match('[a-zA-Z-][a-zA-Z-]{2,}', w)]
        return word_tokens
    
    def word_freq_dist(self):
        self.word_freq = nltk.FreqDist(self.word_tokens)
        most_freq_count = max(self.word_freq.values())
        for k,v in self.word_freq.items():
            self.freq_dist[k] = v/most_freq_count
    
    def score_sentences(self):
        for sent in self.sent_tokens:
            words = self.tokenizer(sent)
            for word in words:
                if word.lower() in self.freq_dist.keys():
                    if len(words) < self.max_len:
                        # if key does not exist add it and the freq_dist for the first word
                        if sent not in self.sent_scores.keys():
                            self.sent_scores[sent] = self.freq_dist[word.lower()]
                        else: 
                            # the key exists and we just add the freq_dist of the following words. 
                            # We are just summing up the freq_dists for the sentence
                            self.sent_scores[sent] += self.freq_dist[word.lower()]
    
    def summarize(self, k):
        self.top_sents = Counter(self.sent_scores)
        for t in self.top_sents.most_common(k):
            self.summary += t[0].strip()+'. '
            self.scores.append((t[1],t[0]))
    

def main():
    text_summary = SimpleSummarize(filename="CNNImpeachmentArticle.txt", k=3)
    print(text_summary.summary)
    print(text_summary.scores)


if __name__ == '__main__':
    main()
