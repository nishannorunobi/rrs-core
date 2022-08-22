def hello():
  print('hello')
#All Imports
import nltk
#nltk.download('all')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
import numpy as np
import gensim
import math
import logger as log
from gensim.utils import tokenize
def remove_one_length_words(sentence):
  tokens = list(tokenize(sentence))
  tokens = [i for i in tokens if len(i) > 1]
  new_sentence = ' '.join(tokens)
  return new_sentence
#Normalize the words ( lemmatize and stemming)
from nltk.stem import WordNetLemmatizer 
# Init the Wordnet Lemmatizer
def lematize_sentence(input_sentence):
  output_sentence = ''
  tokens = list(tokenize(input_sentence))
  lemmatizer = WordNetLemmatizer()
  for word in tokens:
    value = lemmatizer.lemmatize(word, 'v') #here default pos , verb is used
    output_sentence = output_sentence + ' '+ value
  return output_sentence
  
#Remove Stopwords
#Add your stop words to gensim default
from gensim.parsing.preprocessing import STOPWORDS
#all_stopwords_gensim = STOPWORDS.union(set(['.','It','and']))
from gensim.parsing.preprocessing import remove_stopwords
all_stopwords = gensim.parsing.preprocessing.STOPWORDS
#all_stopwords
def remove_stop_words(input_sentence):
  output_sentence = remove_stopwords(input_sentence)
  return output_sentence
#Remove non english (japan, german, etc) words
nltk.download('words')
words = set(nltk.corpus.words.words())

def remove_non_english_words(text):
  return " ".join(w for w in nltk.wordpunct_tokenize(text) \
           if w.lower() in words or not w.isalpha())
def get_tokens(doc):
  tokens = []
  clean_text = doc.replace('\n', ' ').replace('\r', '')
  clean_text = remove_non_english_words(clean_text)
  clean_text = clean_text.lower()
  clean_text = remove_stop_words(clean_text)
  clean_text = remove_one_length_words(clean_text)
  clean_text = lematize_sentence(clean_text)
  clean_text = remove_stop_words(clean_text)
  tokens = tokens + list(tokenize(clean_text))
  return tokens
def get_document_tokens(doc_list):
  doc_tokens = []
  for idx,doc in enumerate(doc_list):
    tokens = get_tokens(doc)
    doc_tokens.append(tokens)
  return doc_tokens

def get_document_frequency_dictionary(doc_dict):
  dictionary = {}
  for key in doc_dict.keys():
    doc_text_dict = doc_dict.get(key,'')
    doc_text = doc_text_dict.get('text','')
    # log.error(doc_text)
    # exit()
    doc_tokens = get_tokens(doc_text)
    token,counts = np.unique(doc_tokens,return_counts=True)
    dictionary[key] = token,counts
  return dictionary

def get_vocabulary_list(document_freq_dict):
  all_words = []
  for key in document_freq_dict.keys():
    tokens,counts = document_freq_dict.get(key)
    all_words = np.append(all_words,tokens)
  all_words = set(all_words)
  return all_words
def get_corpus_frequency_dictionary(doc_list):
  all_words = []
  doc_tokens = get_document_tokens(doc_list)
  for index,tokens in enumerate(doc_tokens):
    all_words = np.append(all_words,tokens)
  
  words,counts = np.unique(all_words,return_counts=True)
  corpus_dict = dict(zip(words,counts))
  return corpus_dict
def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    cos = -10
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)

    return cos
def get_category_level(value):
  cat_val = value*10
  cat_val = math.floor(cat_val)
  cat_val*=10
  return cat_val