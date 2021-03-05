# Deep NLP

# importing the  libraries

import numpy as np
import tensorflow as tf
import re  #used to clean the text
import time

###### Data preprocessing ############

lines = open('movie_lines.txt', encoding = 'utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

id2line = {}

# creatiung a dict for id and lines 
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]] = _line[4]
        
    
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))
    

questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# cleaning the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"he ' s", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"that ' s", "that is ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"how's", "how is",text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"can ' t", "cannot", text)
    text = re.sub(r"\'re", "are",text)
    text = re.sub(r"\'ll","will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'d","would", text)
    text = re.sub(r"there's","there is", text)
    text = re.sub(r"it's","it is", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"[-()\"#/@;:<>~?...|.,!$%]", "",text)
    return text

# cleaning the question    
clean_questions= []
for line in questions:
    clean_questions.append(clean_text(line))

# cleaning the answer
clean_answers= []
for line in answers:
    clean_answers.append(clean_text(line))

    
# count the occurence of word
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word]+=1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else: 
            word2count[word]+=1
            
    
# appling threshold
threshold = 20
questionswords2int = {}
word_numbers=0

for word,count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_numbers;
        word_numbers+=1
        
word_numbers=0
answerswords2int = {}
for word, count in  word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_numbers
        word_numbers+=1

# adding the last tokens to these two dict: 
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1
    
for token in tokens:
    answerswords2int[token] = len(answerswords2int)+1    

# inverse the answerswords2int dict
answersints2words ={w_i:w for w,w_i in answerswords2int.items()}


## addign EOS(End-of-String) to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
 

## Translating all the questions and answers into integers 
## and replacing all the words by integer 
question_to_int = []
for question in clean_questions:
    ints =[]
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    question_to_int.append(ints)

answer_to_int = []
for answer in clean_answers:
    ints =[]
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answer_to_int.append(ints)
    
# sorting the question and answer by the length of question
    ## this will helps us to speed up the traning
sorted_clean_questions =[]
sorted_clean_answers =[]
for length in range(1,25+1):
    for i in enumerate(question_to_int):
        if(len(i[1]))==length:
            sorted_clean_questions.append(question_to_int[i[0]])
            sorted_clean_answers.append(answer_to_int[i[0]])
            

            
            

########    Building seq2seq model   ##############
# creating the placeholder for the inputa and targets
        
def model_inputs():
    inputs = tf.placeholder(tf.int32,[None,None],name='input')
    targets = tf.placeholder(tf.int32,[None,None],name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_probe = tf.placeholder(tf.float32, name='keep_probe')
    return inputs,targets, lr, keep_probe









        
        
            

    
    
    
