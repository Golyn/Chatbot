import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow
import tflearn
import json
import numpy as np
import random
import speech_recognition as sr

stemmer = LancasterStemmer()
 

with open('vocabs.json') as file:
    data = json.load(file)
 
words = []
labels = []
patterns = []
tags = []


for intent in data['vocabs']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)  
        words.extend(wrds)
        patterns.append(wrds)
        tags.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)


output_empty = [0 for i in range(len(labels))]

training = []
output = []


for x, doc in enumerate(patterns):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = output_empty[:]
    output_row[labels.index(tags [x])] = 1

    training.append(bag)

    output.append(output_row)

training = np.array(training)
output = np.array(output)


network = tflearn.input_data([None, len(training[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(output[0]), activation='softmax')
network = tflearn.regression(network)

model = tflearn.DNN(network)

model.fit(training,output, n_epoch=1000, batch_size=8, show_metric=True)

#model.save('chat_model.tflearn')

def bag_of_words(sentence, words):
    bags = [0 for i in range(len(words))]
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(w) for w in s_words]
    
    for s in s_words:
        for i, w in enumerate(words):
            if w == s:
                bags[i] = 1

    return np.array(bags)

def chat():
    print('Hello ! (type quit to exit chatbot)')

    while True:
        r = sr.Recognizer()

    #detecting microphone as source
        with sr.Microphone() as source:
    # enable the individual to speak
            print('speak now')

    #initializing audio to listen to what will come out of the source
            audio = r.listen(source)

     # to handle 
     #try first block and if not successful do what's in the except block
            try:
        #initializing recognizer from google using google's API
        # converts the audio to text and prints it if is audible
                text = r.recognize_google(audio)

                print('You said ' + text)
                if text.lower() == 'quit':
                    break
# argmax gives u the index of the highest value
# the labels contains the tags
                result = model.predict([bag_of_words(inp, words)])[0] 
                result_index = np.argmax(result)

                tag = labels[result_index]

                for tg in data['vocabs']:
                    if tg['tag'] == tag:
                       responses = tg['responses']

#prints random choices of responses based on the 
                print(random.choice(responses))
            except:
                print("...")

chat()






    






