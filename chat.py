
import keras
import nltk
import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
from googlesearch import *
from IPython.display import display, HTML
import webbrowser
import requests
import billboard
import time
from pygame import mixer
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

data_file=open('/content/sample_data/intents.json').read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('/content/sample_data/word.pkl','wb'))
pickle.dump(classes,open('/content/sample_data/class.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]

    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1

    training.append([bag,output_row])

random.shuffle(training)
training=np.array(training, dtype=object)
X_train=list(training[:,0])
y_train=list(training[:,1])

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
weights=model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
model.save('/content/sample_data/model.h5',weights)

from keras.models import load_model
model = load_model('/content/sample_data/model.h5')
intents = json.loads(open('/content/sample_data/intents.json').read())
words = pickle.load(open('/content/sample_data/word.pkl','rb'))
classes = pickle.load(open('/content/sample_data/class.pkl','rb'))


#Predict
def lemtokens(tokens):
  return [lemmatizer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)

def clean_up(text):
    return lemtokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]

    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def gbu_search(user_response):
  f=open('/content/gbu.txt','r',errors='ignore')
  raw_doc=f.read()
  raw_data=raw_doc.lower()
  sent_tokens=nltk.sent_tokenize(raw_doc)
  word_tokens=nltk.word_tokenize(raw_doc)
  robo1_response=''
  sent_tokens.append(user_response)
  TfidfVec=TfidfVectorizer(tokenizer=clean_up,stop_words='english')
  tfidf=TfidfVec.fit_transform(sent_tokens)
  vals=cosine_similarity(tfidf[-1],tfidf)
  idx=vals.argsort()[0][-2]
  flat=vals.flatten()
  flat.sort()
  req_tfidf=flat[-2]
  if(req_tfidf==0):
    robo1_response=robo1_response+"I am sorry! I don't understand you"
    return robo1_response
  else:
    robo1_response=robo1_response+sent_tokens[idx]
    return robo1_response




def get_response(return_list,intents_json):

    if len(return_list)==0:
        tag='noanswer'
    else:
        tag=return_list[0]['intent']
    if tag=='search':
      ques=input("What would you like to know")
      return gbu_search(ques)
    if tag=='datetime':
        print(time.strftime("%A"))
        print (time.strftime("%d %B %Y"))
        print (time.strftime("%I:%M:%S %p"))

    if tag=='google':
        query=input('Enter query...')
        search_results = search(query, tld="co.in", num=1, stop=1, pause=2)
        html_content = ""
        for url in search_results:
         html_content += f'<a href="{url}" target="_blank">{url}</a><br>'
        display(HTML(html_content))

    if tag=='news':
        main_url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=c05545c2d1ba450589899b25b64c69fc"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = []

        for ar in article:
            results.append([ar["title"],ar["url"]])

        for i in range(10):
            print(i + 1, results[i][0])
            print(results[i][1],'\n')

    if tag=='timer':
      minutes_seconds=input('Minutes to timer..')
      if ':' in minutes_seconds:
        mins, secs = minutes_seconds.split(':')
        mins = int(mins)  # Convert minutes part to integer
        secs = int(secs)  # Convert seconds part to integer
      else:
        mins = 0  # No minutes given, so default to 0
        secs = int(minutes_seconds)  # Assume the input is in seconds

    # Total seconds = (minutes * 60) + seconds
      total_seconds = mins * 60 + secs

    # Countdown logic
      while total_seconds:
        mins, secs = divmod(total_seconds, 60)  # Convert seconds back to minutes and seconds
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")  # Print the timer and overwrite the previous output
        time.sleep(1)  # Wait for 1 second
        total_seconds -= 1  # Decrease the time
      print('Time\'s up!')  #


    list_of_intents= intents_json['intents']
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result

def response(text):
    return_list=predict_class(text,model)
    response=get_response(return_list,intents)
    return response

while(1):
    x=input()
    print("GREETER: ",end="")
    print(response(x))
    if x.lower() in ['bye','goodbye','get lost','see you']:
        break


#Self learning
ans=input('Help me Learn?')
if ans.lower()=='yes':
  tag=input('Please enter general category of your question  ')
  flag=-1
  for i in range(len(intents['intents'])):
    if tag.lower() in intents['intents'][i]['tag']:
        intents['intents'][i]['patterns'].append(input('Enter your message: '))
        intents['intents'][i]['responses'].append(input('Enter expected reply: '))
        flag=1

  if flag==-1:


    intents['intents'].append (
      {'tag':tag,
       'patterns': [input('Please enter your message')],
       'responses': [input('Enter expected reply')]})

  with open('/content/sample_data/intents.json','w') as outfile:
        outfile.write(json.dumps(intents,indent=4))
