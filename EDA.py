import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import os

train= pd.read_csv("C:\\Users\\Ashish/Downloads/NAOP_new.csv" ,encoding='latin-1')

train.head()

#Let’s see if there are any null values present in our dataset:
train.isnull().sum()

     
     
# Remove all columns from column index 3 to 24
train.drop(train.iloc[:, 3:], inplace = True, axis = 1) 

#Removing index column
train.drop(train.iloc[:, 0:1], inplace = True, axis = 1) 


train.isnull()


train['Type'].value_counts()





# cleaning data 
import re
stop_words = []
with open("C:/Users/Ashish/Desktop/datasci_assignment/text mining\\stop.txt") as f:
    stop_words = f.read()


# splitting the entire string by giving separator as "\n" to get list of 
# all stop words
stop_words = stop_words.split("\n")


"this is awsome 1231312 $#%$# a i he yu nwj"

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    i = re.sub("/&foo(\=[^&]*)?(?=&|$)|^foo(\=[^&]*)?(&|$)/"," ",i)
    i = re.sub("\\b(\\w+)(?:\\W+\\1\\b)+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

"This is Awsome 1231312 $#%$# a i he yu nwj".split(" ")

cleaning_text("This is Awsome 1231312 $#%$# a i he yu nwj")


              
# testing above function with sample text => removes punctuations, numbers
cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .. are you?")


train.Posts = train.Posts.apply(cleaning_text)


# Separate A class dataset
A_class = train[train.Type=='A']

#making wordcloud
post_word=''
stopwords = set(stop_words) 
for val in A_class.Posts: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
    
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    post_word += " ".join(tokens)+" "

from wordcloud import WordCloud, STOPWORDS 

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(post_word) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


###############################################



# Separate b class dataset
B_class = train[train.Type=='B']

#making wordcloud
post_word=''
  stopwords = set(STOPWORDS) 
for val in B_class.Posts: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
    
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    post_word += " ".join(tokens)+" "

from wordcloud import WordCloud, STOPWORDS 

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(post_word) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 




########################################




# Separate c class dataset
C_class = train[train.Type=='C']

#making wordcloud
post_word=''
  stopwords = set(STOPWORDS) 
for val in C_class.Posts: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
    
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    post_word += " ".join(tokens)+" "

from wordcloud import WordCloud, STOPWORDS 

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(post_word) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 





########################




# Separate c class dataset
D_class = train[train.Type=='D']

#making wordcloud
post_word=''
  stopwords = set(STOPWORDS) 
for val in D_class.Posts: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
    
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    post_word += " ".join(tokens)+" "

from wordcloud import WordCloud, STOPWORDS 

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(post_word) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

# handing the imbalance data by over sampling

from sklearn.utils import resample

df_minority_oversampled = resample(D_class,replace=True,n_samples=5697,random_state=123)
df_oversampled = pd.concat([df_minority_oversampled])
df_oversampled.Type.value_counts()
df_minority_oversampled = resample(B_class,replace=True,n_samples=5697,random_state=123)
df_oversampled = pd.concat([df_minority_oversampled,df_oversampled])
df_minority_oversampled = resample(C_class,replace=True,n_samples=5697,random_state=123)
df_oversampled = pd.concat([df_minority_oversampled,df_oversampled])
df_oversampled = pd.concat([A_class,df_oversampled])
df_oversampled.Type.value_counts()



