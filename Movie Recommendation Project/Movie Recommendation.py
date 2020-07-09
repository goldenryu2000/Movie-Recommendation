#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
import cv2
warnings.filterwarnings('ignore')


# # Load the Data file

# In[2]:


column_names = ["user_id" , "movie_id" , "rating" , "timestamp"]
u_data = pd.read_csv('ml-100k/u.data',sep = '\t' , names = column_names)


# # Load the item file

# In[3]:


movie_titles = movie_titles = pd.read_csv('ml-100k/u.item' , sep = "\|" , header = None)


# In[4]:


movie_titles =movie_titles[[0,1]]
movie_titles.columns = ["movie_id" ,"title"]


# In[5]:


u_data= pd.merge(u_data ,movie_titles , on = "movie_id")


# In[6]:


ratings = pd.DataFrame(u_data.groupby('title').mean()['rating'])


# In[7]:


ratings['num of ratings']  =pd.DataFrame(u_data.groupby('title').count()['rating'])


# In[8]:


moviemat = u_data.pivot_table(index="user_id" ,columns="title" , values="rating")


# In[9]:


def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name.title()] #get the movie from Movie Matrix.
    similar_to_movie = moviemat.corrwith(movie_user_ratings) #Find the correaltion
    corr_movie = pd.DataFrame(similar_to_movie , columns=["Correlation"]) #Make a dataframe
    corr_movie.dropna(inplace= True) #Drop the NaN values
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation' , ascending =False)
    return predictions


# In[10]:


print("*** Using MovieLens 100k dataset***")
m_name = input("Enter the Movie Name: ")
m_year = input("Enter The Year of the Movie: ")
movie_name = m_name+" ("+m_year+")"
try: 
    pred  = predict_movies(movie_name)
    print(pred.head())
except KeyError:
    print("Did you write the correct name or year?")

