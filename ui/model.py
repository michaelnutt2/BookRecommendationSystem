import tensorflow.keras as tf
import pandas as pd
import numpy as np

USER = '24499'
BOOK = '3628'

#ratings_df = pd.read_csv("data/ratings.csv") 
#books_df = pd.read_csv("data/books.csv")
#print(list(ratings_df.book_id.unique()))

# Loading the model
def load_model:
    return tf.models.load_model('model/model')

def new_prediction(USER, BOOK, new_model):
    # Take in user and book IDs
    book = np.array([int(BOOK)])
    user = np.array([int(USER)])

    # make prediction
    return new_model.predict([book, user])
    
