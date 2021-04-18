"""
Book Recommendation system based on tutorial from
https://heartbeat.fritz.ai/build-train-and-deploy-a-book-recommender-system-using-keras-tensorflow-js-b96944b936a7
using the dataset:
https://www.kaggle.com/zygmunt/goodbooks-10k
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# warnings.filterwarnings('ignore')
# %matplotlib inline

import tensorflow.keras as tf
from sklearn.model_selection import train_test_split


def main():
    # Import the data
    ratings_df = pd.read_csv("../data/ratings.csv")
    books_df = pd.read_csv("../data/books.csv")

    # Print some stats on the data
    print(ratings_df.shape)
    print(ratings_df.user_id.nunique())
    print(ratings_df.book_id.nunique())
    ratings_df.isna().sum()

    # Split to train/test sets
    Xtrain, Xtest = train_test_split(ratings_df, test_size=0.2, random_state=1)
    print(f"Shape of train data: {Xtrain.shape}")
    print(f"Shape of test data: {Xtest.shape}")

    # Get the number of unique entities in books and users columns
    nbook_id = ratings_df.book_id.nunique()
    nuser_id = ratings_df.user_id.nunique()

    # Book Input Network
    input_books = tf.layers.Input(shape=[1])
    embed_books = tf.layers.Embedding(nbook_id + 1, 15)(input_books)
    books_out = tf.layers.Flatten()(embed_books)

    # user input network
    input_users = tf.layers.Input(shape=[1])
    embed_users = tf.layers.Embedding(nuser_id + 1, 15)(input_users)
    users_out = tf.layers.Flatten()(embed_users)

    conc_layer = tf.layers.Concatenate()([books_out, users_out])
    x = tf.layers.Dense(128, activation='relu')(conc_layer)
    x_out = x = tf.layers.Dense(1, activation='relu')(x)
    model = tf.Model([input_books, input_users], x_out)

    opt = tf.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error')

    model.summary()
    
    hist = model.fit(
        [Xtrain.book_id, Xtrain.user_id],
        Xtrain.rating,
        batch_size=64,
        epochs=10,
        verbose=1,
        validation_data=([Xtest.book_id, Xtest.user_id], Xtest.rating)
    )

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(train_loss, color='r', label='Train Loss')
    plt.plot(val_loss, color='b', label='Validation Loss')
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    plt.show()

    model.save('model')

    # Extract embeddings
    book_em = model.get_layer('embedding')
    book_em_weights = book_em.get_weights()[0]
    book_em_weights.shape

    books_df_copy = books_df.copy()
    books_df_copy = books_df_copy.set_index("book_id")

    b_id = list(ratings_df.book_id.unique())
    b_id.remove(10000)
    dict_map = {}
    for i in b_id:
        dict_map[i] = books_df_copy.iloc[i]['title']

    out_v = open('vecs.tsv', 'w')
    out_m = open('meta.tsv', 'w')
    for i in b_id:
        book = dict_map[i]
        embeddings = book_em_weights[i]
        out_m.write(book + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

    out_v.close()
    out_m.close()

    # Making recommendations for user 100
    book_arr = np.array(b_id) # get all book ids
    user = np.array([100 for i in range(len(b_id))])

    pred = model.predict([book_arr, user])
    print(pred)

    pred = pred.reshape(-1) # reshape to single dimension
    pred_ids = (-pred).argsort()[0:5]
    print(pred_ids)

    print(books_df.iloc[pred_ids])

    web_book_data = books_df[["book_id", "title", "image_url", "authors"]]
    web_book_data = web_book_data.sort_values('book_id')
    web_book_data.head()

    web_book_data.to_json(r'web_book_data.json', orient='records')


if __name__ == '__main__':
    main()
