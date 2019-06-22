# coding: utf-8

import pickle
import time

import numpy as np

class Log():
    def log(self, text, log_time=False):
        print('log: %s' % text)
        if log_time:
            print('time: %s' % time.asctime(time.localtime(time.time())))

def pickle_save(object, file_path):
    f = open(file_path, 'wb')
    pickle.dump(object, f)

def pickle_load(file_path):
    f = open(file_path, 'rb')
    return pickle.load(f)

# read all genres of each movie as a list: [genre], all in int32
def read_genre_file(file_path, genre_cnt):
    genre_data = np.loadtxt(fname=file_path, delimiter='\t')
    item_genres = []
    genre_items = [[] for i in range(genre_cnt)]
    for i in range(len(genre_data)):
        item_genres.append(int(genre_data[i]))
        genre_items[int(genre_data[i])].append(i)
    return item_genres, genre_items