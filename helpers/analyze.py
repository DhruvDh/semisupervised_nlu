import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import os
# from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
# device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _filter(x): return " ".join(
    [word for word in x.lower().split() if word not in stopwords.words('english')])


# t = GPT2Tokenizer.from_pretrained('gpt2')



def clean(x):
    y = " ".join([t.strip() for t in x.split()]).strip()
    return re.sub(r'\s([?.,!"](?:\s|$))', r'\1', y)

def clean_and_get_playlist(row):
    playlist = ''

    if not pd.isna(row['playlist_owner']):
        playlist += ' ' + row['playlist_owner']

    if not pd.isna(row['playlist']):
        playlist += ' ' + row['playlist']

    if playlist == '':
        return np.NaN
    else:
        return clean(playlist)


def clean_and_get_restaurant(row):
    restaurant = ''

    if not pd.isna(row['restaurant_name']):
        restaurant += ' ' + row['restaurant_name']

    else:
        if not pd.isna(row['sort']):
            restaurant += ' ' + row['sort']

        if not pd.isna(row['cuisine']):
            restaurant += ' ' + row['cuisine']
        
        if not pd.isna(row['restaurant_type']):
            restaurant += ' ' + row['restaurant_type']

    spatial_relation_used = False
    if not pd.isna(row['spatial_relation']) and not pd.isna(row['poi']):
        restaurant += ' ' + row['spatial_relation'] + ' ' + row['poi']
        spatial_relation_used = True
    elif not pd.isna(row['poi']):
        restaurant += ' near ' + row['poi']

    if not pd.isna(row['city']):
        if not pd.isna(row['spatial_relation']) and not spatial_relation_used:
            restaurant += ' ' + row['spatial_relation'] + ' ' + row['city']
            spatial_relation_used = True
        else:
            restaurant += ' in ' + row['city']
        if not pd.isna(row['state']):
            restaurant += ' ' + row['state']
        if not pd.isna(row['country']):
            restaurant += ' ' + row['country']
    else:
        if not pd.isna(row['state']):
            if not pd.isna(row['spatial_relation']) and not spatial_relation_used:
                restaurant += ' ' + \
                    row['spatial_relation'] + ' ' + row['state']
                spatial_relation_used = True
            else:
                restaurant += ' in ' + row['state']
            if not pd.isna(row['country']):
                restaurant += ' ' + row['country']
        else:
            if not pd.isna(row['country']):
                if not pd.isna(row['spatial_relation']) and not spatial_relation_used:
                    restaurant += ' ' + \
                        row['spatial_relation'] + ' ' + row['country']
                    spatial_relation_used = True
                else:
                    restaurant += ' in ' + row['country']

    if restaurant == '':
        return np.NaN
    else:
        return clean(restaurant)


def clean_and_get_location(row):
    location = ''
    if not pd.isna(row['spatial_relation']):
        location += ' ' + row['spatial_relation']

    if not pd.isna(row['geographic_poi']):
        location += ' ' + row['geographic_poi']

    if not pd.isna(row['city']):
        location += ' ' + row['city']

    if not pd.isna(row['state']):
        location += ' ' + row['state']

    if not pd.isna(row['country']):
        location += ' ' + row['country']

    if location == '':
        return np.NaN
    else:
        return clean(location)


def clean_and_get_music(row):
    song = ''
    if not pd.isna(row['genre']):
        song += ' ' + row['genre']

    if not pd.isna(row['playlist']):
        song += ' ' + row['playlist']

    if not pd.isna(row['track']):
        song += ' ' + row['track']

    if not pd.isna(row['album']):
        song += ' ' + row['album']

    if not pd.isna(row['album']):
        song += ' ' + row['album']

    if not pd.isna(row['artist']):
        song += ' ' + row['artist']

    if song == '':
        return np.NaN
    else:
        return clean(song)

def clean_and_get_rating(row):
    rating = ''

    if not pd.isna(row['rating_value']):
        rating += ' ' + row['rating_value']
        if not pd.isna(row['rating_unit']):
            rating += ' ' + row['rating_unit']
        
        if not pd.isna(row['best_rating']):
            rating += ' out of ' + str(int(row['best_rating']))
    
    if rating == '':
        return np.NaN
    else:
        return clean(rating)


def clean_and_get_creative_work(row):
    work = ''

    if not pd.isna(row['object_type']):
        work += ' the ' + row['object_type']
        if not pd.isna(row['object_name']):
            work += ' ' + row['object_name']
    
    else:
        if not pd.isna(row['object_name']):
            work += ' ' + row['object_name']
    
    if work == '':
        return np.NaN
    else:
        return clean(work)


def clean_and_get_event (row):
    location = ''

    if not pd.isna(row['object_type']):
        location += ' the ' + row['object_type']
        if not pd.isna(row['movie_type']):
            location += ' for ' + row['movie_type']
        elif not pd.isna(row['movie_name']):
            location += ' for ' + row['movie_name']
    
    elif not pd.isna(row['movie_type']):
        location += row['movie_type']
        if not pd.isna(row['movie_name']):
            location += ' called ' + row['movie_name']
        
    elif not pd.isna(row['movie_name']):
        location += row['movie_name']
    
    if not pd.isna(row['spatial_relation']):
        location += ' ' + row['spatial_relation']
        if not pd.isna(row['object_location_type']):
            location += ' ' + row['object_location_type']
        elif not pd.isna(row['location_name']):
            location += ' ' + row['location_name']

    elif not pd.isna(row['object_location_type']):
        location += ' at a ' + row['object_location_type']

    if not pd.isna(row['location_name']):
        location += ' at ' + row['location_name']

    if location == '':
        return np.NaN
    else:
        return clean(location)

def get_data(path_to_intents = os.path.join('..', 'data', 'raw')):
    intents = os.listdir(path_to_intents)
    data = {}
    for intent in intents:
        data[intent] = {}
        data[intent]['df'] = pd.read_csv(os.path.join(
            path_to_intents, intent, intent + '.csv'))

    data['AddToPlaylist']['df']['playlist'] = data['AddToPlaylist']['df'].apply(
        clean_and_get_playlist, axis=1)

    data['BookRestaurant']['df']['place'] = data['BookRestaurant']['df'].apply(
        clean_and_get_restaurant, axis=1)

    data['GetWeather']['df']['location'] = data['GetWeather']['df'].apply(
        clean_and_get_location, axis=1)

    data['PlayMusic']['df']['music'] = data['PlayMusic']['df'].apply(
        clean_and_get_music, axis=1)

    data['RateBook']['df']['rating'] = data['RateBook']['df'].apply(
        clean_and_get_rating, axis=1)
        
    data['SearchCreativeWork']['df']['work'] = data['SearchCreativeWork']['df'].apply(
        clean_and_get_creative_work, axis=1)
        
    data['SearchScreeningEvent']['df']['event'] =  data['SearchScreeningEvent']['df'].apply(
        clean_and_get_event, axis=1)

    return data

path_to_intents = os.path.join('..', 'data', 'raw')
intents = os.listdir(path_to_intents)
questions = [
    ['Which playlist?', 'Where should I add?', 'What should I add to?', 'What was the playlist?'],
    ['Where do they want to eat?', 'Which place?', 'Which eatery?', 'Where?'],
    ['Where?', 'Which location?'],
    ['What should I play?', 'What do you want to hear?'],
    ['What much should I rate?', 'What is the rating?', 'How would they like to rate it?', 'Rate how much?',],
    ['Find what?', 'What should I look for?'],
    ['Find what?', 'What should I look for?'],
]

entities = [
    'playlist',
    'place',
    'location',
    'music',
    'rating',
    'work',
    'event'
]

train_data = {}
test_data = {}
split_at = 0.8

data = get_data()

for intent in intents:
    train_data[intent] = data[intent]['df'].head(int(len(data[intent]['df'])*split_at))

    test_data[intent] = data[intent]['df'].tail(int(len(data[intent]['df'])*(1.0-split_at)))

from random import shuffle
with open(os.path.join('..', 'data', 'train.txt'), 'w', encoding='utf-8') as f:
    train_list = []
    for i, intent in enumerate(intents):
        for index, row in train_data[intent].iterrows():
            for question in questions[i]:
                if row['text'].strip().endswith(".") or row['text'].strip().endswith("?"):
                    train_list.append("<s> " + row['text'] + ' ' + question + '\n' + str(row[entities[i]]) + " </s>\n\n")
                else:
                    train_list.append("<s> " + row['text'] + '. ' + question + '\n' + str(row[entities[i]]) + " </s>\n\n")
                    
    shuffle(train_list)
    f.writelines(train_list)

    



with open(os.path.join('..', 'data', 'test.txt'), 'w', encoding='utf-8') as f:
    test_list = []
    for i, intent in enumerate(intents):
        for index, row in test_data[intent].iterrows():
            for question in questions[i]:
                if row['text'].strip().endswith(".") or row['text'].strip().endswith("?"):
                    test_list.append("<s> " + row['text'] + ' ' + question + '\n' + str(row[entities[i]]) + " </s>\n\n")
                else:
                    test_list.append("<s> " + row['text'] + '. ' + question + '\n' + str(row[entities[i]]) + " </s>\n\n")
                     
    shuffle(test_list)
    f.writelines(test_list)     