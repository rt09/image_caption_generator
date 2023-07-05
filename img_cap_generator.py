# Importing required librarires/packages

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import argparse
import string
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.text import Tokenizer #for text tokenization
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense#Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from tqdm import tqdm
from tensorflow.python.client import device_lib




# loadint the data and performing data cleaning that is uppercase to lowecase, remove punctuation etc

def load_doc(filename):
  # Open file to read
   file = open(filename, 'r')
   text = file.read()
   file.close()
   return text

# get all images with their captions
def img_capt(filename):
   file = load_doc(filename)
   captions = file.split('\n')
   descriptions ={}
   for caption in captions[:-1]:
    img, caption = caption.split('\t')
    if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [ caption ]
    else:
            descriptions[img[:-2]].append(caption)
   return descriptions

#Data cleaning function will convert all upper case alphabets to lowercase, removing punctuations and words containing numbers
def txt_clean(captions):
    table = str.maketrans('', '', string.punctuation)
    clean_captions = {}
    for img, caps in captions.items():
        clean_captions[img] = []
        for img_caption in caps:
            img_caption = img_caption.replace("-", " ")
            descp = img_caption.split()
            descp = [wrd.lower() for wrd in descp]
            descp = [wrd.translate(table) for wrd in descp]
            descp = [wrd for wrd in descp if len(wrd) > 1]
            descp = [wrd for wrd in descp if wrd.isalpha()]
            img_caption = ' '.join(descp)
            clean_captions[img].append(img_caption)
    return clean_captions

def txt_vocab(descriptions):
  # To build vocab of all unique words
   vocab = set()
   for key in descriptions.keys():
    [vocab.update(d.split()) for d in descriptions[key]]
   return vocab
#To save all descriptions in one file

def save_descriptions(descriptions, filename):
   lines = list()
   for key, desc_list in descriptions.items():
        for desc in desc_list:
           lines.append(key + '\t' + desc )
   data = "n".join(lines)
   file = open(filename,"w")
   file.write(data)
   file.close()

dataset_images = "D:\image_caption_generator\Flickr8k_Dataset\Flicker8k_Dataset"

#to prepare our text data
filename ="D:\image_caption_generator\Flickr8k_text\Flickr8k.token.txt"

#loading the file that contains all data and map them into descriptions dictionary 
descriptions = img_capt(filename)
# print(type(descriptions))
# print("Length of descriptions =" ,len(descriptions))

#cleaning the descriptions
clean_descriptions = txt_clean(descriptions)
# print(clean_descriptions)
# print(type(clean_descriptions))

#to build vocabulary
vocabulary = txt_vocab(clean_descriptions)
# print("Length of vocabulary = ", len(vocabulary))

#saving all descriptions in one file
save_descriptions(clean_descriptions, "descriptions.txt")
# print((descriptions['1000268201_693b08cb0e.jpg']))

def extract_features(directory):
    model = Xception( include_top=False, pooling='avg' )
    features = {}
    for pic in tqdm(os.listdir(directory)):
           file = directory + "/" + pic
           image = Image.open(file)
           image = image.resize((299,299))
           image = np.expand_dims(image, axis=0)
          #image = preprocess_input(image)
           image = image/127.5
           image = image - 1.0
           feature = model.predict(image)
           features[pic] = feature
    return features


features = extract_features(dataset_images)
dump(features, open("features.p","wb"))
#to directly load the features from the pickle file.
features = load(open("features.p","rb"))
# print(features)

#load the data
def load_photos(filename):
   file = load_doc(filename)
   photos = file.split("\n")[:-1]
   return photos

def load_clean_descriptions(clean_descriptions, photos):
    
    # loading clean_descriptions
    trn_captions = {}
    for i in photos:
        if i in clean_descriptions:
            caps=clean_descriptions[i]
            trn_captions[i] = []
            for img_caption in caps:
               descp = img_caption.split()
               img_caption = ' '.join(descp)
               trn_captions[i].append(img_caption)
    return trn_captions

def load_features(photos):
  #loading all features
   all_features = load(open("features.p","rb"))
  #selecting only needed features
   features = {k:all_features[k] for k in photos}
   return features


filename= "D:\image_caption_generator\Flickr8k_text\Flickr_8k.trainImages.txt"

#train = loading_data(filename)
train_imgs = load_photos(filename)
# print(train_imgs)
train_descriptions = load_clean_descriptions(clean_descriptions, train_imgs)
# print(train_descriptions)
train_features = load_features(train_imgs)
# print((train_features))


#convert dictionary to clear list of descriptions
def dict_to_list(descriptions):
   all_desc = []
   for key in descriptions.keys():
       [all_desc.append(d) for d in descriptions[key]]
   return all_desc
#creating tokenizer class
#this will vectorise text corpus
#each integer will represent token in dictionary
def create_tokenizer(descriptions):
   desc_list = dict_to_list(descriptions)
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(desc_list)
   return tokenizer
# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1


# print(vocab_size)
#calculate maximum length of descriptions to decide the model structure parameters.
def max_length(descriptions):
   desc_list = dict_to_list(descriptions)
   return max(len(d.split()) for d in desc_list)

max_length_val = max_length(clean_descriptions)
# print(max_length_val)

#data generator, used by model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length):
   while 1:
      for key, description_list in descriptions.items():
          #retrieve photo features
           feature = features[key][0]
           inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature)
           yield [[inp_image, inp_seq], op_word]

def create_sequences(tokenizer, max_length, desc_list, feature):
   x_1, x_2, y = list(), list(), list()
  # move through each description for the image
   for desc in desc_list:
      # encode the sequence
      seq = tokenizer.texts_to_sequences([desc])[0]
      # divide one sequence into various X,y pairs
      for i in range(1, len(seq)):
          # divide into input and output pair
           in_seq, out_seq = seq[:i], seq[i]
          # pad input sequence
           in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          # encode output sequence
           out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          # store
           x_1.append(feature)
           x_2.append(in_seq)
           y.append(out_seq)
   return np.array(x_1), np.array(x_2), np.array(y)
#To check the shape of the input and output for your model
# [a,b],c = next(data_generator(train_descriptions, train_features, tokenizer, max_length))
# print(a.shape, b.shape, c.shape)
#((47, 2048), (47, 32), (47, 7577))


# define the captioning model
def define_model(vocab_size, max_length):
  # features from the CNN model compressed from 2048 to 256 nodes
   inputs1 = Input(shape=(2048,))
   fe1 = Dropout(0.5)(inputs1)
   fe2 = Dense(256, activation='relu')(fe1)
  # LSTM sequence model
   inputs2 = Input(shape=(max_length,))
   se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
   se2 = Dropout(0.5)(se1)
   se3 = LSTM(256)(se2)
  # Merging both models
   decoder1 = add([fe2, se3])
   decoder2 = Dense(256, activation='relu')(decoder1)
   outputs = Dense(vocab_size, activation='softmax')(decoder2)
  # merge it [image, seq] [word]
   model = Model(inputs=[inputs1, inputs2], outputs=outputs)
   model.compile(loss='categorical_crossentropy', optimizer='adam')
  # summarize model
   print(model.summary())
   # plot_model(model, to_file='model.png', show_shapes=True)
   return model

# train our model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length_val)
print(device_lib.list_local_devices())
model = define_model(vocab_size, max_length_val)
epochs =2
steps = len(train_descriptions)
# print(steps)
# creating a directory named models to save our models
os.makedirs("models", exist_ok=True)
# generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
# model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
# model.save("models/model_" + str(0) + ".h5")


for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length_val)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")


def extract_features(filename, model):
   try:
           image = Image.open(filename)
   except:
           print("ERROR: Can't open image! Ensure that image path and extension is correct")
   resz_image = image.resize((299,299))
   image = np.array(resz_image)
      # for 4 channels images, we need to convert them into 3 channels
   print(image.shape[2])
   image = np.expand_dims(image, axis=0)
   image = image/127.5
   image = image - 1.0
   feature = model.predict(image)
   return feature
def word_for_id(integer, tokenizer):
   for word, index in tokenizer.word_index.items():
      if index == integer:
       return word
   return None
def generate_desc(model, tokenizer, photo, max_length):
   in_text =""
   for i in range(max_length):
      sequence = tokenizer.texts_to_sequences([in_text])[0]
      sequence = pad_sequences([sequence], maxlen=max_length)
      pred = model.predict([photo,sequence], verbose=0)
      pred = np.argmax(pred)
      word = word_for_id(pred, tokenizer)
      if word is None:
         break
      in_text += ' ' + word
      if word == 'end':
         break
   return in_text
# max_length = 32
img_path="Flickr8k_Dataset/Flicker8k_Dataset/3646970605_d25c25340b.jpg"
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_0.h5')
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length_val)
print("nn")
print(description)
img.show()
