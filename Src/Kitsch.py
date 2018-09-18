# -*- coding: utf-8 -*-
import codecs
import copy
import locale
import os
import random
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.layers import CuDNNGRU
from keras.layers import Dropout, Dense, Bidirectional
from keras.models import Sequential
from keras.models import load_model

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')


class Kitsch:
    def __init__(self, data_path, max_len, step, file_path="../ModelWeights/bestModelSoFar.hdf5"):
        self.__lower_map = {
            ord(u'I'): u'ı',
            ord(u'İ'): u'i',
        }
        self.__data_path = data_path
        self.__max_len = max_len
        self.__step = step
        self.__model = None
        self.__file_path = file_path

    def main(self):
        self.__first_lines, text = self.read_data()
        text = self.clean_data(text)
        self.set_vocab(text)

        # text = text[:int(len(text) / 1000)]

        X, Y = self.reshape_data(text)
        self.__model = self.build_model()
        self.__model = self.load_model_from_file()
        callbacks = self.get_callbacks()
        self.train(callbacks, X, Y)

    def on_epoch_end(self, epoch, logs):
        seed_idx = random.randint(10, len(self.__first_lines) - 3)
        seed = '\r\n'.join(self.__first_lines[seed_idx:seed_idx + 3]) + '\r\n'
        seed = seed.translate(self.__lower_map).lower()
        seed = self.clean_data(seed)
        starter_seed = copy.deepcopy(seed)
        print('----- Will Generate after the seed: \n"' + seed + '"\n-----')

        for diversity in [0.5, 1.0, 1.5]:
            print()
            print('----- diversity:', diversity)
            seed = starter_seed
            sys.stdout.write(starter_seed)

            for i in range(128):
                x = np.zeros((1, self.__max_len, len(self.__vocab)))
                for t, char in enumerate(seed):
                    x[0, t, self.__char_indices[char]] = 1.

                preds = self.__model.predict(x, verbose=0)[0]
                next_index = self.generate_sample(preds, diversity)
                next_char = self.__indices_char[next_index]

                seed = seed[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    def load_model_from_file(self):
        print('Loading...')
        if os.path.exists(self.__file_path):
            return load_model(self.__file_path)
        else:
            return self.__model

    def train(self, callbacks_list, X, Y, epochs=1000, verbose=1):
        for epoch in range(1, epochs):
            print()
            print('-' * 100)
            print('Iteration', epoch)
            self.__model.fit(X, Y, batch_size=128, epochs=1, callbacks=callbacks_list, verbose=verbose)

    def set_vocab(self, text):
        self.__vocab = sorted(list(set(text)))
        print(self.__vocab, len(self.__vocab))
        self.__char_indices = dict((c, i) for i, c in enumerate(self.__vocab))
        self.__indices_char = dict((i, c) for i, c in enumerate(self.__vocab))

    def build_model(self):
        print('Build model...')
        model = Sequential()
        model.add(Bidirectional(CuDNNGRU(512, input_shape=(self.__max_len, len(self.__vocab)), return_sequences=True)))
        model.add(Dropout(0.25))
        model.add(Bidirectional(CuDNNGRU(512, input_shape=(self.__max_len, len(self.__vocab)), return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(CuDNNGRU(256, input_shape=(self.__max_len, len(self.__vocab)), return_sequences=False)))
        model.add((Dense(len(self.__vocab), activation='relu')))
        model.add((Dense(len(self.__vocab), activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def get_callbacks(self):
        checkpoint = ModelCheckpoint(self.__file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        generate_sample = LambdaCallback(on_epoch_end=self.on_epoch_end)
        return [checkpoint, generate_sample]

    def reshape_data(self, text):
        # cut the text in semi-redundant sequences of maxlen characters
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.__max_len, self.__step):
            sentences.append(text[i: i + self.__max_len])
            next_chars.append(text[i + self.__max_len])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        X = np.zeros((len(sentences), self.__max_len, len(self.__vocab)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.__vocab)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.__char_indices[char]] = 1
            y[i, self.__char_indices[next_chars[i]]] = 1

        return X, y

    def generate_sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def read_data(self):
        with codecs.open(self.__data_path, "r", "UTF-8") as f:
            text = f.read()
            firstLines = [k[0] if k[0] else k[1] for k in [e.split('\n') for e in text.split('***')]]
            return firstLines, text

    def clean_data(self, text):
        text = text.replace('~', '').replace('â', '').replace('***', '').replace('1', '').replace('2',
                                                                                                  '').replace(
            '3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace(
            '9',
            '').replace(
            '0', '').replace('-', '').replace('\x91', '').replace('\x92', '').replace('\x93', '').replace('*',
                                                                                                          '').replace(
            '\x94', '').replace('(', '').replace(')', '').replace('_', '').replace('&', '').replace('^',
                                                                                                    '').replace(
            '/', '').replace("'", "").replace(';', ',').replace('\r\n', '\n').replace('w', '').replace('x', '').replace(
            ':', '').replace(';', '').replace('\t', ' ')
        text = text.translate(self.__lower_map).lower()

        return text


if __name__ == '__main__':
    mdl = Kitsch(data_path='../Data/kucukiskender.txt', max_len=42 * 3, step=1)
    mdl.main()
