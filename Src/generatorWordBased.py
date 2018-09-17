# -*- coding: utf-8 -*-
import codecs
import locale
import random
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')
lower_map = {
    ord(u'I'): u'ı',
    ord(u'İ'): u'i',
}


def readPoemText():
    with codecs.open('kucukiskender.txt', "r", "UTF-8") as f:
        text = f.read()
        firstLines = [k[0] if k[0] else k[1] for k in [e.split('\r\n') for e in text.split('***')]]
        text = text.replace('\r\n', '\n').replace(';', '').replace(':', '').replace('\t', '').replace('~', '').replace(
            'â', '').replace('***', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5',
                                                                                                                    '').replace(
            '6', '').replace('7', '').replace('8', '').replace('9', '').replace('0', '').replace('-', '').replace(
            '\x91', '').replace('\x92', '').replace('\x93', '').replace('*', '').replace('\x94', '').replace('(',
                                                                                                             '').replace(
            ')', '').replace('_', '').replace('&', '').replace('^', '').replace('/', '').replace("'", "")
        text = text.translate(lower_map).lower()
        words = []
        for s in text.split('\n'):
            words += s.split(' ')
        return firstLines, words


def generate():
    firstLines, text = readPoemText()
    # text=text[:int(len(text)/1000)]
    chars = sorted(list(set(text)))
    print('Total chars: %s' % len(chars))
    print(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(512, input_shape=(maxlen, len(chars))))
    # model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    filepath = "./weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # train the model, output generated text after each iteration
    for iteration in range(1, 600):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        seed = firstLines[random.randint(0, len(firstLines) - 1)] + '\r\n'

        model.fit(X, y, batch_size=128, nb_epoch=1, callbacks=callbacks_list)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            generated += seed
            print('----- Generating with seed: "' + seed + '"\n')
            sys.stdout.write(generated)

            for i in range(600):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


if __name__ == '__main__':
    generate()
