from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
import io
import argparse

model = None
text = ""
char_indices = []
indices_char = []


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_Epoch_End(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    with open("epochs_run.txt", 'w') as f:
        f.write(str(epoch))

    start_index = random.randint(0, len(text) - maxlen - 1)
    print_Result(start_index, epoch)


def print_Result(start_index, epoch = 0):
    for diversity in [0.2, 0.5, 1.0, 1.2, 1.5]: # How random are the letters
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        if epoch >= 20:
            for i in range(400): # length of carol chars

                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
        print()

def setup_data(path):
    global text, char_indices, indices_char
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 15
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return chars, maxlen, x, y

def train_Model(load_best = False, epochs = 0):
    global model
    print('Build model...')

    if not load_best:
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars), activation='softmax'))
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        model = load_model('best_model.h5')

    print_callback = LambdaCallback(on_epoch_end=on_Epoch_End)

    earlystop_callback = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

    checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)

    history = model.fit(x, y,
              batch_size=128,
              epochs=100,
              initial_epoch=epochs,
              callbacks=[print_callback, earlystop_callback, checkpoint_callback])

    model.save("model.h5")

    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('model acc and loss')
    plt.ylabel('acc and loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'acc'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "-R", "--Run", type=str, help="t = training, c = continue, g = generate", required=True)
    parser.add_argument("-p", "-P", "--Path", type=str, help="path to model", required=False)

    args = parser.parse_args()
    if args.Run != "t" and args.Run != "c" and args.Run != "g":
        print("t = training, c = continue, g = generate")
        exit()
    elif args.Run == "g" and args.Path == None:
        print("this requires a path to a model")
        exit()

    path = 'tensor.txt'
    chars, maxlen, x, y = setup_data(path)

    if args.Run == "t":
        train_Model()
    if args.Run == "c":
        with io.open("epochs_run.txt", encoding='utf-8') as f:
            number_of_epochs = int(f.read())
        train_Model(load_best = True, epochs = number_of_epochs+1)
    if args.Run == "g":
        start_index = random.randint(0, len(text) - maxlen - 1)
        model = load_model('best_model.h5')
        print_Result(start_index, 30)

