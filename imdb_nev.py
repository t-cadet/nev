from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, Callback
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

import logging
import json

import random
import numpy as np
import heapq
import copy
import itertools

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='log.txt'
    )

def getImdb():
    batch_size = 256
    epochs = 6
    num_words = 2048
    maxlen = 256
    val_split = 8192
    input_shape = (num_words,)

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    x_train = vectorize_sequences(x_train, num_words)
    x_val = x_train[:val_split]
    xVtrain = x_train[val_split:val_split * 3]
    x_test = vectorize_sequences(x_test, num_words)

    y_train = np.asarray(y_train).astype('float32')
    y_val = y_train[:val_split]
    yVtrain = y_train[val_split:val_split * 3]
    y_test = np.asarray(y_test).astype('float32')

    return (batch_size, input_shape, xVtrain, x_train, x_test, x_val, y_val, yVtrain, y_train, y_test, epochs)


batch_size, input_shape, xVtrain, x_train, x_test, x_val, y_val, yVtrain, y_train, y_test, epochs = getImdb()

early_stopper_fit = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1, verbose=0, mode='auto')
early_stopper_train = EarlyStopping(monitor='acc', min_delta=0.001, patience=3, verbose=0, mode='auto')


class Individual:
    nb_layers = range(1, 8 + 1)
    nb_weights = [2**n for n in range(0, 6 + 1)]
    activation = ['relu', 'elu', 'tanh', 'sigmoid',
                  'hard_sigmoid', 'softplus', 'linear']
    optimizer = ['rmsprop', 'adam', 'adagrad', 'adadelta', 'adamax', 'nadam']

    # Creates a random space or initialize one from a dictionary or JSON string
    def __init__(self, space=None):
        self.space = (json.loads(space) if isinstance(space, str) else space) if space is not None else {
            "nb_weights": np.random.choice(Individual.nb_weights, np.random.choice(Individual.nb_layers)).tolist(),
            "activation": np.random.choice(Individual.activation),
            "optimizer": np.random.choice(Individual.optimizer)
        }
        self.fitness = None
        self.gen = None

    def _createModel(self, dropout=0.2):
        model = Sequential()
        model.add(Dense(
            self.space["nb_weights"][0], activation=self.space["activation"], input_shape=input_shape))
        for i in range(1, len(self.space["nb_weights"])):
            model.add(
                Dense(self.space["nb_weights"][i], activation=self.space["activation"]))
            if dropout is not None:
                model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=self.space["optimizer"],
                      metrics=['accuracy'])
        return model

    def computeFitness(self, dropout=0.2, verbose=1):
        if verbose > 0: self.print(fit=False)
        model = self._createModel(dropout)
        self.fitness = max(
            model.fit(
                xVtrain,
                yVtrain,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_val, y_val),
                callbacks=[early_stopper_fit]
            ).history['val_acc']
        )
        K.clear_session()
        if verbose > 0: self.print(archi=False)

    # Trains the model on all training data
    def train(self, epoch, dropout=0.2):
        self.print(fit=False)
        self.model = self._createModel(dropout)
        self.fitness = self.model.fit(x_train, y_train,
                                      batch_size=batch_size,
                                      epochs=epoch,
                                      verbose=1,
                                      callbacks=[early_stopper_train]).history['acc'][-1]
        logging.info("Fitness: " + str(self.fitness))

    # Tests the model
    def test(self):
        self.print()
        score = self.model.evaluate(x_test, y_test)
        logging.info("Test score: " + str(score))

    @staticmethod
    def breed(father, mother):
        keys = list(father.space.keys())
        random.shuffle(keys)
        cut = random.randint(1, len(keys) - 1)

        child1 = copy.deepcopy(father)
        child2 = copy.deepcopy(father)
        for k in keys[:cut]: child1.space[k] = mother.space[k] 
        for k in keys[cut:]: child2.space[k] = mother.space[k]
        return child1, child2

    def _mutNearby(self, param, axis):
        return axis[(random.choice([-1, 1]) + axis.index(param)) % len(axis)]

    def _mutActivation(self):
        self.space["activation"] = self._mutNearby(self.space["activation"], Individual.activation)
    
    def _mutOptimizer(self):
        self.space["optimizer"] = self._mutNearby(self.space["optimizer"], Individual.optimizer)
    
    def _mutLayers(self):
        weight = self.space["nb_weights"]
        if random.randint(0,1)==0 and len(weight)>1:
            weight.pop(random.randrange(len(weight))) # rm layer
        else:
            weight.insert(random.randrange(len(weight) + 1), random.choice(Individual.nb_weights))  # add layer

    def _mutWeights(self):
        weight = self.space["nb_weights"]
        print(weight)
        i = random.randrange(len(weight))
        weight[i] = self._mutNearby(weight[i], Individual.nb_weights)

    def mutate(self):
        mutations = [self._mutLayers, self._mutWeights, self._mutActivation, self._mutOptimizer]
        random.choice(mutations)()
        return self

    def print(self, archi=True, fit=True, toStr=False, gen=False):
        msg = ""
        if fit: 
            msg = "Fitness: " + str(self.fitness)
            if gen or archi: msg += " | "
        if gen:
            msg += "Gen: " + str(self.gen)
            if archi: msg += " | "
        if archi: msg += "Architecture: " + json.dumps(self.space)
        if toStr: return msg
        logging.info(msg)


class Pop:

    def __init__(self, size, individuals=[]):
        assert(size >= len(individuals))
        self.size = size
        self.generation = [Individual(space) for space in individuals] + [Individual() for _ in range(size - len(individuals))]
        self.nextGen = []
        self.best = []

    def evolve(self, nb_gen=16, selection=0.25, verbose=1):  # implement delta to stop when it doesn't improve
        assert(selection <= 0.33)
        top = int(selection * self.size)

        for i in range(nb_gen):
            for ind in self.generation: ind.computeFitness(verbose=verbose)
            self.generation.sort(key=lambda ind: ind.fitness, reverse=True)
            self.best = list(heapq.merge(self.best, self.generation, key=lambda ind: ind.fitness, reverse=True))
            self.generationSummary(i)
            
            self.nextGen[:2 * top] = [copy.deepcopy(ind).mutate() for ind in (self.generation[:top] + random.sample(self.generation[top:], top))]  # keep firsts and others at random
            self.nextGen[2 * top:3 * top] = list(itertools.chain.from_iterable([Individual.breed(self.generation[k], self.generation[k + 1]) for k in range(top)]))  # breed firsts
            self.nextGen[3 * top:] = [Individual() for _ in range(self.size - 3 * top)] # fill with new random individuals

            self.enforceUniqueness()

            self.generation = self.nextGen

    # use ordered dictionary with unique id based on space attribute
    def enforceUniqueness(self):
        for ind in self.nextGen:
            while (ind.space in [e.space for e in self.best]):
                ind.mutate()

    def printTop(self, n=None):
        if n is None: n = min(len(self.generation), len(self.best))
        else: n = min(n, len(self.best))
        logging.info("Top " + str(n) + ":")
        for i in range(n):
            self.best[i].print(gen=True)

    def generationSummary(self, i):
        avg = sum(ind.fitness for ind in self.generation)/len(self.generation)
        for ind in self.generation:
            ind.gen = i
        logging.info("----- Generation " + str(i) + " -----")
        logging.info("Average fitness: " + str(avg))
        logging.info("Best: " + self.best[0].print(toStr=True))

def main():
    pop = Pop(8)
    pop.evolve(4)
    pop.printTop()

#if __name__ == '__main__':
 #   main()