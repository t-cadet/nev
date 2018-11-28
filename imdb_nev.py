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

import multiprocessing as mp

def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj, methname)(*arg[2:])

def getImdb():
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

    return (input_shape, xVtrain, x_train, x_test, x_val, y_val, yVtrain, y_train, y_test, epochs)

print("Loading data...")
input_shape, xVtrain, x_train, x_test, x_val, y_val, yVtrain, y_train, y_test, epochs = getImdb()

early_stopper_fit = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1, verbose=0, mode='auto')
early_stopper_train = EarlyStopping(monitor='acc', min_delta=0.001, patience=3, verbose=0, mode='auto')


class Indiv:
    nb_layers = range(1, 6 + 1)
    nb_weights = [1, 2, 4, 8, 16, 32, 64]
    batch_size = [32, 64, 128, 256]
    dropout_rate = {'min': 0.1, 'max': 0.6, 'precision': 2}
    activation = ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softplus', 'linear']
    optimizer = ['rmsprop', 'adam', 'adagrad', 'adadelta', 'adamax', 'nadam']

    @staticmethod
    def _gaussianInRange(range, mean=None):
        if mean is None:
            mean = (range['max']+range['min'])/2
        sd = mean/4
        return np.clip(round(np.random.normal(mean, sd), range["precision"]), range["min"], range["max"])

    # Creates a random space or initialize one from a dictionary or JSON string
    def __init__(self, space=None):
        self.space = (json.loads(space) if isinstance(space, str) else space) if space is not None else {
            "nb_weights": np.random.choice(Indiv.nb_weights, np.random.choice(Indiv.nb_layers)).tolist(),
            "batch_size": random.choice(Indiv.batch_size),
            "activation": random.choice(Indiv.activation),
            "optimizer": random.choice(Indiv.optimizer),
            "dropout_rate": Indiv._gaussianInRange(Indiv.dropout_rate)
        }
        self.fitness = None
        self.gen = None

    def _createModel(self):
        model = Sequential()
        model.add(Dense(self.space["nb_weights"][0], activation=self.space["activation"], input_shape=input_shape))
        
        for i in range(1, len(self.space["nb_weights"])):
            model.add(Dense(self.space["nb_weights"][i], activation=self.space["activation"]))
            model.add(Dropout(self.space["dropout_rate"]))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=self.space["optimizer"],
                      metrics=['accuracy'])
        return model

    def computeFitness(self, verbose=1):
        if verbose > 0: self.print(fit=False)
        model = self._createModel()
        self.fitness = max(
            model.fit(
                xVtrain,
                yVtrain,
                batch_size=self.space["batch_size"],
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_val, y_val),
                callbacks=[early_stopper_fit]
            ).history['val_acc']
        )
        K.clear_session()
        if verbose > 0: self.print(archi=False)
        return self

    # Trains the model on all training data
    def train(self, epoch):
        self.print(fit=False)
        self.model = self._createModel()
        self.fitness = self.model.fit(x_train, y_train,
                                      batch_size=self.space["batch_size"],
                                      epochs=epoch,
                                      verbose=1,
                                      callbacks=[early_stopper_train]).history['acc'][-1]
        self.print(archi=False)

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
        self.space["activation"] = self._mutNearby(self.space["activation"], Indiv.activation)
    
    def _mutOptimizer(self):
        self.space["optimizer"] = self._mutNearby(self.space["optimizer"], Indiv.optimizer)

    def _mutBatchSize(self):
        self.space["batch_size"] = self._mutNearby(self.space["batch_size"], Indiv.batch_size)

    def _mutDropoutRate(self):
        self.space["dropout_rate"] = self._gaussianInRange(Indiv.dropout_rate, mean=self.space["dropout_rate"])
    
    def _mutLayers(self):
        weight = self.space["nb_weights"]
        if len(weight)==max(Indiv.nb_layers) or (random.randint(0,1)==0 and len(weight)>min(Indiv.nb_layers)):
            weight.pop(random.randrange(len(weight))) # rm layer
        else:
            weight.insert(random.randrange(len(weight) + 1), random.choice(Indiv.nb_weights))  # add layer

    def _mutWeights(self):
        weight = self.space["nb_weights"]
        i = random.randrange(len(weight))
        weight[i] = self._mutNearby(weight[i], Indiv.nb_weights)

    def mutate(self):
        mutations = [self._mutLayers, self._mutWeights, self._mutActivation, self._mutOptimizer, self._mutBatchSize, self._mutDropoutRate]
        random.choice(mutations)()
        return self

    def print(self, archi=True, fit=True, toStr=False, gen=False, precision=5):
        msg = ""
        if fit and self.fitness is not None: 
            msg = "Fitness: " + str(round(self.fitness, precision))
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
        self.generation = [Indiv(space) for space in individuals] + [Indiv() for _ in range(size - len(individuals))]
        self.nextGen = []
        self.best = []

    def evolve(self, nb_gen=16, selection=0.33, verbose=1):  # implement delta to stop when it doesn't improve
        assert(selection <= 0.33)
        top = int(selection * self.size)

        for i in range(nb_gen):
            self.generation = list(mp.Pool(1).map(worker, ((ind, "computeFitness", verbose) for ind in self.generation))) #compute fitness of different indiv in parallel
            self.generation.sort(key=lambda ind: ind.fitness, reverse=True)
            self.best = list(heapq.merge(self.best, self.generation, key=lambda ind: ind.fitness, reverse=True))
            self.generationSummary(i)
            
            self.nextGen[:2 * top] = [copy.deepcopy(ind).mutate() for ind in (self.best[:top] + random.sample(self.generation, top))]  # keep firsts and others at random
            self.nextGen[2 * top:3 * top] = list(itertools.chain.from_iterable([Indiv.breed(self.best[k], self.best[k + 1]) for k in range(top)]))  # breed firsts
            self.nextGen[3 * top:] = [Indiv() for _ in range(self.size - 3 * top)] # fill with new random individuals

            self.enforceUniqueness()

            self.generation = self.nextGen

    # more efficient: use ordered dictionary with unique id based on space attribute
    def enforceUniqueness(self):
        for ind in self.nextGen:
            while (ind.space in [e.space for e in self.best]+[e.space for e in self.nextGen if e!=ind]):
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
        logging.info(self.printTop(99999999))

SIZE_POP = 8
NB_GEN = 4
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename="log"+str(SIZE_POP)+"_"+str(NB_GEN)+".txt"
    )

def main():
    import time
    # import os
    # import tensorflow as tf
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    start = time.time()
    pop = Pop(SIZE_POP)
    pop.evolve(NB_GEN, verbose=1)
    end = time.time()
    logging.info("time: " + str(end-start))
    pop.printTop(SIZE_POP*NB_GEN)

if __name__ == '__main__':
   main()
