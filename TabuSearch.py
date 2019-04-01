from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmax
from scipy.spatial.distance import pdist
from datetime import datetime

class TabuSearch:
    """
    Conducts tabu search
    """
    __metaclass__ = ABCMeta
    data = None
    cols = None
    cur_steps = None
    tabu_size = None
    tabu_list = None
    initial_state = None
    current = None
    best = None
    max_steps = None
    max_score = None
    def __init__(self, data, cols, initial_state, tabu_size, max_steps, max_score=None):
        """
        :param initial_state: initial state, should implement __eq__ or __cmp__
        :param tabu_size: number of states to keep in tabu list
        :param max_steps: maximum number of steps to run algorithm for
        :param max_score: score to stop algorithm once reached
        """
        self.data = data
        self.cols = cols
        self.initial_state = initial_state
        if isinstance(tabu_size, int) and tabu_size > 0:
            self.tabu_size = tabu_size
        else:
            raise TypeError('Tabu size must be a positive integer')
        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise TypeError('Maximum steps must be a positive integer')
        if max_score is not None:
            if isinstance(max_score, (int, float)):
                self.max_score = float(max_score)
            else:
                raise TypeError('Maximum score must be a numeric type')
    def __str__(self):
        return ('TABU SEARCH: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST SCORE: %f \n' +
                'BEST MEMBER: %s \n\n') % \
               (self.cur_steps, self._score(self.best), self.best)
    def __repr__(self):
        return self.__str__()
    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm
        :return: None
        """
        self.cur_steps = 0
        self.tabu_list = deque(maxlen=self.tabu_size)
        self.current = self.initial_state
        self.best = self.initial_state
    def _score(self, state):
        score = [self.group_score(member) for member in state]
        return sum(score)
    def _neighborhood(self):
        neighborhood = []
        moves = []
        m = len(self.current)
        for i in range(m):
            for j in range(m):
                if i != j:
                    for k in range(len(self.current[i])):
                        if [i, k] not in self.tabu_list:
                            for b in range(len(self.current[j])):
                                if [j, b] not in self.tabu_list:
                                    k_item = self.current[i][k]
                                    b_item = self.current[j][b]
                                    neighbor = deepcopy(self.current)
                                    neighbor[i][k] = b_item
                                    neighbor[j][b] = k_item
                                    neighborhood.append(neighbor)
                                    moves.append([i, k, j, b])
        return neighborhood, moves

    def group_score(self, member):
        dis = []
        for col in self.cols:
            sample = self.data[col].iloc[member]
            dis.append(pdist(sample.values).mean())
        return sum([item ** 2 for item in dis])

    def _best(self, neighborhood, moves):
        """
        Finds the best member of a neighborhood
        :param neighborhood: a neighborhood
        :return: best member of neighborhood
        """
        score = [self.group_score(member) for member in self.current]
        scores = []
        for item in range(len(neighborhood)):
            movex = moves[item][0]
            movey = moves[item][2]
            score_item = deepcopy(score)
            score_item[movex] = self.group_score(neighborhood[item][movex])
            score_item[movey] = self.group_score(neighborhood[item][movey])
            scores.append(sum(score_item))
        ind = argmax(scores)
        return neighborhood[ind], moves[ind]
    def prepare_results(self):
        self.data['group'] = 0
        for i in range(len(self.best)):
            self.data.iloc[self.best[i], -1] = i + 1
        return self.data
    def run(self, verbose=True):
        """
        Conducts tabu search
        :param verbose: indicates whether or not to print progress regularly
        :return: best state and objective function value of best state
        """
        self._clear()
        print('clear')
        j = 0
        for i in range(self.max_steps):
            self.cur_steps += 1
            print(self)
            neighborhood, moves = self._neighborhood()
            print('start', datetime.now())
            neighborhood_best, move = self._best(neighborhood, moves)
            print('end', datetime.now())
            self.tabu_list.append(move[0:2])
            self.tabu_list.append(move[2:])
            self.current = deepcopy(neighborhood_best)
            if (self._score(neighborhood_best) > self._score(self.best)):
                self.best = deepcopy(self.current)
                j = 0
            else:
                j = j+1
            if j >= 10:
                print("TERMINATING - 10 steps ")
                return self.prepare_results()
        print("TERMINATING - REACHED MAXIMUM STEPS")
        return self.prepare_results()