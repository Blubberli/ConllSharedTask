#!/usr/bin/python2.7

from __future__ import division
import numpy as np
from scipy import linalg
import pickle
import math
import codecs
import sys


class TrainDependencies:

    pos_matrix = np.zeros((1, 1))
    morph_matrix = np.zeros((1,1))
    lemma_matrix = np.zeros((1,1))

    lemma_to_number = {}
    pos_to_number = {}
    morph_to_number = {}

    number_to_lemma = {}
    number_to_pos = {}
    number_to_morph = {}

    direction_to_given_pos = {}
    direction_from_given_pos = {}
    distance_given_pos = {}

    pos_prob = {}

    root_String = "root"

    def __init__(self, path):

        morph_given_pos = {}
        lemma_given_pos = {}

        num_of_Pos = 0

        current_lemma_number = 0
        current_pos_number = 0
        current_morph_number = 0

        #number of sentences = number of "root"-tags
        sentence_count = 0

        doc = codecs.open(path, "r", encoding='utf-8')

        for line in doc:

            if line != "\n":

                line_split = line.split("\t")

                if len(line_split) >= 7:

                    lemma = line_split[2]
                    pos = line_split[3]
                    morph = line_split[5]

                    if "-" not in line_split[0] and pos != "_" and pos not in self.pos_to_number.keys():

                        self.pos_to_number.update({pos : current_pos_number})

                        self.number_to_pos.update({current_pos_number : pos})

                        current_pos_number += 1

                    if pos != "_":

                        num_of_Pos += 1

                        if pos not  in self.pos_prob.keys():

                            self.pos_prob.update({pos : 1})

                        else:

                            val = self.pos_prob.get(pos)
                            val += 1
                            self.pos_prob.update({pos : val})


                    if "-" not in line_split[0] and morph != "_" and morph not in self.morph_to_number.keys():

                        self.morph_to_number.update({morph : current_morph_number})

                        self.number_to_morph.update({current_morph_number : morph})

                        current_morph_number += 1

                    if morph != "_":

                        if morph not in morph_given_pos.keys():

                            morph_given_pos.update({morph : {}})

                        if pos != "_":

                            tempDic = morph_given_pos.get(morph)

                            if pos not in tempDic.keys():

                                tempDic.update({pos : 1})

                            else:

                                val = tempDic.get(pos)
                                val += 1
                                tempDic.update({pos : val})


                    if "-" not in line_split[0] and lemma != "_" and lemma not in self.lemma_to_number.keys():

                        self.lemma_to_number.update({lemma : current_lemma_number})

                        self.number_to_lemma.update({current_lemma_number : lemma})

                        current_lemma_number += 1

                    if lemma != "_":

                        if lemma not in lemma_given_pos.keys():

                            lemma_given_pos.update({lemma : {}})

                        if pos != "_":

                            tempDic = lemma_given_pos.get(lemma)

                            if pos not in tempDic.keys():

                                tempDic.update({pos : 1})

                            else:

                                val = tempDic.get(pos)
                                val += 1
                                tempDic.update({pos : val})

            else:
                sentence_count += 1

        print "Document has been read"

        #find a distinct name for root
        while self.root_String in self.lemma_to_number.keys() or self.root_String in self.morph_to_number.keys() or self.root_String in self.pos_to_number.keys():

            self.root_String += self.root_String

        #save the name for the root
        f = open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/root.txt","w")
        f.write(self.root_String)
        f.close()

        #root given root is equal to one
        morph_given_pos.update({self.root_String : {self.root_String : 1.0}})

        for morph in morph_given_pos.keys():

            tempDic = morph_given_pos.get(morph)

            for pos in self.pos_prob.keys():

                if pos in tempDic:

                    val = tempDic.get(pos)

                    tempDic.update({pos : (val / self.pos_prob.get(pos))})
                else:

                    tempDic.update({pos : 0.0})

        #root given root is equal to one
        lemma_given_pos.update({self.root_String : {self.root_String : 1.0}})

        for lemma in lemma_given_pos:

            tempDic = lemma_given_pos.get(lemma)

            for pos in self.pos_prob:

                if pos in tempDic:

                    val = tempDic.get(pos)

                    tempDic.update({pos : (val / self.pos_prob.get(pos))})
                else:

                    tempDic.update({pos : 0.0})


        print "Trained Probabilities"

        pickle.dump(lemma_given_pos, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/lemma_given_pos.p", "wb"))
        pickle.dump(morph_given_pos, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/morph_given_pos.p", "wb"))

        self.pos_to_number.update({self.root_String : current_pos_number})
        self.number_to_pos.update({current_pos_number : self.root_String})


        self.morph_to_number.update({self.root_String : current_morph_number})
        self.number_to_morph.update({current_morph_number : self.root_String})


        self.lemma_to_number.update({self.root_String : current_lemma_number})
        self.number_to_lemma.update({current_lemma_number : self.root_String})


        pickle.dump(self.lemma_to_number, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/Lemma_To_Number.p", "wb"))
        pickle.dump(self.number_to_lemma, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/Number_To_Lemma.p", "wb"))

        pickle.dump(self.pos_to_number, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/Pos_To_Number.p", "wb"))
        pickle.dump(self.number_to_pos, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/Number_To_Pos.p", "wb"))

        pickle.dump(self.morph_to_number, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/Morph_To_Number.p", "wb"))
        pickle.dump(self.number_to_morph, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/Number_To_Morph.p", "wb"))

        print "Saved Dictionaries"

        current_pos_number +=1
        current_morph_number +=1
        current_lemma_number +=1

        self.pos_matrix = np.zeros((current_pos_number, current_pos_number))
        self.morph_matrix = np.zeros((current_morph_number, current_morph_number))
        self.lemma_matrix = np.zeros((current_lemma_number, current_lemma_number))

        print "Initialized Matrices"

        self.train(path)

        print "Trained Dependencies"

         #the number of 'root'-tags is equal to the number of sentences
        self.pos_prob.update({self.root_String : sentence_count})

        for pos in self.pos_prob.keys():

            val = self.pos_prob.get(pos)

            self.pos_prob.update({pos : (val / num_of_Pos)})

        pickle.dump(self.pos_prob, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/pos_prob.p", "wb"))
        pickle.dump(self.direction_to_given_pos, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/direction_to_given_pos.p", "wb"))
        pickle.dump(self.direction_from_given_pos, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/direction_from_given_pos.p", "wb"))
        pickle.dump(self.distance_given_pos, open("/home/tobi/Schreibtisch/Statistical_Parsing/project_de/distance_given_pos.p", "wb"))


        pos_matrix_weighted = self.ppmi(self.pos_matrix)
        morph_matrix_weighted = self.ppmi(self.morph_matrix)
        lemma_matrix_weighted = self.ppmi(self.lemma_matrix)

        print "Weighted Matrices"

        truncated_pos_matrix_weighted = self.reduce_dimensions(pos_matrix_weighted)
        truncated_morph_matrix_weighted = self.reduce_dimensions(morph_matrix_weighted)

        self.store(truncated_pos_matrix_weighted, "trained_de_pos_weighted", "/home/tobi/Schreibtisch/Statistical_Parsing/project_de/")
        self.store(truncated_morph_matrix_weighted, "trained_de_morph_weighted", "/home/tobi/Schreibtisch/Statistical_Parsing/project_de/")
        self.store(lemma_matrix_weighted, "trained_de_lemma_weighted", "/home/tobi/Schreibtisch/Statistical_Parsing/project_de/")

        print "Saved Matrices"

    def train(self, path):

        pos_sequence = []
        morph_sequence = []
        lemma_sequence = []

        arcs = []


        pos_pos_count = {}

        doc = codecs.open(path, "r", encoding='utf-8')

        for line in doc:

            if line == "\n":

                for i in range(len(arcs)):

                    child_index = i
                    parent_index = arcs[i]

                    child_pos = pos_sequence[i]
                    parent_pos = self.root_String

                    child_lemma = lemma_sequence[i]
                    parent_lemma = self.root_String

                    child_morph = morph_sequence[i]
                    parent_morph = self.root_String

                    if arcs[i] > -1:
                        parent_pos = pos_sequence[arcs[i]]
                        parent_morph = morph_sequence[arcs[i]]
                        parent_lemma = lemma_sequence[arcs[i]]

                    if parent_pos != "_" and child_pos != "_":

                        #train probabilities for direction (from and to) without root
                        if (child_index < parent_index) and parent_pos != self.root_String:

                            dir = "right"

                            if dir in self.direction_from_given_pos.keys():

                                tempDic = self.direction_from_given_pos.get(dir)

                                if child_pos in tempDic.keys():

                                    val = tempDic.get(child_pos)
                                    val += 1

                                    tempDic.update({child_pos : val})

                                else:
                                    tempDic.update({child_pos : 1})

                            else:
                                self.direction_from_given_pos.update({dir : {child_pos : 1}})

                            dir = "left"

                            if dir in self.direction_to_given_pos.keys():

                                tempDic = self.direction_to_given_pos.get(dir)

                                if parent_pos in tempDic.keys():

                                    val = tempDic.get(parent_pos)
                                    val += 1

                                    tempDic.update({parent_pos : val})

                                else:
                                    tempDic.update({parent_pos : 1})
                            else:
                                self.direction_to_given_pos.update({dir : {child_pos : 1}})


                        #train probabilities for direction (from and to) without root
                        if (child_index > parent_index) and parent_pos != self.root_String:

                            dir = "left"

                            if dir in self.direction_from_given_pos.keys():

                                tempDic = self.direction_from_given_pos.get(dir)

                                if child_pos in tempDic.keys():

                                    val = tempDic.get(child_pos)
                                    val += 1

                                    tempDic.update({child_pos : val})

                                else:
                                    tempDic.update({child_pos : 1})
                            else:
                                self.direction_from_given_pos.update({dir : {child_pos : 1}})


                            dir = "right"

                            if dir in self.direction_to_given_pos.keys():

                                tempDic = self.direction_to_given_pos.get(dir)

                                if parent_pos in tempDic.keys():

                                    val = tempDic.get(parent_pos)
                                    val += 1

                                    tempDic.update({parent_pos : val})

                                else:
                                    tempDic.update({parent_pos : 1})

                            else:
                                self.direction_to_given_pos.update({dir : {child_pos : 1}})

                        #train probabilities for distance, without root
                        if parent_pos != self.root_String:

                            #count how often two pos occur together
                            if child_pos not in pos_pos_count.keys():

                                tempDic = {}
                                tempDic.update({parent_pos : 1})
                                pos_pos_count.update({child_pos : tempDic})

                            elif parent_pos not in pos_pos_count.get(child_pos):

                                tempDic = pos_pos_count.get(child_pos)
                                tempDic.update({parent_pos : 1})

                            else:

                                val = pos_pos_count.get(child_pos).get(parent_pos)
                                val += 1
                                pos_pos_count.get(child_pos).update({parent_pos : val})

                            distance = child_index - parent_index

                            if child_pos not in self.distance_given_pos.keys():
                                self.distance_given_pos.update({child_pos : {parent_pos : {distance : 1}}})

                            elif parent_pos not in self.distance_given_pos.get(child_pos):

                                tempDic = self.distance_given_pos.get(child_pos)
                                tempDic.update({parent_pos : {distance : 1}})

                            elif distance not in self.distance_given_pos.get(child_pos).get(parent_pos).keys():
                                tempDic = self.distance_given_pos.get(child_pos).get(parent_pos)
                                tempDic.update({distance : 1})

                            else:

                                val = self.distance_given_pos.get(child_pos).get(parent_pos).get(distance)
                                val += 1
                                self.distance_given_pos.get(child_pos).get(parent_pos).update({distance : val})


                        parent_pos_number = self.pos_to_number.get(parent_pos)

                        child_pos_number = self.pos_to_number.get(child_pos)

                        self.pos_matrix[child_pos_number][parent_pos_number] += 1

                    if parent_morph != "_" and child_morph != "_":

                        parent_morph_number = self.morph_to_number.get(parent_morph)

                        child_morph_number = self.morph_to_number.get(child_morph)

                        self.morph_matrix[child_morph_number][parent_morph_number] += 1

                    if parent_lemma != "_" and child_lemma != "_":

                        parent_lemma_number = self.lemma_to_number.get(parent_lemma)

                        child_lemma_number = self.lemma_to_number.get(child_lemma)

                        self.lemma_matrix[child_lemma_number][parent_lemma_number] += 1

                pos_sequence = []
                morph_sequence = []
                lemma_sequence = []

                arcs = []

            else:

                line_split = line.split("\t")

                if "-" not in line_split[0] and len(line_split) >= 7 and line_split[6] != "_":

                    pos_sequence.append(line_split[3])
                    morph_sequence.append(line_split[5])
                    lemma_sequence.append(line_split[2])

                    arcs.append((int(float(line_split[6]))-1))


        for pos in self.pos_prob.keys():

            right_dir = "right"
            left_dir = "left"

            if pos in self.direction_from_given_pos.get(right_dir):

                val = self.direction_from_given_pos.get(right_dir).get(pos)
                self.direction_from_given_pos.get(right_dir).update({pos : (val / self.pos_prob.get(pos))})

            else:
                self.direction_from_given_pos.get(right_dir).update({pos : 0.0})

            if pos in self.direction_to_given_pos.get(right_dir):

                val = self.direction_to_given_pos.get(right_dir).get(pos)
                self.direction_to_given_pos.get(right_dir).update({pos : (val / self.pos_prob.get(pos))})

            else:
                self.direction_to_given_pos.get(right_dir).update({pos : 0.0})


            if pos in self.direction_from_given_pos.get(left_dir):

                val = self.direction_from_given_pos.get(left_dir).get(pos)
                self.direction_from_given_pos.get(left_dir).update({pos : (val / self.pos_prob.get(pos))})

            else:
                self.direction_from_given_pos.get(left_dir).update({pos : 0.0})

            if pos in self.direction_to_given_pos.get(left_dir):

                val = self.direction_to_given_pos.get(left_dir).get(pos)
                self.direction_to_given_pos.get(left_dir).update({pos : (val / self.pos_prob.get(pos))})

            else:
                self.direction_to_given_pos.get(left_dir).update({pos : 0.0})


        for pos1 in self.distance_given_pos.keys():

            for pos2 in self.distance_given_pos.get(pos1).keys():

                co_occurrence = pos_pos_count.get(pos1).get(pos2)

                for distance in self.distance_given_pos.get(pos1).get(pos2).keys():

                    val = self.distance_given_pos.get(pos1).get(pos2).get(distance)

                    self.distance_given_pos.get(pos1).get(pos2).update({distance : (val / co_occurrence)})


    def reduce_dimensions(self, matrix):

        number_of_columns = matrix.shape[1]
        num_of_rows = matrix.shape[0]

        k = int(3/4 * num_of_rows)

        'If k is larger than the maximum number of dimensions'
        if k > min(number_of_columns, num_of_rows):
            print "k too large!"
            sys.exit(0)

        'Perform Singular Value Decomposition'
        u, s, v = linalg.svd(matrix, full_matrices=False)

        print "Matrix has been decomposed."

        'Take the top k singular values'
        truncated_s = s[range(k)]

        'adjust the rows and columns of the other two matrices'
        truncated_u = np.delete(u, range(k, number_of_columns), 1)

        truncated_v = np.delete(v, range(k, num_of_rows), 0)

        'Multiplicate them'
        truncated_matrix = np.dot(np.dot(truncated_u, np.diag(truncated_s)), truncated_v)

        print "Truncated matrix has been created."

        return truncated_matrix

    '''
    Method that weights the matrix elements with PPMI/ PPMI with discount factor
    :arg matrix, numOfRows, numOfColumns, destinationPath
    '''
    def ppmi(self, matrix):

        numOfRows = matrix.shape[0]
        numOfColumns = matrix.shape[1]

        updatedPPMIMatrix = np.zeros((numOfRows, numOfColumns))
        print "updatedPPMIMatrix initialized."

        rowVectorSums = self.getRowVectorSums(matrix, numOfRows, numOfColumns)
        columnVectorSums = self.getColumnVectorSum(matrix, numOfRows, numOfColumns)

        totalSum = 0

        for vector in rowVectorSums:

            totalSum += rowVectorSums.get(vector)

        for row in range(numOfRows):

            for column in range(numOfColumns):

                if matrix[row][column] != 0:

                    denominator = ((rowVectorSums.get(row) / totalSum) *
                                   (columnVectorSums.get(column) / totalSum))

                    numerator = matrix[row][column] / totalSum

                    newEntry = math.log(numerator / denominator)

                    if newEntry <= 0:
                        newEntry = 0

                    delta = ((matrix[row][column] / (matrix[row][column] + 1)) *
                             ((min(columnVectorSums.get(column), rowVectorSums.get(row))) /
                              (min(columnVectorSums.get(column), rowVectorSums.get(row)) + 1)))
                    updatedPPMIMatrix[row][column] = delta * newEntry
                    '''
                    if updatedPPMIMatrix[row][column] > 2:
                        updatedPPMIMatrix[row][column] = 1
                    else:
                        updatedPPMIMatrix[row][column] = 0
                    '''
                else:
                    updatedPPMIMatrix[row][column] = 0
        print "Calculated weighted Matrix"

        return updatedPPMIMatrix


    '''
    Method that calculates and stores the sums of all row vectors
    :arg matrix, numOfRows, numOfColumns
    :returns dictionary
    '''
    def getRowVectorSums(self, matrix, numOfRows, numOfColumns):

        rowVectorSums = {}

        for row in range(numOfRows):

            rowVectorSum = 0

            for column in range(numOfColumns):

                rowVectorSum += matrix[row][column]

            rowVectorSums.update({row: rowVectorSum})

        return rowVectorSums

    '''
    Method that calculates and stores the sums of all column vectors
    :arg matrix, numOfRows, numOfColumns
    :returns dictionary
    '''
    def getColumnVectorSum(self, matrix, numOfRows, numOfColumns):

        columnVectorSums = {}

        for column in range(numOfColumns):

            columnVectorSum = 0

            for row in range(numOfRows):

                columnVectorSum += matrix[row][column]

            columnVectorSums.update({column: columnVectorSum})

        return columnVectorSums

    '''
    Method that stores the matrices
    :arg matrix, file, destinationPath
    '''
    def store(self, matrix, file, destinationPath):

        fullPath = destinationPath + file
        np.save(fullPath, matrix)

        print str(file) + " stored."
        print ""

depTrain = TrainDependencies("/home/tobi/Schreibtisch/de-ud-train.conllu")