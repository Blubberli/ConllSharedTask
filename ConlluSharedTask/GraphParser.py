#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pickle
import codecs
from scipy import linalg

class GraphParser:

    pos_matrix = np.zeros
    morph_matrix = np.zeros
    lemma_matrix =  np.zeros

    pos_to_number = {}
    morph_to_number = {}
    lemma_to_number = {}

    direction_to_given_pos = {}
    direction_from_given_pos = {}
    distance_given_pos = {}

    lemma_given_pos = {}
    morph_given_pos = {}

    root_string = ""

    def __init__(self, pos_matrix_path, morph_matrix_path, lemma_matrix_path,
                 pos_to_number_path, morph_to_number_path, lemma_to_number_path,
                 morph_given_pos_path, lemma_given_pos_path, direction_to_given_pos_path,
                 direction_from_given_pos_path, distance_given_pos_path,
                 root_path, test_set_path, parsed_sentences_path):

        self.pos_matrix = np.load(pos_matrix_path)
        self.morph_matrix = np.load(morph_matrix_path)
        self.lemma_matrix = np.load(lemma_matrix_path)

        self.pos_to_number = pickle.load(open(pos_to_number_path, "rb"))
        self.morph_to_number = pickle.load(open(morph_to_number_path, "rb"))
        self.lemma_to_number = pickle.load(open(lemma_to_number_path, "rb"))

        self.direction_to_given_pos = pickle.load(open(direction_to_given_pos_path, "rb"))
        self.direction_from_given_pos = pickle.load(open(direction_from_given_pos_path, "rb"))
        self.distance_given_pos = pickle.load(open(distance_given_pos_path, "rb"))

        self.morph_given_pos = pickle.load(open(morph_given_pos_path, "rb"))
        self.lemma_given_pos = pickle.load(open(lemma_given_pos_path, "rb"))

        with open(root_path, 'r') as f:
            self.root_string = f.readline()

        sentences = self.read_test_sents(test_set_path)

        parsed_sentences = []

        for sentence in sentences:

            parsed_sentences.append(self.parse(sentence))


        self.save_parses(parsed_sentences, parsed_sentences_path)


        print "Done."


    def parse(self, sentence):

        sentence_length = len(sentence)

        sentence_matrix = np.zeros((sentence_length, sentence_length))

        #the sentence as a list of lemmas
        sentence_lemmas = []

        for child_index in range(sentence_length):

            child_triple = sentence[child_index]

            child_pos = child_triple[1]
            child_pos_number = self.pos_to_number.get(child_pos)

            child_morph = child_triple[2]
            child_morph_number = self.morph_to_number.get(child_morph)

            child_lemma = child_triple[0]
            child_lemma_number = self.lemma_to_number.get(child_lemma)

            sentence_lemmas.append(child_lemma)

            for parent_index in range(sentence_length):

                parent_triple = sentence[parent_index]

                parent_pos = parent_triple[1]
                parent_pos_number = self.pos_to_number.get(parent_pos)

                parent_morph = parent_triple[2]
                parent_morph_number = self.morph_to_number.get(parent_morph)

                parent_lemma = parent_triple[0]
                parent_lemma_number = self.lemma_to_number.get(parent_lemma)

                pos_value = self.pos_matrix[child_pos_number][parent_pos_number]


                pos_affinity = pos_value

                if child_pos != self.root_string and parent_pos != self.root_string:

                    to_dir = "right"
                    from_dir = "left"

                    if child_index < parent_index:

                        to_dir = "left"
                        from_dir = "right"

                    distance = child_index - parent_index

                    if child_pos in self.distance_given_pos.keys() and\
                                    parent_pos in self.distance_given_pos.get(child_pos) and \
                                    distance in self.distance_given_pos.get(child_pos).get(parent_pos).keys():

                        pos_affinity *=  self.distance_given_pos.get(child_pos).get(parent_pos).get(distance)
                        pos_affinity *= self.direction_from_given_pos.get(from_dir).get(child_pos) *\
                                        self.direction_to_given_pos.get(to_dir).get(parent_pos)

                    else:
                        pos_affinity *= self.direction_from_given_pos.get(from_dir).get(child_pos) *\
                                        self.direction_to_given_pos.get(to_dir).get(parent_pos)


                morph_affinity = 0

                if child_morph != "_" and child_morph_number != None and parent_morph != "_" and parent_morph_number != None:

                    morph_value = self.morph_matrix[child_morph_number][parent_morph_number]

                    morph_affinity = np.tanh(morph_value)

                    #morph_affinity =  self.morph_given_pos.get(child_morph).get(child_pos) *\
                                      #self.morph_given_pos.get(parent_morph).get(parent_pos) * morph_value


                lemma_affinity = 0

                if child_lemma_number != None and parent_lemma_number != None:

                    lemma_value = self.lemma_matrix[child_lemma_number][parent_lemma_number]

                    lemma_affinity = np.tanh(lemma_value) #* self.lemma_given_pos.get(child_lemma).get(child_pos) *\
                            #self.lemma_given_pos.get(parent_lemma).get(parent_pos)

                affinity = pos_affinity #+ (1/6) * lemma_affinity + (1/3) * morph_affinity
                #affinity = lemma_affinity #pos_affinity #+ lemma_affinity + (10/10) * morph_affinity
                #affinity = morph_affinity #+ lemma_affinity + (10/10) * morph_affinity


                sentence_matrix[child_index][parent_index] = affinity


        truncated_matrix = self.reduce_dimensions(sentence_matrix)

        return self.construct_mst(truncated_matrix, sentence_lemmas)

    def reduce_dimensions(self, matrix):

        num_of_columns = matrix.shape[1]
        num_of_rows = matrix.shape[0]

        k = int((3/4) * num_of_rows)

        'Perform Singular Value Decomposition'
        u, s, v = linalg.svd(matrix, full_matrices=False)

        #print "Matrix has been decomposed."

        'Take the top k singular values'
        truncated_s = s[range(k)]

        'adjust the rows and columns of the other two matrices'
        truncated_u = np.delete(u, range(k, num_of_columns), 1)

        truncated_v = np.delete(v, range(k, num_of_rows), 0)

        'Multiplicate them'
        truncated_matrix = np.dot(np.dot(truncated_u, np.diag(truncated_s)), truncated_v)

        #print "Truncated matrix has been created."

        #print truncated_matrix
        #print "\n\n\n"

        return truncated_matrix



    def construct_mst(self, sentence_matrix, sentence):

        root = self.root_string

        #root has only one child
        has_root = False

        vertices = []

        vertices.append(sentence.index(root))

        edges = []

        while len(vertices) != len(sentence):

            max_edge = (-1) * float("inf")
            vertex_tuple = ()

            for parent in vertices:

                if ((parent == sentence.index(root)) and (not has_root)) or parent != sentence.index(root):

                    has_root = True

                    for child in range(len(sentence)):

                        if child not in vertices:

                            if sentence_matrix[child][parent] >= max_edge:

                                max_edge = sentence_matrix[child][parent]
                                vertex_tuple = (child, parent)

            edges.append(vertex_tuple)
            vertices.append(vertex_tuple[0])

        parsed_sentence = []

        for child_index in range(len(sentence)):

            for tuple in edges:

                if tuple[0] == child_index:

                    parsed_sentence.append((sentence[child_index], sentence[tuple[1]]))

        return parsed_sentence

    def read_test_sents(self, test_set_path):

        sentences = []
        sentence = []

        doc = codecs.open(test_set_path, "r", encoding='utf-8')

        for line in doc:

            if line != "\n":

                line_split = line.split("\t")

                if len(line_split) >= 7:

                    if "-" not in line_split[0]:
                        lemma = line_split[2]
                        pos = line_split[3]
                        morph = line_split[5]

                        triple = (lemma, pos, morph)

                        sentence.append(triple)
            else:

                sentence.append((self.root_string, self.root_string, self.root_string))
                sentences.append(sentence)
                sentence = []

        return sentences

    def save_parses(self, parsed_sentences, parsed_sentences_path):

        output_string = ""

        for sentence in parsed_sentences:

            if sentence != None and len(sentence) != 0:

                word_number = 1

                for tuple in sentence:

                    output_string += str(word_number) + "\t" + str(tuple[0].encode('utf-8')) + "\t" + str(tuple[1].encode('utf-8')) + "\n"
                    word_number += 1

                output_string += "\n"

        f = open(parsed_sentences_path,"w")
        f.write(output_string)
        f.close()

gp = GraphParser("/home/tobi/Schreibtisch/Statistical_Parsing/project_no/trained_no_pos_weighted.npy",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/trained_no_morph_weighted.npy",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/trained_no_lemma_weighted.npy",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/Pos_To_Number.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/Morph_To_Number.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/Lemma_To_Number.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/morph_given_pos.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/lemma_given_pos.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/direction_to_given_pos.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/direction_from_given_pos.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/distance_given_pos.p",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/root.txt",
                 "/home/tobi/Schreibtisch/no-ud-test.conllu",
                 "/home/tobi/Schreibtisch/Statistical_Parsing/project_no/parsed_sentences.txt")