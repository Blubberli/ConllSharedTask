import unittest
import numpy as np
from find_arcs.input_to_embeddings import input_to_embeddings as testclass

class input_to_embeddings_test(unittest.TestCase):

    def setUp(self):
        self._wordsinput = [["Das", "ist", "toll", "end1", "end2"], ["Haupthaus", "start2", "Oma", "super", "nett"]]
        self._posinput = [["NOUN", "DET", "PROPN", "DET", "NOUN"], ["DET", "PROPN", "DET", "NOUN", "PROPN"]]
        self._morphinput = [["Definite=Ind|Gender=Masc|Number=Sing", "_", "_", "Definite=Ind|Gender=Masc|Number=Sing", "DEF"],
                      ["_", "_", "Definite=Ind|Gender=Masc|Number=Sing", "_", "Definite=Ind|Gender=Masc|Number=Sing"]]
        self._labelsinput = [1,0]
        self._rangeinput = [2.3, 4.1]
        embedding_file = '/home/neele/Dokumente/DeepLearning/wikipedia-100-mincount-30-window-8-cbow.bin'
        self._embeddings = testclass.read_word_embeddings(embedding_file)
        self._unknownembedding = testclass.create_unknownembeddings(100)


    def test_read_embeddings(self):
        np.testing.assert_equal(len(self._embeddings.vocab)>1, True)

    def test_label_matrix(self):
        lablematrixshape = testclass.get_oneDim_matrix(self._labelsinput).shape
        np.testing.assert_equal(lablematrixshape, (2,))

    def test_range_input_matrix(self):
        range_input_matrixshape = testclass.get_oneDim_matrix(self._rangeinput).shape
        np.testing.assert_equal(range_input_matrixshape, (2,))

    def test_unknowndictionary(self):
        start = self._unknownembedding["start1"]
        np.testing.assert_equal(len(start), 100)

    def test_wordmatrix(self):
        matrix = testclass.create_input(2, 100, self._wordsinput, self._embeddings, self._unknownembedding)
        shape = matrix.shape
        print(shape)
        np.testing.assert_equal(shape, (2,100*5))

if __name__ == "__main__":
    unittest.main()