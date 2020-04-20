import os, pickle, json, ast
import pandas as pd
from scipy import spatial
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import date

class TaskSolver:

    W2V_DICT = dict()

    def __init__(self):
        pass

    def solve(self, task_name, **kwargs):

        self.gen_w2v_dict()

        if task_name == 'calculate-cosin-similarity':
            self.task_calculate_cosin_similarity(kwargs.get('word1'), kwargs.get('word2'))
        elif task_name == 'k-nearest-words':
            self.task_k_nearest_words(kwargs.get('k'), kwargs.get('word'))
        elif task_name == 'synonym-antonym-classification':
            self.task_synonym_antonym_classification()
        elif task_name == 'test-cosin-similarity-with-visim-400-dataset':
            self.test_with_visim_400_data_set()

    def task_calculate_cosin_similarity(self, word1, word2, print_to_screen=True):

        sim = 0

        if word1 in self.W2V_DICT and word2 in self.W2V_DICT:

            sim = (2 - spatial.distance.cosine(self.W2V_DICT[word1], self.W2V_DICT[word2])) / 2

            if (print_to_screen): print("Similarity between '{}' and '{}' is: {}".format(word1, word2, sim))
        else:
            if word1 not in self.W2V_DICT and print_to_screen: print("Word '{}' not in vocab".format(word1))
            if word2 not in self.W2V_DICT and print_to_screen: print("Word '{}' not in vocab".format(word2))

        return sim

    def test_with_visim_400_data_set(self):

        visim_400_df = pd.read_csv(
            os.path.abspath('./Word-Similarity/datasets/ViSim-400/Visim-400.txt'), 
            sep="\t")

        rs, sim1_arr, sim2_arr = [], [], []

        for index, row in visim_400_df.iterrows():

            word_1, word_2 = row['Word1'], row['Word2']
            sim_1, sim_2 = row['Sim1'], row['Sim2']

            if word_1 in self.W2V_DICT and word_2 in self.W2V_DICT:

                sim = self.task_calculate_cosin_similarity(word_1, word_2, False)

                rs.append(sim)
                
                sim1_arr.append(sim_1)
                sim2_arr.append(sim_2)

            else:
                if word_1 not in self.W2V_DICT: print("Word '{}' not in vocab".format(word_1))
                if word_2 not in self.W2V_DICT: print("Word '{}' not in vocab".format(word_2))

        print("Pearson correlation coefficient: ", stats.pearsonr(rs, sim1_arr))
        print("Spearman's rank correlation coefficient: ", stats.spearmanr(rs, sim1_arr))

    def task_k_nearest_words(self, k, word):

        k = int(k)

        if word not in self.W2V_DICT: 
            print("Word '{}' not in vocab".format(word))
            return

        sims = []

        for key in self.W2V_DICT:

            if key != word:

                sims.append({
                    'key': key,
                    'sim': self.task_calculate_cosin_similarity(key, word, False)
                })
        
        k_list = sorted(sims, key=lambda k: k['sim'], reverse=True)[0: (k - 1)]

        print("{} most-similar words of '{}' are:".format(k, word))

        for w in k_list:

            print("Word {} with similarity is {}".format(w.get('key'), w.get('sim')))

        return k_list

    def task_synonym_antonym_classification(self):

        self.prepare_data()

        self.train_synonym_antonym_classification()

        self.test_synonym_antonym_classification()

    def test_synonym_antonym_classification(self):

        clf = pickle.load(open('./main/model/svm.model', 'rb'))

        X_test, Y_test = [], []

        for file in [
            './Word-Similarity/datasets/ViCon-400/400_noun_pairs.txt', 
            './Word-Similarity/datasets/ViCon-400/400_verb_pairs.txt',
            './Word-Similarity/datasets/ViCon-400/600_adj_pairs.txt'
            ]:

            f = open(file, 'r', encoding="utf8")

            for index, line in enumerate(f):

                line_arr = line.split()

                if index == 0: continue

                word1, word2, relation = line_arr[0], line_arr[1], line_arr[2]

                if word1 in self.W2V_DICT and word2 in self.W2V_DICT:
                    
                    np_vec1, np_vec2 = np.array(self.W2V_DICT[word1]), np.array(self.W2V_DICT[word2])

                    vec = np.concatenate((
                        np_vec1,
                        np_vec2,
                        np_vec1 + np_vec2, 
                        np_vec1 * np_vec2,
                        np.absolute(np_vec1 - np_vec2),
                        # np.array([self.task_calculate_cosin_similarity(word1, word2, False)])
                    ), axis=0)

                    X_test.append(vec)

                    if relation == 'SYN': Y_test.append(1)
                    elif relation == 'ANT': Y_test.append(-1)

        X_test = preprocessing.scale(X_test, axis=1)
        pred = clf.predict(X_test)

        print("Test date: {}".format(date.today()))
        print("Precision: {}".format(precision_score(Y_test, pred)))
        print("Recall: {}".format(recall_score(Y_test, pred)))
        print("F1: {}".format(f1_score(Y_test, pred)))

        log = """
        Test date: {}
        Precision: {}
        Recall: {}
        F1: {}
        \n
        ----------------------------------------
        """.format(
            date.today(), 
            precision_score(Y_test, pred), 
            recall_score(Y_test, pred), 
            f1_score(Y_test, pred))

        log_f = open('./main/log', 'w+')

        log_f.write(log)

        log_f.close()

    def train_synonym_antonym_classification(self):

        X_train, Y_train = pickle.load(open('./main/dataset/antonym-synonym/antonym-synonym-pairs.bin', 'rb+'))

        clf = LinearSVC(C=1, verbose=1, class_weight='balanced', dual=False)
        
        clf.fit(preprocessing.scale(X_train, axis=1), Y_train)

        pickle.dump(clf, open('./main/model/svm.model', 'wb+'))

        return clf

    def prepare_data(self):

        X, Y = [], []

        for file in [
            './Word-Similarity/antonym-synonym set/Antonym_vietnamese.txt', 
            './Word-Similarity/antonym-synonym set/Synonym_vietnamese.txt'
            ]:
            
            f = open(file, 'r', encoding="utf8")

            for index, line in enumerate(f):

                line_arr = line.split()
                
                if len(line_arr) < 2: continue

                word1, word2 = line_arr[0], line_arr[1]

                if word1 in self.W2V_DICT and word2 in self.W2V_DICT:

                    np_vec1, np_vec2 = np.array(self.W2V_DICT[word1]), np.array(self.W2V_DICT[word2])

                    vec = np.concatenate((
                        np_vec1,
                        np_vec2,
                        np_vec1 + np_vec2, 
                        np_vec1 * np_vec2,
                        np.absolute(np_vec1 - np_vec2),
                        # np.array([self.task_calculate_cosin_similarity(word1, word2, False)])
                    ), axis=0)

                    X.append(vec)

                    if os.path.basename(f.name) == 'Antonym_vietnamese.txt': Y.append(-1)
                    else: Y.append(1)


        X, Y = np.array(X), np.array(Y)

        pickle.dump(
            ( X.astype(np.float64), Y ),
            open('./main/dataset/antonym-synonym/antonym-synonym-pairs.bin', 'wb+')
        )

    def gen_w2v_dict(self):

        with open('./main/dataset/w2v/w2v-dict.json', 'w+') as f:
            
            if f.read(1):

                f.seek(0)

                self.W2V_DICT = json.load(f)

        if not self.W2V_DICT:

            with open('./Word-Similarity/word2vec/W2V_150.txt', 'r', encoding="utf8") as f:

                for index, line in enumerate(f):

                    line_arr = line.split()
                    
                    if index > 1:

                        self.W2V_DICT.update({line_arr[0]: np.array(line_arr[1:]).astype(float).tolist()})

            f = open("./main/dataset/w2v/w2v-dict.json","w+")

            f.write(json.dumps(self.W2V_DICT))

            f.close()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Helper")

    parser.add_argument(
        "--task",
        required=True,
        metavar="path",
        help="""
        Task name: 
            0 => Cosin Similarity
            1 => Test Cosine Similarity with Visim-400 dataset
            2 => K Nearest Words
            3 => Synonym Antonym Classification
        """,
    )

    parser.add_argument(
        "--word",
        metavar="path",
        help="Target word used in 'K Nearest Words' task",
    )

    parser.add_argument(
        "--k",
        metavar="path",
        help="Number of 'Nearest Words' used in 'K Nearest Words' task",
    )

    parser.add_argument(
        "--word1",
        metavar="path",
        help="Source word used in 'Cosin Similarity' task",
    )

    parser.add_argument(
        "--word2",
        metavar="path",
        help="Target word used in 'Cosin Similarity' task",
    )

    args = parser.parse_args()

    task = args.task 
    k = args.k 
    word = args.word 
    word1 = args.word1
    word2 = args.word2

    switcher = {
        '0': 'calculate-cosin-similarity',
        '1': 'test-cosin-similarity-with-visim-400-dataset',
        '2': 'k-nearest-words',
        '3': 'synonym-antonym-classification'
    }

    task_name = switcher.get(task, "Invalid task")

    print (
        """Argument:
        Task => {}
        K => {}
        Word => {}
        Word1 => {}
        Word2 => {}
        """
        .format(task_name, k, word, word1, word2))

    task_solver = TaskSolver()

    task_solver.solve(
        task_name, 
        k=k,
        word=word,
        word1=word1,
        word2=word2
    )
