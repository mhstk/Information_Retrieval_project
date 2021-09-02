from main import PersianIR, Persian_stemmer
from vector import Vector
from heap import Max_Heap
import pickle

CLUSTER_PATH = '.\\clustered_docs'
CLUSTER_STOPWPRDS_NAME = '.\\data\\clustered_stopwrods.txt'
CLUSTER_OBJ_NAME = '.\\data\\cluster_ir.obj'


K = 10


class Cluster:
    def __init__(self, name):
        self.name = name
        self.docId_dir = {}
        self.doc_number = 0
        self.vectors = {}
        self.center_vector = None

    def add_doc(self, doc_addr):
        self.doc_number += 1
        self.docId_dir[str(doc_addr)] = self.doc_number

    def update_center(self, doc_id):
        if self.center_vector is None:
            self.center_vector = self.vectors[doc_id]
        else:
            self.center_vector.add_vector(self.vectors[doc_id])

        


class Cluster_ir:
    def __init__(self):
        self.clusters = []
        self.pr = None
        self.doc_address = []
        self.doc_vectors = []


    def get_address(self, doc_id):
        return list(self.pr.doc_list.keys())[list(self.pr.doc_list.values()).index(doc_id)]


    def find_clustered_docs(self, path):
        import os
        for dirr in os.listdir(path): 
            nc = Cluster(str(dirr))
            for subdirr, _, files in os.walk(os.path.join(path, dirr)):
                for filee in files:
                    file_address = str(os.path.join(subdirr, filee))
                    nc.add_doc(file_address)
                    self.doc_address.append(file_address)
            self.clusters.append(nc)

    def run(self, doc_path, stopwords_path, cl_obj_path):
        self.find_clustered_docs(doc_path)
        print('creating persian ir...')
        self.initial_pr(stopwords_path)
        print('calculating doc vectors...')
        self.cal_vector_docs()


    def initial_pr(self, stopwords_path):
        self.pr = PersianIR()
        # self.pr.find_stopwords(self.doc_address, stopwords_path)
        self.pr.read_stopwords(stopwords_path)
        self.pr.create_positional_index(self.doc_address)
        
    
    def cal_vector_docs(self):
        for cluster in self.clusters:
            print(f'processing {cluster.name}...')
            for addr in cluster.docId_dir.keys():
                doc_id = self.pr.doc_list[addr]
                doc_vec = self.pr.doc_to_vector(doc_id)
                cluster.docId_dir[addr] = doc_id
                cluster.vectors[doc_id] = doc_vec
                cluster.update_center(doc_id)
            cluster.center_vector = cluster.center_vector.div_by_num(cluster.doc_number)


    
    def select_cluster(self, query):
        query_vec = self.pr.query_to_vector(self.pr.query_terms(query))
        cluster_score = []
        for i, cluster in enumerate(self.clusters):
            sim = Vector.similaroty_cos(query_vec, cluster.center_vector)
            cluster_score.append((i, sim))
        # print(query_vec)

        print([(self.clusters[x].name, y) for x,y in cluster_score])
        
        selected = max(cluster_score, key=lambda x: x[1])
        return self.clusters[selected[0]], query_vec



    def search_query_rocchio(self, query , k):

        print('\nRocchio result:')
        cluster, query_vec = self.select_cluster(query)
        print(cluster.name)
        similarity = Max_Heap()
        for doc_id, vector in cluster.vectors.items():
            sim = Vector.similaroty_cos(query_vec, vector)
            similarity.push((doc_id, sim))

        ans = []
        for _ in range(k):
            res = similarity.pop()
            if res:
                ans.append(res)
        
        
        for x in ans:
            print( x[0], ' ' , self.get_address(x[0]), '\t', x[1])


        


    
    def search_query_knn(self, query, k):

        query_vec = self.pr.query_to_vector(self.pr.query_terms(query))
        similarity = Max_Heap()
        for i, cluster in enumerate(self.clusters):
            for doc_id, vector in cluster.vectors.items():
                sim = Vector.similaroty_cos(query_vec, vector)
                # if doc_id == 215 or doc_id == 108 or doc_id == 155:
                #     print(doc_id, self.get_address(doc_id))
                #     print(vector.get_size())
                #     print(query_vec.get_size())
                #     print(Vector.multi_2_vec(vector, query_vec))
                similarity.push((i, sim, doc_id))

        ans = []
        categoris = {i: 0 for i in range(len(self.clusters))}
        for _ in range(k):
            res = similarity.pop()
            if res:
                categoris[res[0]] += 1
                ans.append(res)

        print('\nKNN result:')
        category, _ = max(list(categoris.items()), key= lambda x: x[1])
        print(self.clusters[category].name)
        print()
        
        
        for x in ans:
            print(self.clusters[x[0]].name , '\t', x[2], ' ' , self.get_address(x[2]), '\t', x[1])

        
        


        


def save_cluster_ir_obj(obj,  addr):
    with open(addr , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_cluster_ir_obj(addr):
    with open(addr , 'rb') as f:
        obj = pickle.load(f)
    return obj


#######################
#to calculate doc's vector and cluster's center
#######################
# import sys
# sys.setrecursionlimit(10000)
# c = Cluster_ir()
# c.run(CLUSTER_PATH, CLUSTER_STOPWPRDS_NAME , CLUSTER_OBJ_NAME)
# print('saving obj...')
# save_cluster_ir_obj(c , CLUSTER_OBJ_NAME)
# exit()



# load calculated cluster_ir object to save calculating time
c = load_cluster_ir_obj(CLUSTER_OBJ_NAME)



inp = input('> ')
while inp != 'e':
    c.search_query_rocchio(inp, K)
    c.search_query_knn(inp, K)
    inp = input('> ')
