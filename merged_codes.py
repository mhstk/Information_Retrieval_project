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
                # print(doc_id)
                # print(vector.get_size())
                # print(query_vec.get_size())
                # print(Vector.multi_2_vec(vector, query_vec))
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





class Max_Heap:
    # initializing the constructor with arr (array that we have to convert into heap). The default value is None([])
    def __init__(self, arr=[]):
        # Initializing the heap with no elements in it
        self._heap = []
        
        # If the array by the user is not empty, push all the elements
        if arr is not None:
            for root in arr:
                self.push(root)

# push is used to insert new value to the heap
    def push(self, value):
        
        # Appending the value given by user at the last
        self._heap.append(value)
        # Calling the bottom_up() to ensure heap is in order.
        # here we are passing our heap 
        Max_Heap._bottom_up(self._heap, len(self) - 1)

# push is used to insert new value to the heap
    def pop(self):
        if len(self._heap)!=0:
        # swapping the root value with the last value.

            Max_Heap._swap(self._heap, len(self) - 1, 0)
        # storing the popped value in the root variable

            root = self._heap.pop()

        #Calling the top_down function to ensure that the heap is still in order 
            Max_Heap._top_down(self._heap, 0)
            
        else:
            root=None
        return root


    def __str__(self):
        out = []
        while True:
            root =  self.pop()
            if not root:
                break
            out.append(root)
        for i in out:
            self.push(i)
        return str(out)
        


# It tells the length of the heap
    def __len__(self):
        return len(self._heap)
# print the first element (The root)
    def peek(self):
        if len(self._heap)!=0:
            return(self._heap[0])
        else:
            return("heap is empty")


    # Swaps value in heap between i and j index
    @staticmethod
    def _swap(L, i, j):
        L[i], L[j] = L[j], L[i]

    # This is a private function used for traversing up the tree and ensuring that heap is in order
    @staticmethod
    def _bottom_up(heap, index):
        # Finding the root of the element
        root_index = (index - 1) // 2
        
        # If we are already at the root node return nothing
        if root_index < 0:
            return

        # If the current node is greater than the root node, swap them
        if heap[index][1] > heap[root_index][1]:
            Max_Heap._swap(heap, index,root_index)
        # Again call bottom_up to ensure the heap is in order
            Max_Heap._bottom_up(heap, root_index)

    # This is a private function which ensures heap is in order after root is popped
    @staticmethod
    def _top_down(heap, index):
        child_index = 2 * index + 1
        # If we are at the end of the heap, return nothing
        if child_index >= len(heap):
            return

        # For two children swap with the larger one
        if child_index + 1 < len(heap) and heap[child_index][1] < heap[child_index + 1][1]:
            child_index += 1

        # If the child node is smaller than the current node, swap them
        if heap[child_index][1] > heap[index][1]:
            Max_Heap._swap(heap, child_index, index)
            Max_Heap._top_down(heap, child_index)



class Node:
    def __init__(self):
        self.data = None
        self.order = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def addNode(self, data , order=-1):
        if order == -1:
            order = data
        curr = self.head
        if curr is None:
            n = Node()
            n.data = data
            n.order = order
            self.head = n
            return

        if curr.order > order:
            n = Node()
            n.data = data
            n.order = order
            n.next = curr
            self.head = n
            return

        while curr.next is not None:
            if curr.next.order > order:
                break
            curr = curr.next
        n = Node()
        n.data = data
        n.order = order
        n.next = curr.next
        curr.next = n
        return

    def __str__(self):
        data = []
        curr = self.head
        while curr is not None:
            data.append(curr.data)
            curr = curr.next
        return "[%s]" %(', '.join(str(i) for i in data))


    def to_list(self):
        data = []
        curr = self.head
        while curr is not None:
            data.append(curr.data)
            curr = curr.next
        return data

    def to_list_ith(self, i=0):
        data = []
        curr = self.head
        while curr is not None:
            data.append(curr.data[i])
            curr = curr.next
        return data

    def __repr__(self):
        return self.__str__()



import re
import pickle
import math
from patrica import trie
from linklist import LinkedList
from vector import Vector
from heap import Max_Heap


DOC_PATH = './docs/'
DIC_FILE_NAME = "./data/Dictionary.fa"
DIC_PCKL_NAME = "./data/Dictionary.obj"
VERB_FILE_NAME = "./data/VerbList.fa"
VERB_PCKL_NAME = "./data/VerbList.obj"
STOPWORDS_FILE_NAME = "./data/stopwords.txt"

VERB_BOOL = True
NOUN_BOOL = True

USE_CHAMPION_LISTS = False
CHAMPION_LIST_LENGTH = 10
K = 15


class Persian_stemmer:
    def __init__(self, verb_dic_bool=True, noun_dic_bool=True):
        self.persian_dictionary = trie()
        self.verb_dic_bool = verb_dic_bool
        self.noun_dic_bool = noun_dic_bool
        self.verb_dic = trie()
        if noun_dic_bool:
            self.load_dictionary_pckl()
        if verb_dic_bool:
            self.load_verb_pckl()
        


 
    def load_verb(self):
        words = []
        with open(VERB_FILE_NAME , 'r' , encoding='utf-8') as f:
            words = f.readlines()
            for word in words:
                verbs = word.split('\t')
                self.verb_dic[verbs[0]] = verbs[2]
                self.verb_dic[verbs[1]] = verbs[2]
        return self.verb_dic


    # def create_pers_verb_pckl(self):
    #     obj = self.load_verb()
    #     with open(VERB_PCKL_NAME , 'wb') as f:
    #         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def load_verb_pckl(self):
        with open(VERB_PCKL_NAME , 'rb') as f:
            self.verb_dic = pickle.load(f)
        return self.verb_dic


    def load_dictionary(self):
        words = []
        with open(DIC_FILE_NAME , 'r' , encoding='utf-8') as f:
            words = f.readlines()
        for word in words:
            self.persian_dictionary[word.strip()] = True
        return self.persian_dictionary


    # def create_pers_dic_pckl(self):
    #     obj = self.load_dictionary()
    #     with open(DIC_PCKL_NAME , 'wb') as f:
    #         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def load_dictionary_pckl(self):
        with open(DIC_PCKL_NAME , 'rb') as f:
            self.persian_dictionary = pickle.load(f)
        return self.persian_dictionary
        


    def persian_normalize(self, text):
        trans_src = ''
        trans_to = ''

        trans_src += '1234567890'
        trans_to += '۱۲۳۴۵۶۷۸۹۰'

        trans_src += 'يةۀكؤإأ'
        trans_to += 'یههکواا'

        translation = text.maketrans(trans_src, trans_to)
        text = text.translate(translation)


        text = re.compile('[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652]').sub('', text) #remove Fathan, ...
        return text


    def remove_whitespaces(self, text):
        patterns = [
            (r' +', ' '),  
            (r'\n\n+', '\n\n'), 
            (r'[ـ\r]', '') #keshida and \r
        ]
        text = self.sub_patterns(patterns, text)

        return text


    def punc_normalize(self, text):
        punc = r'-!$%^&*\(\)_+|~=`\{\}\[\]:";\'<>?,./\\><؟\?،؛«»'
        patterns = [
            (r'['+punc+r']', ' ')
        ]
        text = self.sub_patterns(patterns, text)

        return text




    def postix_normalize(self, text):
        patterns = [
            ('\u200c' , ' '),
            ('\u200b' , ''),
            (r'([^ ]ه) ی ', r'\1'+'\u200cی'),  # مثال خانه ی
            (r'(^| )(ن?می) ', r'\1' + r'\2'+'\u200c'),  #می, نمی
            (r'([^\n\d ]{2,}) (تر(ین?)?|گری?|های?)(?=[ \n]|$)', r'\1'+'\u200c'+r'\2'),  #تر, تری, ترین, گر, گری, ها, های
            (r'([^ ]ه) (ا(م|یم|ش|ند|ی|ید|ت))(?=[ \n]|$)', r'\1'+'\u200c'+r'\2'),  #ام, ایم, اش, اند, ای, اید, ات,

        ]
        text = self.sub_patterns(patterns, text)

        return text






    def normalize(self, text):  
        text = self.persian_normalize(text)
        text = self.punc_normalize(text)
        text = self.remove_whitespaces(text)
        text = self.postix_normalize(text)
        return text


    def tokenize(self, text):
        tokens = text.split()
        return tokens


    def stemmer(self, tokens):
        n_tokens = []
        for token in tokens:
            stemmed = self.token_stemmer(token)
            if len(stemmed) >= 1:
                n_tokens.append(stemmed)
        return n_tokens


    def token_stemmer(self, token):
        #if token is a verb
        if self.verb_dic_bool:
            verb_token = token.replace('\u200c' , ' ')
            if verb_token in self.verb_dic:
                if isinstance(self.verb_dic[verb_token], str):
                    return str(self.verb_dic[verb_token])
        
        #patterns used in normalized
        main_patterns = [
            (r'\d*' , ''), #remove numbers
            ('([^ ]ه)\u200cی ', r'\1'),
            ('([^ ]ه)\u200c(ا(م|یم|ش|ند|ی|ید|ت))$' , r'\1'),
            ('[\u200c](تر(ین?)?|گری?|های?)' , ''),
            ('^(ن?می)[\u200c]' , ''),
        ]
        before = token
        token = self.sub_patterns(main_patterns, token)
        #this means removed part was not part of stemm
        if before != token:
            return token

        if self.noun_dic_bool:

            #if the token without any stemming be in persian_dict, the token is stemm
            if token in self.persian_dictionary:
                return token

            #remove signs of plurals
            first_patterns = [
                (r'(?<=[^ او])(ان|ات|ی)$' , ''),
            ]
            token = self.sub_patterns(first_patterns, token)
            #if single of word be in persian_dict, means stemming was right
            if token in self.persian_dictionary:
                return token

            #otherwise stemming might be wrong, and it wasn't plural signs
            token = before
        

        #normal stemming, if the word not found in any previous part, we do stemming for the token with these patterns(in order)
        patterns = [
            (r'(تر(ین?)?|گری?)(?=$)' , ''), #تر ، ترین، گر، گری در آخر کلمه
            (r'^(ن?می|ن|ب)(?=[^ ]{2,}$)' , ''), #ب،ن ، می، نمی اول کلمه
            (r'(?<=[^ او])(م|ت|ش|مان|تان|شان|ی|یم|ید|د|ند)$' , ''), # حذف شناسه های مالکیت و فعل در آخر کلمه
            (r'(ان|ات|ها|های)$' , ''), #ان، ات، ها، های آخر کلمه
            (r'([^ ]ه) (ا(م|یم|ش|ند|ی|ید|ت))$' , r'\1'), #حذف ام،ای،ایم،اید،اش،اند
        ]
        token = self.sub_patterns(patterns, token)
        
        return token

    def sub_patterns(self, patterns_list, text):
        patterns = [(re.compile(x[0]), x[1]) for x in patterns_list]
        for pattern, rep in patterns:
            text = pattern.sub(rep, text)
        return text

    def stemmed_tokens(self, text):
        return self.stemmer(self.tokenize(self.normalize(text)))

    def positioned_tokens(self, tokens):
        token_position = []
        for i, token in enumerate(tokens):
            token_position.append((token , i))
        return token_position


class PersianIR:
    def __init__(self, ps_verb_bool=True, ps_noun_bool=True):
        self.ps = Persian_stemmer(ps_verb_bool, ps_noun_bool)
        self.last_term_id = 1
        self.last_doc_id = 1
        self.term_frequency = {}
        self.dictionary = {}
        self.postings_lists = {}
        self.doc_list = {}
        self.stopwords = []
        self.num_documents = 0
        self.documents_frequency = {}
        self.champion_lists = {}

    def file_text(self, addr):
        with open(addr , 'r', encoding='utf-8') as f:
            data = f.read()
        return data

    def read_a_doc(self, text):
        return self.ps.positioned_tokens(self.ps.stemmed_tokens(text))


    def create_positional_index(self, docs_addr):
        self.num_documents = len(docs_addr)
        for doc_addr in docs_addr:
            doc_id = self.create_get_doc_id(doc_addr)
            text = self.file_text(doc_addr)
            tokens = self.read_a_doc(text)
            

            for token, pos in tokens:
                if token in self.stopwords:
                    continue
                term_id = self.create_get_term_id(token)
                self.increase_term_freq(term_id, doc_id)
                posting_list = self.create_get_posting_list(term_id)
                posting_list.addNode((doc_id , pos) , doc_id)

        return self.postings_lists



    def create_get_posting_list(self, term_id):
        if term_id in self.postings_lists:
            return self.postings_lists[term_id]
        self.postings_lists[term_id] = LinkedList()
        return self.postings_lists[term_id]


    def create_get_doc_id(self, doc_addr):
        if doc_addr in self.doc_list:
            return self.doc_list[doc_addr]
        self.doc_list[doc_addr] = self.last_doc_id
        self.last_doc_id += 1
        return self.doc_list[doc_addr]


    def create_get_term_id(self, term):
        if term in self.dictionary:
            return self.dictionary[term]
        self.dictionary[term] = self.last_term_id
        self.documents_frequency[self.last_term_id] = 0
        self.last_term_id += 1
        return self.dictionary[term]

    def increase_term_freq(self, term_id, doc_id):
        if (term_id, doc_id) in self.term_frequency:
            self.term_frequency[(term_id, doc_id)] += 1
        else: 
            self.term_frequency[(term_id, doc_id)] = 1
            self.documents_frequency[term_id] += 1


    def get_term_freq(self, term_id, doc_id):
        if (term_id, doc_id) in self.term_frequency:
            return self.term_frequency[(term_id, doc_id)]
        return 0


    def find_stopwords(self, docs_addr, stopwords_file_name=STOPWORDS_FILE_NAME, write_to_file=True):
        term_freq = {}
        for doc_addr in docs_addr:
            text = self.file_text(doc_addr)
            tokens = self.read_a_doc(text)
            

            for token, _ in tokens:
                if token in term_freq:
                    term_freq[token] += 1
                else:
                    term_freq[token] = 1
                
        term_freq = {k: v for k, v in sorted(term_freq.items(), key=lambda item: item[1], reverse=True )}
        max_v = max(term_freq.values())
        self.stopwords = [k for k,v in term_freq.items() if v >= max_v*0.2]

        with open(stopwords_file_name , 'w', encoding='utf-8') as fwriter: 
            fwriter.writelines([k+'\n' for k in self.stopwords])
        
        return self.stopwords


    def read_stopwords(self, stopwords_file_name=STOPWORDS_FILE_NAME):
        with open(stopwords_file_name, 'r', encoding='utf-8') as freader:
            for line in freader.readlines():
                self.stopwords.append(line[:-1])


    def posting_list_by_term(self, term):
        if term in self.dictionary:
            term_id = self.dictionary[term]
            return self.postings_lists[term_id].to_list()
        return []

    def posting_list_by_term_docid_only(self, term):
        if term in self.dictionary:
            term_id = self.dictionary[term]
            l = list(set(self.postings_lists[term_id].to_list_ith(0)))
            l.sort()
            return l
        return []

    def query_terms(self, text):
        queris = self.ps.stemmed_tokens(text)
        # print('queris: ' , queris)
        return queris


    def boolean_query(self, text):
        queris = list(set(self.query_terms(text)))
        if len(queris) == 1:
            return self.posting_list_by_term_docid_only(queris[0])
        elif len(queris) > 1:
            score = {}
            ans = []
            for doc_id in self.doc_list.values():
                score[doc_id] = 0
                for term in queris:
                    if doc_id in self.posting_list_by_term_docid_only(term):
                        score[doc_id] += 1
                ans.append(doc_id)

            ans.sort(key=lambda x: score[x], reverse=True)
            return ans



    def tf(self, term_id, doc_id):
        return 0 if self.get_term_freq(term_id, doc_id) == 0 else\
             (1+math.log10(self.get_term_freq(term_id, doc_id)))

    def idf(self, term_id):
        return math.log10(self.num_documents/self.documents_frequency[term_id])

    def tf_idf(self, term_id, doc_id):
        # return self.tf(term_id, doc_id) * self.idf(term_id)
        return self.tf(term_id, doc_id) * self.idf(term_id)


    def get_term_id(self, term):
        if term in self.dictionary:
            return self.dictionary[term]
        return False

    def index_elimination_doc_list(self, query_terms, use_champion_list=False):
        doc_id_list = []
        for query in query_terms:
            if self.get_term_id(query):
                term_id = self.get_term_id(query)
            else:
                continue
            if not use_champion_list:
                doc_id_list += self.posting_list_by_term_docid_only(query)
            else:
                doc_id_list += self.champion_lists[term_id].to_list()
            doc_id_list = list(set(doc_id_list))
        doc_id_list.sort()
        return doc_id_list



    def query_term_frequency(self, q_terms):
        tf_query = {}
        for t in q_terms:
            if t in tf_query:
                tf_query[t] += 1
            else:
                tf_query[t] = 1
        term_tf_query = {}
        for q, qf in tf_query.items():
            if self.get_term_id(q):
                term_tf_query[self.get_term_id(q)] =  ( 1+math.log10(qf) ) 
        return term_tf_query
    

    def query_to_vector(self, q_terms):
        v = Vector()
        # print('\nQuery')
        term_tf_query = self.query_term_frequency(q_terms)
        for term_id in self.dictionary.values():
            if term_id in term_tf_query:
                v.add(term_tf_query[term_id])
                # print(term_id , '     ' , list(self.dictionary.keys())[list(self.dictionary.values()).index(term_id)] ,'     ' , term_tf_query[term_id])
            else:
                v.add(0.0)
        return v
        


    def doc_to_vector(self, doc_id):
        v = Vector()
        # if doc_id == 16 or doc_id == 4:
        #     print('\n\n\nDOC')
        for term_id in self.dictionary.values():
            temp = self.tf_idf(term_id, doc_id)
            # if (doc_id == 16 or doc_id == 4) and temp != 0.0:
            #     print(term_id, list(self.dictionary.keys())[list(self.dictionary.values()).index(term_id)], '    ', temp)
            v.add(temp)
        return v


    
    def vector_query_search_get_best_k(self, text, k, use_champion_list=False):
        query_terms = self.query_terms(text)
        doc_list = self.index_elimination_doc_list(query_terms, use_champion_list=use_champion_list)

        print('len doc_list', len(doc_list))

        query_vec = self.query_to_vector(query_terms)

        similarity = Max_Heap()
        for doc_id in doc_list:
            doc_vec = self.doc_to_vector(doc_id)
            # print(doc_id)
            # print(doc_vec.get_size())
            # print(query_vec.get_size())
            # print(Vector.multi_2_vec(doc_vec, query_vec))
            similarity.push((doc_id, Vector.similaroty_cos(doc_vec, query_vec)))

        ans = []
        for _ in range(k):
            res = similarity.pop()
            if res:
                ans.append(res)

        return ans

    def create_champion_lists(self, k):
        for term_id in self.dictionary.values():
            champion_list = LinkedList()
            temp_heap = Max_Heap()
            added_doc = []
            for doc_id, _ in self.postings_lists[term_id].to_list():
                if doc_id in added_doc:
                    continue
                temp_heap.push((doc_id, self.tf_idf(term_id, doc_id)))
                added_doc.append(doc_id)
                
            for _ in range(k):
                node = temp_heap.pop()
                if node:
                    champion_list.addNode(node[0])
            self.champion_lists[term_id] = champion_list
            
        return



if __name__ == '__main__':


    # #*****************#
    # #phase 1 examples:
    # #*****************#
    # p = PersianIR(VERB_BOOL, NOUN_BOOL)
    # addr_list = [f'{DOC_PATH}{x}.txt' for x in range(1,101)]
    # # p.find_stopwords(addr_list)
    # # exit()
    # p.read_stopwords()
    # p.create_positional_index(addr_list)
    

    # query = input('> ') 
    # while query != 'e':
    #     res = p.boolean_query(query)
    #     print(res)
    #     query = input('> ') 





    #*****************#
    #phase 2 examples:
    #*****************#
    p = PersianIR(VERB_BOOL, NOUN_BOOL)
    addr_list = [f'{DOC_PATH}{x}.txt' for x in range(1,101)]
    # p.find_stopwords(addr_list)
    # exit()
    p.read_stopwords()
    p.create_positional_index(addr_list)
    p.create_champion_lists(k=CHAMPION_LIST_LENGTH)


    query = input('> ') 
    while query != 'e':
        res = p.vector_query_search_get_best_k(query, K, USE_CHAMPION_LISTS)
        for x in res:
            print('doc_id:', x[0], '\tscore:', x[1])
        query = input('> ') 




"""
A PATRICIA trie implementation for efficient matching of string collections on
text.

This class has an (Py2.7+) API nearly equal to dictionaries.

*Deleting* entries is a "half-supported" operation only. The key appears
"removed", but the trie is not actually changed, only the node state is
changed from terminal to non-terminal. I.e., if you frequently delete keys,
the compaction will become fragmented and less efficient. To mitigate this
effect, make a copy of the trie (using a copy constructor idiom)::

    T = trie(**T)

If you are only interested in scanning for the *presence* of keys, but do not
care about mapping a value to each key, using `None` as the value of your
keys and scanning with ``key(S, i, j, None)`` at every offset ``i:j`` in the
string ``S`` is perfectly fine (because the return value will be the key
string iff a full match was made and `None` otherwise). In other words, it
is not necessary to create slices of strings to scan in a window only::

    >>> T = trie(present=None)
    >>> T.key('is absent here', 3, 9, None) # scan in second word [3:9]
    >>> T.key('is present here', 3, 10, None) # scan in second word [3:10]
    'present'

License: Apache License v2 (http://www.apache.org/licenses/LICENSE-2.0.html)
"""

__author__ = 'Florian Leitner <florian.leitner@gmail.com>'
__version__ = '10'


class _NonTerminal():
    pass


__NON_TERMINAL__ = _NonTerminal()

# recursion functions


def _count(node):
    "Count the number of terminal nodes in this branch."
    count = 0 if (node._value is __NON_TERMINAL__) else 1
    for _, child in node._edges.values():
        count += _count(child)
    return count


def _keys(node, accu):
    "Yield keys of terminal nodes in this branch."
    for key, value in _items(node, accu):
        yield key


def _items(node, accu):
    "Yield key, value pairs of terminal nodes in this branch."
    if node._value is not __NON_TERMINAL__:
        yield ''.join(accu), node._value
    for edge, child in node._edges.values():
        accu.append(edge)
        for key, value in _items(child, accu):
            yield key, value
        accu.pop()


def _values(node):
    "Yield values of terminal nodes in this branch."
    if node._value is not __NON_TERMINAL__:
        yield node._value
    for edge, child in node._edges.values():
        for value in _values(child):
            yield value


class trie():
    """
    Usage Example::

      >>> T = trie('root', key='value', king='kong') # a root and two nodes
      >>> T['four'] = None # setting new values as in a dict
      >>> '' in T # check if the value exits (note: the [empty] root is '')
      True
      >>> 'kong' in T # existence checks as in a dict
      False
      >>> T['king'] # get the value for an exact key ... as in a dict!
      'kong'
      >>> T['kong'] # error from non-existing keys (as in a dict...)
      Traceback (most recent call last):
          ...
      KeyError: 'kong'
      >>> len(T) # count keys ("terminals") in the tree
      4
      >>> sorted(T) # plus "traditional stuff": keys(), values(), and items()
      ['', 'four', 'key', 'king']
      >>> # scanning a text S with key(S), value(S), and item(S):
      >>> S = 'keys and kewl stuff'
      >>> T.key(S) # report the (longest) key that is a prefix of S
      'key'
      >>> T.value(S, 9) # using offsets; NB: empty root always matches!
      'root'
      >>> del T[''] # interlude: deleting keys and root is the empty key
      >>> T.item(S, 9) # raise error if no key is a prefix of S
      Traceback (most recent call last):
          ...
      KeyError: 'k'
      >>> # info: the error string above contains the matched path so far
      >>> T.item(S, 9, default=None) # avoid the error by specifying a default
      (None, None)
      >>> # iterate all matching content with keys(S), values(S), and items(S):
      >>> list(T.items(S))
      [('key', 'value')]
      >>> T.isPrefix('k') # reverse lookup: check if S is a prefix of any key
      True
      >>> T.isPrefix('kong')
      False
      >>> sorted(T.iter('k')) # and get all keys that have S as prefix
      ['key', 'king']
    """

    def __init__(self, *value, **branch):
        """
        Create a new tree node.
        Any arguments will be used as the ``value`` of this node.
        If keyword arguments are given, they initialize a whole ``branch``.
        Note that `None` is a valid value for a node.
        """
        self._edges = {}
        self._value = __NON_TERMINAL__
        if len(value):
            if len(value) == 1:
                self._value = value[0]
            else:
                self._value = value
        for key, val in branch.items():
            self[key] = val

    @staticmethod
    def __offsets(strlen, start, end):
        # Return the correct start, end offsets for a string of length `strlen`.
        return (max(0, strlen + start) if start < 0 else start,
                strlen if end is None else end)

    @staticmethod
    def __check(value, match, default):
        if value is not __NON_TERMINAL__:
            return match, value
        elif default is not __NON_TERMINAL__:
            return None, default
        else:
            raise KeyError(match)

    def _find(self, path, start, *end):
        if start < len(path) and path[start] in self._edges:
            edge, child = self._edges[path[start]]
            if path.startswith(edge, start, *end):
                return child, start + len(edge)
        return None, start  # return None

    def _next(self, path, start, *end):
        try:
            edge, child = self._edges[path[start]]
            if path.startswith(edge, start, *end):
                return child, start + len(edge)
        except KeyError:
            pass
        raise KeyError(path)  # raise error

    def _scan(self, rvalFun, string, start=0, *end):
        node = self
        start, _ = trie.__offsets(len(string), start, None)
        while node is not None:
            if node._value is not __NON_TERMINAL__:
                yield rvalFun(string, start, node._value)
            node, start = node._find(string, start, *end)

    def __setitem__(self, key, value):
        node = self
        keylen = len(key)
        idx = 0
        while keylen != idx:
            if key[idx] in node._edges:
                node, idx = node.__followEdge(key, idx)
            else:
                # no common prefix, create a new edge and (leaf) node
                node._edges[key[idx]] = (key[idx:], trie(value))
                break
        else:
            node._value = value

    def __followEdge(self, key, idx):
        edge, child = self._edges[key[idx]]
        if key.startswith(edge, idx):
            # the whole prefix matches; advance
            return child, idx + len(edge)
        else:
            # split edge after the matching part of the key
            pos = 1
            last = min(len(edge), len(key) - idx)
            while pos < last and edge[pos] == key[idx + pos]:
                pos += 1
            split = trie()
            split._edges[edge[pos]] = (edge[pos:], child)
            self._edges[key[idx]] = (edge[:pos], split)
            return split, idx + pos

    def __getitem__(self, key):
        node = self
        keylen = len(key)
        idx = 0
        while keylen != idx:
            node, idx = node._next(key, idx)
        if node._value is __NON_TERMINAL__:
            raise KeyError(key)
        else:
            return node._value

    def __delitem__(self, key):
        node = self
        keylen = len(key)
        idx = 0
        while keylen != idx:
            node, idx = node._next(key, idx)
        if node._value is __NON_TERMINAL__:
            raise KeyError(key)
        node._value = __NON_TERMINAL__

    def __contains__(self, key):
        node = self
        keylen = len(key)
        idx = 0
        while idx != keylen and node is not None:
            node, idx = node._find(key, idx)
        return False if node is None else (node._value is not __NON_TERMINAL__)

    def __iter__(self):
        return _keys(self, [])

    def __len__(self):
        return _count(self)

    def __repr__(self):
        string = ['trie({']
        first = True
        for key, value in _items(self, []):
            if first:
                first = False
            else:
                string.append(', ')
            string.append(repr(key))
            string.append(': ')
            string.append(repr(value))
        string.append('})')
        return ''.join(string)

    def key(self, string, start=0, end=None, default=__NON_TERMINAL__):
        """
        Return the longest key that is a prefix of ``string`` (beginning at
        ``start`` and ending at ``end``).
        If no key matches, raise a `KeyError` or return the ``default`` value
        if it was set.
        """
        return self.item(string, start, end, default)[0]

    def keys(self, *scan):
        """
        Return all keys (that are a prefix of ``string``
        (beginning at ``start`` (and terminating before ``end``))).
        """
        l = len(scan)
        if l == 0:
            return _keys(self, [])
        else:
            if l == 1:
                scan = (scan[0], 0)
            getKey = lambda string, idx, value: string[scan[1]:idx]
            return self._scan(getKey, *scan)

    def value(self, string, start=0, end=None, default=__NON_TERMINAL__):
        """
        Return the value of the longest key that is a prefix of ``string``
        (beginning at ``start`` and ending at ``end``).
        If no key matches, raise a `KeyError` or return the ``default`` value
        if it was set.
        """
        return self.item(string, start, end, default)[1]

    def values(self, *scan):
        """
        Return all values (for keys that are a prefix of ``string``
        (beginning at ``start`` (and terminating before ``end``))).
        """
        l = len(scan)
        if l == 0:
            return _values(self)
        else:
            if l == 1:
                scan = (scan[0], 0)
            getValue = lambda string, idx, value: value
            return self._scan(getValue, *scan)

    def item(self, string, start=0, end=None, default=__NON_TERMINAL__):
        """
        Return the key, value pair of the longest key that is a prefix of
        ``string`` (beginning at ``start`` and ending at ``end``).
        If no key matches, raise a `KeyError` or return the `None`,
        ``default`` pair if any ``default`` value was set.
        """
        node = self
        strlen = len(string)
        start, end = trie.__offsets(strlen, start, end)
        idx = start
        last = self._value
        while idx < strlen:
            node, idx = node._find(string, idx, end)
            if node is None:
                break
            elif node._value is not __NON_TERMINAL__:
                last = node._value
        return trie.__check(last, string[start:idx], default)

    def items(self, *scan):
        """
        Return all key, value pairs (for keys that are a prefix of ``string``
        (beginning at ``start`` (and terminating before ``end``))).
        """
        l = len(scan)
        if l == 0:
            return _items(self, [])
        else:
            if l == 1:
                scan = (scan[0], 0)
            getItem = lambda string, idx, value: (string[scan[1]:idx], value)
            return self._scan(getItem, *scan)

    def isPrefix(self, prefix):
        "Return True if any key starts with ``prefix``."
        node = self
        plen = len(prefix)
        idx = 0
        while idx < plen:
            len_left = plen - idx
            for edge, child in node._edges.values():
                e = edge[:len_left] if (len_left < len(edge)) else edge
                if prefix.startswith(e, idx):
                    node = child
                    idx += len(edge)
                    break
            else:
                return False
        return True

    def iter(self, prefix):
        "Return an iterator over all keys that start with ``prefix``."
        node = self
        plen = len(prefix)
        idx = 0
        while idx < plen:
            try:
                node, idx = node._next(prefix, idx)
            except KeyError:
                break
        return node._accumulate(prefix, idx)

    def _accumulate(self, prefix, idx):
        node = self
        accu = [prefix]
        if idx != len(prefix):
            remainder = prefix[idx:]
            for edge, child in node._edges.values():
                if edge.startswith(remainder):
                    node = child
                    accu.append(edge[len(remainder):])
                    break
            else:
                return iter([])
        return _keys(node, accu)


import math
class Vector:
    def __init__(self, val=None):
        if val:
            self.val = val
        else:
            self.val = ()


    def add(self, val):
        self.val += (val , )

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return self.__str__()


    def get_size(self):
        if len(self) == 0:
            return 0
        pow_sum = 0
        for v in self.val:
            pow_sum += v*v
        return math.sqrt(pow_sum)



    def div_by_num(self,num):
        ans = Vector()
        for v in self.val:
            ans.add(v/num)
        return ans


    def normalize(self):
        self.val = self.div_by_num(self.get_size())


    def add_vector(self, vector):
        ans = Vector()
        if self.get_size() == vector.get_size():
            for i in range(len(self.val)):
                ans.add(self.val[i] + vector.val[i])
        return ans


    def __len__(self):
        return len(self.val)


    @staticmethod
    def multi_2_vec(vector1, vector2):
        ans = 0
        for i in range(len(vector1.val)):
            ans += vector1.val[i] * vector2.val[i]
        return ans


    @staticmethod
    def similaroty_cos(vector1, vector2):
        if vector2.get_size() == 0 or vector1.get_size() == 0:
            return 0.0
        return Vector.multi_2_vec(vector1, vector2)/(vector1.get_size() * vector2.get_size())



