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

USE_CHAMPION_LISTS = True
CHAMPION_LIST_LENGTH = 5
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
                if score[doc_id] != 0:
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
                # term_tf_query[self.get_term_id(q)] =  ( 1+math.log10(qf) ) * self.idf(self.get_term_id(q))
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
        # if doc_id == 155:
        #     print('\n\n\nDOC')
        for term_id in self.dictionary.values():
            temp = self.tf_idf(term_id, doc_id)
            # if (doc_id == 155) and temp != 0.0:
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

