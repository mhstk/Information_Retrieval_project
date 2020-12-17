import re
import pickle
from patrica import trie
from linklist import LinkedList
DOC_PATH = './myDoc/'
DIC_FILE_NAME = "./data/Dictionary.fa"
DIC_PCKL_NAME = "./data/Dictionary.obj"
VERB_FILE_NAME = "./data/VerbList.fa"
VERB_PCKL_NAME = "./data/VerbList.obj"
STOPWORDS_FILE_NAME = "./data/stopwords.fa"

VERB_BOOL = False

class Persian_stemmer:
    def __init__(self, verb_dic_bool=VERB_BOOL):
        self.persian_dictionary = trie()
        self.verb_dic = trie()
        self.load_dictionary_pckl()
        self.verb_dic_bool = verb_dic_bool
        if verb_dic_bool:
            self.load_verb_pckl()
        # self.load_verb()


 
    def load_verb(self):
        words = []
        with open(VERB_FILE_NAME , 'r' , encoding='utf-8') as f:
            words = f.readlines()
            for word in words:
                verbs = word.split('\t')
                self.verb_dic[verbs[0]] = verbs[2]
                self.verb_dic[verbs[1]] = verbs[2]
        return self.verb_dic


    def create_per_verb_pckl(self):
        obj = self.load_verb()
        with open(VERB_PCKL_NAME , 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


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


    def create_per_dic_pckl(self):
        obj = self.load_dictionary()
        with open(DIC_PCKL_NAME , 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


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
        # punc = r'\.:!،؛؟»\]\)\}«\[\(\{'
        punc = r'-!$%^&*\(\)_+|~=`\{\}\[\]:";\'<>?,./\\><؟\?،؛«»'
        patterns = [
            # ('" ([^\n"]+) "', r'"\1"'),  # remove space before and after quotation
            # (' (['+ punc_after +'])', r'\1'),  # remove space before
            # ('(['+ punc_before +']) ', r'\1'),  # remove space after
            # ('(['+ punc_after[:3] +'])([^ '+ punc_after +'\d۰۱۲۳۴۵۶۷۸۹])', r'\1 \2'),  # put space after . and :
            # ('(['+ punc_after[3:] +'])([^ '+ punc_after +'])', r'\1 \2'),  # put space after
            # ('([^ '+ punc_before +'])(['+ punc_before +'])', r'\1 \2'),  # put space before
            (r'['+punc+r']', ' ')
        ]
        text = self.sub_patterns(patterns, text)

        return text




    def postix_normalize(self, text):
        patterns = [
            ('\u200c' , ' '),
            ('\u200b' , ''),
            (r'([^ ]ه) ی ', r'\1‌ی '),  # fix ی space
            (r'(^| )(ن?می) ', r'\1' + r'\2'+'\u200c'),  # put zwnj after می, نمی
            (r'(?<=[^\n\d ]{2}) (تر(ین?)?|گری?|های?)(?=[ \n]|$)', '\u200c'+r'\1'),  # put zwnj before تر, تری, ترین, گر, گری, ها, های
            (r'([^ ]ه) (ا(م|یم|ش|ند|ی|ید|ت))(?=[ \n]|$)', r'\1'+'\u200c'+r'\2'),  # join ام, ایم, اش, اند, ای, اید, ات,
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
        if self.verb_dic_bool:
            if token in self.verb_dic:
                if isinstance(self.verb_dic[token], str):
                    return str(self.verb_dic[token])

        main_patterns = [
            (r'\d*' , ''),
            ('([^ ]ه)\u200c(ا(م|یم|ش|ند|ی|ید|ت))$' , r'\1'),
            ('[\u200c](تر(ین?)?|گری?|های?)' , ''),
            ('^(ن?می)[\u200c]' , ''),
        ]
        before = token
        token = self.sub_patterns(main_patterns, token)
        if before != token:
            return token


        if token in self.persian_dictionary:
            return token

        first_patterns = [
            (r'(?<=[^ او])(ان|ات|ی)$' , ''),
        ]
        token = self.sub_patterns(first_patterns, token)
        if token in self.persian_dictionary:
            return token
        token = before
        

        
        patterns = [
            (r' (تر(ین?)?|گری?)(?=$)' , ''),
            (r'^(ن?می|ب)(?=[^ ]{2,}$)' , ''),
            (r'(?<=[^ او])(م|ت|ش|مان|تان|شان|ی|یم|ید|ند)$' , ''),
            (r'(ان|ات|ها|های)$' , ''),
            (r'([^ ]ه) (ا(م|یم|ش|ند|ی|ید|ت))$' , r'\1'),
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
    def __init__(self):
        self.ps = Persian_stemmer()
        self.last_term_id = 0
        self.last_doc_id = 0
        self.term_frequency = {}
        self.dictionary = {}
        self.postings_lists = {}
        self.doc_list = {}
        self.stopwords = []

    def file_text(self, addr):
        with open(DOC_PATH + addr , 'r', encoding='utf-8') as f:
            data = f.read()
        return data

    def read_a_doc(self, text):
        return self.ps.positioned_tokens(self.ps.stemmed_tokens(text))


    def create_positional_index(self, docs_addr):
        for doc_addr in docs_addr:
            doc_id = self.get_doc_id(doc_addr)
            text = self.file_text(doc_addr)
            tokens = self.read_a_doc(text)
            

            for token, pos in tokens:
                if token in self.stopwords:
                    continue
                term_id = self.get_term_id(token)
                posting_list = self.get_posting_list(term_id)
                posting_list.addNode((doc_id , pos) , doc_id+pos)

        return self.postings_lists



    def get_posting_list(self, term_id):
        if term_id in self.postings_lists:
            return self.postings_lists[term_id]
        self.postings_lists[term_id] = LinkedList()
        return self.postings_lists[term_id]


    def get_doc_id(self, doc_addr):
        if doc_addr in self.doc_list:
            return self.doc_list[doc_addr]
        self.last_doc_id += 1
        self.doc_list[doc_addr] = self.last_doc_id
        return self.last_doc_id


    def get_term_id(self, term):
        if term in self.dictionary:
            return self.dictionary[term]
        self.last_term_id += 1
        self.dictionary[term] = self.last_term_id
        return self.last_term_id
        


    def find_stopwords(self, docs_addr):
        term_freq = {}
        for doc_addr in docs_addr:
            doc_id = self.get_doc_id(doc_addr)
            text = self.file_text(doc_addr)
            tokens = self.read_a_doc(text)
            

            for token, pos in tokens:
                if token in term_freq:
                    term_freq[token] += 1
                else:
                    term_freq[token] = 1
                
        term_freq = {k: v for k, v in sorted(term_freq.items(), key=lambda item: item[1], reverse=True )}
        max_v = max(term_freq.values())
        self.stopwords = [k for k,v in term_freq.items() if v >= max_v*0.2]

        with open(STOPWORDS_FILE_NAME , 'w', encoding='utf-8') as fwriter: 
            fwriter.writelines([k+'\n' for k in self.stopwords])
        
        return self.stopwords


    def read_stopwords(self):
        with open(STOPWORDS_FILE_NAME, 'r', encoding='utf-8') as freader:
            for line in freader.readlines():
                self.stopwords.append(line[:-1])


    def posting_list_by_term(self, term):
        if term in self.dictionary:
            term_id = self.dictionary[term]
            return self.postings_lists[term_id].to_list()
        return None

    def boolean_query(self, text):
        queris = self.ps.stemmed_tokens(text)
        ans = set()
        for term in queris:
            doc_ids = self.posting_list_by_term(term)
            if doc_ids != None:
                # ans |= set([(doc_id, pos) for doc_id, pos in doc_ids])
                ans |= set([doc_id for doc_id, pos in doc_ids])
            # ans += [doc_id for doc_id, pos in self.posting_list_by_term(term)]
            
        return ans


    


    
        




    




if __name__ == '__main__':


    p = PersianIR()


    addr_list = [f'{x}.txt' for x in range(1,11)]



    # print(p.find_stopwords(addr_list))
    p.read_stopwords()
    res = p.create_positional_index(addr_list)
    # res = p.read_a_doc(p.file_text('10.txt'))

    # print(res)

    # print(len(res.items()))
    # print(p.dictionary)

    print(p.boolean_query('یافتم'))


    # with open(DOC_PATH + 'result.txt' , 'w', encoding='utf-8') as f:
    #     for i in res:
    #         f.write(str(i))
    #         f.write('\n')
