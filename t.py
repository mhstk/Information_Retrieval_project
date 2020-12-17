# import re

# print(re.sub(re.compile('([^ ]ه)\u200c(ا(م|یم|ش|ند|ی|ید|ت))$') ,r'\1' , 'کشیده' + '\u200c' +'اند' ))


from main import Persian_stemmer,PersianIR, DOC_PATH



p = PersianIR()
word = 'می رفتم'
# print([word])
# print(p.stemmed_tokens(word))

# p.create_per_verb_pckl()
# p.load_verb_pckl()
# print('here')
# # p.load_verb()

# if word in p.verb_dic:
#     print(p.verb_dic[word])

addr_list = [f'{x}.txt' for x in range(1,11)]
p.find_stopwords(addr_list)


res = p.read_a_doc(p.file_text('3.txt'))



with open(DOC_PATH + 'result.txt' , 'w', encoding='utf-8') as f:
    for i in res:
        f.write(str(i))
        f.write('\n')