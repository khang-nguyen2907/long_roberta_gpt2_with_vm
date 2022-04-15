import os
import pandas as pd
import re
import spacy
from spacy.attrs import intify_attrs
from transformers import AutoTokenizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
nlp2 = spacy.load("en_core_sci_lg")
with open("./files/stopwords.txt") as f: 
    stopword_kaggle = f.read().split("\n")
stopwords_nltk = stopwords.words('english')
new_stop_words = ['many', 'us', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                  'today', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                  'september', 'october', 'november', 'december', 'today', 'old', 'new', 'yesterday', 'tomorrow', '.', ',']
added_stop_words = stopwords_nltk + new_stop_words
stop_words_spacy = nlp.Defaults.stop_words
all_stop_words = stopword_kaggle + added_stop_words + list(stop_words_spacy)
all_stop_words = set(all_stop_words)
print(len(all_stop_words))
#######################################################################################################
#CONSTANTS 
# Special token ids.
PAD_ID = 1
UNK_ID = 3
CLS_ID = 0
SEP_ID = 2
MASK_ID = 50264

# Special token words.
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
CLS_TOKEN = '<s>'
SEP_TOKEN = '</s>'
MASK_TOKEN = '<mask>'

#Never split text
NEVER_SPLIT_TAG = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]
special_tags = set(NEVER_SPLIT_TAG)

#######################################################################################################
#TAGGER: this is uded to extract nount phrase in a input

def get_tags_spacy(nlp, text):
    doc = nlp(text)    
    entities_spacy = [] # Entities that Spacy NER found
    for ent in doc.ents:
        entities_spacy.append([ent.text, ent.start_char, ent.end_char, ent.label_])
    return entities_spacy

def tag_all(nlp, text, entities_spacy):
    # if ('neuralcoref' in nlp.pipe_names):
    #     nlp.pipeline.remove('neuralcoref')    
    # neuralcoref.add_to_pipe(nlp) # Add neural coref to SpaCy's pipe    
    doc = nlp(text)
    return doc

def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result

def tag_chunks(doc):
    #rechunk, update doc for new type of token like very span in spans
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'ENTITY'}, string_store))

def tag_chunks_spans(doc, spans, ent_type):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': ent_type}, string_store))

def clean(text):
    text = text.strip('[(),- :\'\"\n]\s*')
    text = text.replace('—', ' - ')
    text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)(\"\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.\/)(\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z]{3,}\.)([A-Z]+[a-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)([A-Za-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    
    text = re.sub('’', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('‘', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('“', ' "', text, flags=re.UNICODE)
    text = re.sub('”', ' "', text, flags=re.UNICODE)
    text = re.sub("\|", ", ", text, flags=re.UNICODE) 
    text = text.replace('\t', ' ')
    text = re.sub('…', '.', text, flags=re.UNICODE)           # elipsis
    text = re.sub('â€¦', '.', text, flags=re.UNICODE)          
    text = re.sub('â€“', '-', text)           # long hyphen
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
    text = re.sub(' – ', ' . ', text, flags=re.UNICODE).strip()

    return text

def tagger(text):  
    # nlp = spacy.load("en_core_web_sm")
    text = clean(text)
    document = nlp(text)
    tag_chunks(document)    
    
    # chunk - somethin OF something
    spans_change = []
    for i in range(2, len(document)):
        w_left = document[i-2]
        w_middle = document[i-1]
        w_right = document[i]
        if w_left.dep_ == 'attr':
            continue
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY' and (w_middle.text == 'of'): # or w_middle.text == 'for'): #  or w_middle.text == 'with'
            spans_change.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change, 'ENTITY')
    
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: verb + adp; verb + part 
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'ADP' or w_right.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: adp + verb; part  + verb
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_right.pos_ == 'VERB' and (w_left.pos_ == 'ADP' or w_left.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')
    
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk all between LRB- -RRB- (something between brackets)
    start = 0
    end = 0
    spans_between_brackets = []
    for i in range(0, len(document)):
        if ('-LRB-' == document[i].tag_ or r"(" in document[i].text):
            start = document[i].i
            continue
        if ('-RRB-' == document[i].tag_ or r')' in document[i].text):
            end = document[i].i + 1
        if (end > start and not start == 0):
            span = document[start:end]
            try:
                assert (u"(" in span.text and u")" in span.text)
            except:
                pass
                #print(span)
            spans_between_brackets.append(span)
            start = 0
            end = 0
    tag_chunks_spans(document, spans_between_brackets, 'ENTITY')
            
    # chunk entities
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'ENTITY')
    

    chunks = [token.text for token in document if not token.is_punct]
    return chunks

#######################################################################################################
#KNOWLEDGE
class KnowledgeGraph(object):
    def __init__(self, txt_path, encoder_tokenizer, decoder_tokenizer,predicate=True, is_for = "encoder"):
        self.predicate = predicate
        self.txt_file_path = txt_path
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.lookup_table = self._create_lookup_table()
        self.special_tags = set(NEVER_SPLIT_TAG)
        self.end_punct = set([".", "?", "!"])
        self.is_for = is_for

    def get_question_with_kg_kw(self, sent_batch, max_entities):
        inputs = [] #[['a', 'b'], ['c', 'd']]
        for s in sent_batch:
            sent = self.extract_keyword(s) #['Hello', '!', 'My name', 'is', 'Khang', '.'], ...
            inputs.append(sent)
        know_sent_batch = [] #[['a, e1, e2, e3', 'b, e4, e5, e6'], ['c, e7, e8, e9', 'd, e10, e11, e12']]
        for split_sent in inputs: #['a', 'b'], ['c', 'd']
            a_sentence = [] #['a', 'e1', 'e2', 'e3']
            for token in split_sent: #token = 'abcd', ... 
                entities = list(self.lookup_table.get(token.strip(), []))[:max_entities]
                merged_entities = [", ".join(entities)]
                kw_entities = self.extract_terms(merged_entities[0])
                entities = kw_entities #['e1', 'e2', 'e3', ...]
                a_e = ", ".join([token] + entities)
                if len(a_sentence) == 0:
                    a_sentence.append(a_e) #['a, e1, e2, e3']
                else:
                    a_sentence += [a_e] #['a, e1, e2, e3', 'b, e4, e5, e6']
            know_sent_batch.append(", ".join(a_sentence))
        return know_sent_batch
                

    
    def _create_lookup_table(self):
        lookup_table = {}
        print("[KnowledgeGraph] Loading spo from {}".format(self.txt_file_path))
        with open(self.txt_file_path, encoding="utf8", errors='ignore') as f:
            for line in f:
                try:
                    subj, pred, obje = line.strip().split("\t")
                except:
                    #print("[KnowledgeGraph] Bad spo:", line)
                    continue
                subj, pred, obje = subj.strip(), pred.strip(), obje.strip()
                subj, pred, obje = subj.lower(), pred.lower(), obje.lower()
                if self.predicate:
                    value = pred +' ' + obje
                else:
                    value = obje
                if subj in lookup_table.keys():
                    lookup_table[subj].add(value)
                else:
                    lookup_table[subj] = set([value])
        return lookup_table
    
    def extract_keyword(self, text): 
        tokens = tagger(text)
        filtered_tokens = [w for w in tokens if not w.lower() in all_stop_words]
        return filtered_tokens
    
    def extract_terms(self, document): 
        if isinstance(document, str): 
            doc_term = nlp2(document)
            return list(map(str,list(doc_term.ents))) #return a list 
        else: 
            sents = []
            for d in document: 
                sents.append(", ".join(list(map(str,list(doc_term.ents))))) #return a list of sentences_kw
            return sents
    
    def add_knowledge_with_vm(self, sent_batch, max_entities = 2, add_pad = True, max_length = 256): 
        r"""

        args: 
            sent_batch (List) -- list of input sentence, e.g., ["abcd", "efgh]. If len(sent_batch) == 2, then the input will be treated as 
                                 ```[cls] sent_batch[0] [sep] sent_batch[1] [sep]```. Else if len(sent_batch) == 1, then the input will be treated 
                                 as ```[cls] sent_batch[0]```
            max_entities (int) -- In case a token (noun or noun phrase) has a lot of information (entities) in Knowledge Graph (KG). 
                                  ```max_entities``` will restrict the number of entities are injected in that token in the input sentence
            add_pad (bool) -- 
            max_length (int) -- A fixed length. If a sentence is not reach ```max_length```, it will be paded, if it exceeds ```max_length```
                                it will be truncated
        
        Return: 
            know_sent_batch (List) -- a list contains a list of token that are tokenized. On the other hands, a list contains a list of tokens 
                                      from tokenized sentence tree 
            position_batch (List) -- a list contains a list of position index, it is ```soft_position``` in the original paper (K-BERT) paper. 
                                     Because there are many information (entities) are added to the input sentence. Therefore, we can not create 
                                     ```position_ids``` as usual from origional input. We need to use ```position_batch``` as ```position_ids``` 
                                     in the embeddings part. 
            visible_matrix_batch (List) -- It acts as an ```attention_mask```. It decides which token is related to the others 
        """
        if len(sent_batch) > 1: 
            sent_1 = tagger(sent_batch[0]) #sent_1 =  ['What', 'is', 'your name', '?']
            sent_2 = tagger(sent_batch[1]) #sent_2 =  ['Hello', '!', 'My name', 'is', 'Khang', '.']
            sent_1_ = [' '+w if w not in self.end_punct else w for w in sent_1[1:]] 
            sent_2_ = [' '+w if w not in self.end_punct else w for w in sent_2[1:]]
            inputs = [[CLS_TOKEN] + sent_1_ + [SEP_TOKEN] + sent_2_ + [SEP_TOKEN]] # input =  [['<s>', 'What', ' is', ' your name', '?', '</s>', 'Hello', '!', 'My name', 'is', 'Khang', '.', '</s>']]
        else: 
            #input sentence: "Yesterday, I got eye redness. Today, my eyes turns badly. Do I get retinits?"
            sent = tagger(sent_batch[-1]) #['Yesterday', 'I', 'got', 'eye redness', 'Today', 'my', 'eyes turns badly', 'Do', 'I', 'get', 'retinits']
            sent_ = [' '+w if w not in self.end_punct else w for w in sent[1:]] #['Yesterday', ' I', ' got', ' eye redness', ' Today', ' my', ' eyes turns badly', ' Do', ' I', ' get', ' retinits']]
            inputs = [[CLS_TOKEN] + sent_] # [['<s>', 'Yesterday', 'I', 'got', 'eye redness', 'Today', 'my', 'eyes turns badly', 'Do', 'I', 'get', 'retinits']]
        
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        
        for split_sent in inputs: 
            #create tree 
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1 
            abs_idx = -1
            abs_idx_src = []

            for token in split_sent: #token = '<s>', 'abcd', ... 
                entities = list(self.lookup_table.get(token.strip(), []))[:max_entities]
                pr_token = self.encoder_tokenizer.tokenize(token) 
                pr_entities = [self.encoder_tokenizer.tokenize(e) for e in entities]
                sent_tree.append((pr_token, pr_entities)) # [('<s>', [])]

                if token.strip() in self.special_tags: 
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else: 
                    token_pos_idx = [pos_idx + i for i in range(1, len(pr_token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(pr_token) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities: #ent = 'c c', 'd d'
                    pr_ent = self.encoder_tokenizer.tokenize(ent)
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(pr_ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(pr_ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)
                
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
            
            #Get know_sent, pos 
            know_sent = []
            pos = []
            for i in range(len(sent_tree)): 
                word = sent_tree[i][0] # list 
                if len(word) == 1 and word[-1] in self.special_tags: 
                    know_sent += word
                else: 
                    know_sent += word 
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])): 
                    add_word = sent_tree[i][1][j]
                    know_sent += add_word 
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            #calculate visible matrix 
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1
            
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [PAD_TOKEN] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)

        return know_sent_batch, position_batch, visible_matrix_batch

    def add_knowledge_with_vm_kw_kg(self, sent_batch, max_entities = 2, add_pad = True, max_length = 256): 
        r"""
        This function for model with the input: 

        input encoder: <s>question keywords with KG queried information 

        input decoder: <s>question keywords with KG queried information <\s> answer <\s>

        ```tagger``` is replaced by a keyword extraction function 

        Because the answer is just used as a label, therefore it is not processed to have KG information extraction, just question 
        keywords are allowed to be queried. So that, from the SEP_TOKEN, ```do_query``` flag will be set to False, and there is no 
        any token is queried from KG since that. 


        args: 
            sent_batch (List) -- list of input sentence, e.g., ["abcd", "efgh]. If len(sent_batch) == 2, then the input will be treated as 
                                 ```[cls] sent_batch[0] [sep] sent_batch[1] [sep]```. Else if len(sent_batch) == 1, then the input will be treated 
                                 as ```[cls] sent_batch[0]```
            max_entities (int) -- In case a token (noun or noun phrase) has a lot of information (entities) in Knowledge Graph (KG). 
                                  ```max_entities``` will restrict the number of entities are injected in that token in the input sentence
            add_pad (bool) -- 
            max_length (int) -- A fixed length. If a sentence is not reach ```max_length```, it will be paded, if it exceeds ```max_length```
                                it will be truncated
        
        Return: 
            know_sent_batch (List) -- a list contains a list of token that are tokenized. On the other hands, a list contains a list of tokens 
                                      from tokenized sentence tree 
            position_batch (List) -- a list contains a list of position index, it is ```soft_position``` in the original paper (K-BERT) paper. 
                                     Because there are many information (entities) are added to the input sentence. Therefore, we can not create 
                                     ```position_ids``` as usual from origional input. We need to use ```position_batch``` as ```position_ids``` 
                                     in the embeddings part. 
            visible_matrix_batch (List) -- It acts as an ```attention_mask```. It decides which token is related to the others 
        """
        if len(sent_batch) > 1: 
            sent_1 = self.extract_keyword(sent_batch[0]) #sent_1 =  ['What', 'is', 'your name', '?']
            sent_2 = self.extract_keyword(sent_batch[1]) #sent_2 =  ['Hello', '!', 'My name', 'is', 'Khang', '.']
            sent_1_ = [' '+w if w not in self.end_punct else w for w in sent_1[1:]] 
            sent_2_ = [' '+w if w not in self.end_punct else w for w in sent_2[1:]]
            inputs = [[CLS_TOKEN] + sent_1_ + [SEP_TOKEN] + sent_2_ + [SEP_TOKEN]] # input =  [['<s>', 'What', ' is', ' your name', '?', '</s>', 'Hello', '!', 'My name', 'is', 'Khang', '.', '</s>']]
        else: 
            #input sentence: "Yesterday, I got eye redness. Today, my eyes turns badly. Do I get retinits?"
            sent = self.extract_keyword(sent_batch[-1]) #['Hello', '!', 'My name', 'is', 'Khang', '.']
            sent_ = [' '+w if w not in self.end_punct else w for w in sent[1:]] #['Hello', '!', ' My name', ' is', ' Khang', '.']
            inputs = [[CLS_TOKEN] + sent_] # [['<s>', 'Hello', '!', 'My name', 'is', 'Khang', '.']]
        
        do_query = True
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        
        tokenize_lst = []
        for split_sent in inputs: 
            #create tree 
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1 
            abs_idx = -1
            abs_idx_src = []

            for token in split_sent: #token = '<s>', 'abcd', ... 
                if token.strip() == SEP_TOKEN: 
                    do_query = False

                if do_query: 
                    entities = list(self.lookup_table.get(token.strip(), []))[:max_entities]
                    if entities: 
                        merged_entities = [", ".join(entities)]
                        kw_entities = self.extract_terms(merged_entities[0])
                        entities = kw_entities
                else: 
                    entities = []
                
                if self.is_for == "encoder": 
                    pr_token = self.encoder_tokenizer.tokenize(token) 
                    pr_entities = [self.encoder_tokenizer.tokenize(e) for e in entities]
                else:
                    pr_token = self.decoder_tokenizer.tokenize(token) 
                    pr_entities = [self.decoder_tokenizer.tokenize(e) for e in entities]
                sent_tree.append((pr_token, pr_entities)) # [('<s>', [])]

                if token.strip() in self.special_tags: 
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else: 
                    token_pos_idx = [pos_idx + i for i in range(1, len(pr_token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(pr_token) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities: #ent = 'c c', 'd d'
                    if self.is_for == "encoder":
                        pr_ent = self.encoder_tokenizer.tokenize(ent)
                    else: 
                        pr_ent = self.decoder_tokenizer.tokenize(ent)
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(pr_ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(pr_ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)
                
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
            
            #Get know_sent, pos 
            know_sent = []
            pos = []
            for i in range(len(sent_tree)): 
                word = sent_tree[i][0] # list 
                if len(word) == 1 and word[-1] in self.special_tags: 
                    know_sent += word
                else: 
                    know_sent += word 
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])): 
                    add_word = sent_tree[i][1][j]
                    know_sent += add_word 
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            #calculate visible matrix 
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1
            
            tokenize_lst.append(know_sent)
        max_length_ = max([len(x) for x in tokenize_lst])
        max_length_ = ((max_length_ // 512) + 1) * 512
        if max_length_ < max_length: 
            max_length = max_length_
        for know_sent in tokenize_lst:
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [PAD_TOKEN] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)

        return know_sent_batch, position_batch, visible_matrix_batch    

    def tokenizer_with_vm(self, sent_batch, max_entities = 2, add_pad = True, max_length = 256): 
        r"""
        This function for model with the input: 

        input encoder: <s>question keywords with KG queried information 

        input decoder: <s>question keywords with KG queried information <\s> answer <\s>

        ```tagger``` is replaced by a keyword extraction function 

        Because the answer is just used as a label, therefore it is not processed to have KG information extraction, just question 
        keywords are allowed to be queried. So that, from the SEP_TOKEN, ```do_query``` flag will be set to False, and there is no 
        any token is queried from KG since that. 


        args: 
            sent_batch (List) -- list of input sentence, e.g., ["abcd", "efgh]. Going through each element in `sent_batch`
                                  to process them: add special token, group tagging, ... 
            max_entities (int) -- In case a token (noun or noun phrase) has a lot of information (entities) in Knowledge Graph (KG). 
                                  ```max_entities``` will restrict the number of entities are injected in that token in the input sentence
            add_pad (bool) -- 
            max_length (int) -- A fixed length. If a sentence is not reach ```max_length```, it will be paded, if it exceeds ```max_length```
                                it will be truncated
        
        Return: 
            know_sent_batch (List) -- a list contains a list of token that are tokenized of each element in `sent_batch`. On the other hands, a list contains a list of tokens 
                                      from tokenized sentence tree 
            position_batch (List) -- a list contains a list of position index, it is ```soft_position``` in the original paper (K-BERT) paper. 
                                     Because there are many information (entities) are added to the input sentence. Therefore, we can not create 
                                     ```position_ids``` as usual from origional input. We need to use ```position_batch``` as ```position_ids``` 
                                     in the embeddings part. 
            visible_matrix_batch (List) -- It acts as an ```attention_mask```. It decides which token is related to the others 
        """
        
        #input sentence: "Yesterday, I got eye redness. Today, my eyes turns badly. Do I get retinits?"
        inputs = [] #[['<s>', 'Hello', '!', ' My name', ' is', ' Khang', '.'], ['<s>', 'Hi', '!', ' How', ' are', ' you', '?']]
        for s in sent_batch:
            sent = self.extract_keyword(s) #['Hello', '!', 'My name', 'is', 'Khang', '.'], ...
            sent_ = [' '+w if w not in self.end_punct else w for w in sent[1:]] #['Hello', '!', ' My name', ' is', ' Khang', '.'], ...
            inputs.append([CLS_TOKEN] + sent_) # [['<s>', 'Hello', '!', 'My name', 'is', 'Khang', '.']]
        
        
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        
        for split_sent in inputs: 
            #create tree 
            do_query = True
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1 
            abs_idx = -1
            abs_idx_src = []

            for token in split_sent: #token = '<s>', 'abcd', ... 
                if token.strip() == SEP_TOKEN: 
                    do_query = False

                if do_query: 
                    entities = list(self.lookup_table.get(token.strip(), []))[:max_entities]
                    if entities: 
                        merged_entities = [", ".join(entities)]
                        kw_entities = self.extract_terms(merged_entities[0])
                        entities = kw_entities
                else: 
                    entities = []
                
                if self.is_for == "encoder": 
                    pr_token = self.encoder_tokenizer.tokenize(str(token)) 
                    pr_entities = [self.encoder_tokenizer.tokenize(str(e)) for e in entities]
                else:
                    pr_token = self.decoder_tokenizer.tokenize(str(token)) 
                    pr_entities = [self.decoder_tokenizer.tokenize(str(e)) for e in entities]
                sent_tree.append((pr_token, pr_entities)) # [('<s>', [])]

                if token.strip() in self.special_tags: 
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else: 
                    token_pos_idx = [pos_idx + i for i in range(1, len(pr_token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(pr_token) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities: #ent = 'c c', 'd d'
                    ent = str(ent)
                    if self.is_for == "encoder":
                        pr_ent = self.encoder_tokenizer.tokenize(ent)
                    else: 
                        pr_ent = self.decoder_tokenizer.tokenize(ent)
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(pr_ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(pr_ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)
                
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
            
            #Get know_sent, pos 
            know_sent = []
            pos = []
            for i in range(len(sent_tree)): 
                word = sent_tree[i][0] # list 
                if len(word) == 1 and word[-1] in self.special_tags: 
                    know_sent += word
                else: 
                    know_sent += word 
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])): 
                    add_word = sent_tree[i][1][j]
                    know_sent += add_word 
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            #calculate visible matrix 
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1
            
            #after processing a text which is now a list of chunks in original text in `inputs`
            #appending them to 3 lists
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)

        #make sure that 3 lists have the same length corresponding to length of text in `sent_batch`
        #because each text needs to have its own `know_sent`, `pos`, `visible_matrix`
        assert len(know_sent_batch) == len(position_batch) == len(visible_matrix_batch), f"len(know_sent_batch): {len(know_sent_batch)}, len(position_batch): {len(position_batch)}, len(visible_matrix_batch): {len(visible_matrix_batch)} do not equal"
        
        #go find the longest length in `know_sent_batch` because it is not good to pad every sentence 
        #to 4096, there may be some sentences with a few of token, padding it to 4096 leads to lack of memory for training
        max_length_ = max([len(x) for x in know_sent_batch])

        #make sure that `max_length_` to pad a sentence is multiple of 512
        max_length_ = ((max_length_ // 512) + 1) * 512

        #`max_length` is originally set as 4096, it will be the lognest - `max_length_` if it is greater than `max_length_`
        if max_length_ < max_length: 
            max_length = max_length_
        
        #padding
        for idx, (know_sent, pos, visible_matrix) in enumerate(zip(know_sent_batch, position_batch, visible_matrix_batch)):
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [PAD_TOKEN] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch[idx] = know_sent
            position_batch[idx] = pos
            visible_matrix_batch[idx] = visible_matrix
            
            

        return know_sent_batch, position_batch, visible_matrix_batch  

    
