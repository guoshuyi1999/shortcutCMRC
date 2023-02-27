import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
import multiprocessing
import logging
import random
import os
from allennlp.modules.elmo import batch_to_ids
from general_utils import flatten_json, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, find_answer_span, feature_gen, token2id

parser = argparse.ArgumentParser(
    description='Preprocessing train + dev files, about 20 minutes to run on Servers.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words.'
                         'Otherwise consider question words first.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, embedding init, etc.')
parser.add_argument('--train_file', type=str, default='Squad2_data/squad22_train.json')
parser.add_argument('--dev_file', type=str, default='Squad2_data/squad22_dev.json')
parser.add_argument('--output_dir', type=str, default='Squad2_data/')

#6.11加的

parser.add_argument('--no_prepend_answer', action='store_true')      #--no_prepend_answer' train有
parser.add_argument('--no_position', action='store_true')            #--no_position  train/dev都有
#加的
#加的
parser.add_argument('--no_now_question', action='store_true')        #--no_question   train有
parser.add_argument('--input_ablation', type=int,default=0,help='')  #mask-pro/log/wh   train/dev都有  
parser.add_argument('--input_ablation_dev', type=int,default=0,help='')  #mask-pro/log/wh   train/dev都有  




args = parser.parse_args()
input_ablation=args.input_ablation
input_ablation_dev=args.input_ablation_dev
trn_file = args.train_file
dev_file = args.dev_file
wv_file = args.wv_file
wv_dim = args.wv_dim
nlp = spacy.load('en_core_web_sm', disable=['parser'])

random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))
log.info('{} file)'.format(args.train_file))
log.info('{} file)'.format(args.dev_file))
log.info('input_ablation{}'.format(args.input_ablation))
log.info('input_ablation_dev{}'.format(args.input_ablation_dev))

glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
log.info('glove loaded.')

#===============================================================
#=================== Work on training data =====================
#===============================================================
def proc_train(ith, article):
    rows = []
    i=0
    if i<50000:
        # i=i+1
        # print(i)
        for paragraph in article['paragraphs']:
            
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['orig_answer']
                
                answer = answers['text']
                answer_start = answers['answer_start']
                answer_end = answers['answer_start'] + len(answers['text'])
                answer_choice = 0 if answer == 'CANNOTANSWER' else\
                                1 if qa['yesno'] == 'y' else\
                                2 if qa['yesno'] == 'n' else\
                                3 # Not a yes/no question
                if answer_choice != 0:
                    """
                    0: Do not ask a follow up question!
                    1: Definitely ask a follow up question!
                    2: Not too important, but you can ask a follow up.
                    """
                    answer_choice += 10 * (0 if qa['followup'] == "n" else\
                                        1 if qa['followup'] == "y" else\
                                        2)
                else:
                    answer_start, answer_end = -1, -1
                rows.append((ith, question, answer, answer_start, answer_end, answer_choice))
        return rows, context

train, train_context = flatten_json(trn_file, proc_train)
train = pd.DataFrame(train, columns=['context_idx', 'question', 'answer',
                                    'answer_start', 'answer_end', 'answer_choice'])
log.info('train json data flattened.')

print(train)

trC_iter = (pre_proc(c) for c in train_context)
trQ_iter = (pre_proc(q) for q in train.question)
trC_docs = [doc for doc in nlp.pipe(trC_iter, batch_size=64, n_process=args.threads)]
trQ_docs = [doc for doc in nlp.pipe(trQ_iter, batch_size=64, n_process=args.threads)]

# tokens
trC_tokens = [[normalize_text(w.text) for w in doc] for doc in trC_docs]
trQ_tokens = [[normalize_text(w.text) for w in doc] for doc in trQ_docs]
trC_unnorm_tokens = [[w.text for w in doc] for doc in trC_docs]
log.info('All tokens for training are obtained.')

log.info('All tokens for training are obtained.')
###############################
####加input_ablation
def generate_ablated_input(option, parsed_doc, parsed_query):
    # print('option',option)
    if option == "drop_logical_words":
        logical_words = ["not","n't","all","any","each","every","few","if", "more", "most","no","nor","other","same", "some","than",]
        dropped_doc, mask_dict_c= mask_tokens(parsed_doc, logical_words)
        dropped_query, mask_dict_q= mask_tokens(parsed_query, logical_words)

        #devQ_tokens
        ablated_input = {
            "devC_tokens": dropped_doc,
            "devQ_tokens":parsed_query,
            "drop_dict": mask_dict_c,
        }

    elif option == "drop_causal_words":
        causal_words = ["because", "why", "therefore", "cause", "reason", "as", "since"]
        dropped_doc, mask_dict = mask_tokens(parsed_doc, causal_words)
        ablated_input = {
            "devC_tokens": dropped_doc,
            "devQ_tokens":parsed_query,
            "drop_dict": mask_dict,
        }

    elif option == "mask_pronouns":
        pronouns = """i you he she we they it her his mine my our ours their thy your
        hers herself him himself hisself itself me myself one oneself ours
        ourselves ownself self thee theirs them themselves thou thy us""".split()
        # dropped_doc, mask_dict = mask_tokens(
        #     parsed_doc, pronouns, target_poses=["PRP", "PRP$"]
        # )
        #context文章
        dropped_doc, mask_dict_c= mask_tokens(parsed_doc, pronouns)

        dropped_query, mask_dict_q = mask_tokens(parsed_query, pronouns)

        

        ablated_input = {
            "devC_tokens": dropped_doc,
            "devQ_tokens":dropped_query,
            "drop_dict": mask_dict_c,
            "drop_dict_q":mask_dict_q
        }
    

    elif option == "drop_question_except_interrogatives":
        interrogatives = ["what", "when", "where", "who", "which", "why", "whom", "how"]
        dropped_query, mask_dict_q= mask_tokens(
            parsed_query, interrogatives, complement=True  #????????????
        )
        # print('dropped_query',dropped_query)
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": dropped_query,
            "drop_dict": mask_dict_q,
        }
    elif option == "drop_question_is_interrogatives":
        interrogatives = ["what", "when", "where", "who", "which", "why", "whom", "how"]
        dropped_doc, mask_dict_c= mask_tokens(parsed_doc, interrogatives)
        dropped_query, mask_dict_q= mask_tokens(
            parsed_query, interrogatives     
        )
        
        # print('dropped_query',dropped_query)
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": dropped_query,
            "drop_dict": mask_dict_q,
        }

    #no-question"devQ_tokens": [],
    elif option == "drop_all_question_words":
        
        dropped_query, mask_dict_q= mask_all_tokens(
            parsed_query     
        )
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": parsed_query,
        }
        # trQ_tokens = ablated_input["devQ_tokens"]
        # print('trQ_tokens',trQ_tokens)
    
    elif option == "original":
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": parsed_query,
           
        }

    else:
        raise ValueError("Invalid input-ablation option: {}".format(option))

    return ablated_input


#加的[UNK]=drop [MASK]=MASK
def mask_tokens(data, mask_tokens, complement=False, target_poses=None):
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    TOKEN=MASK_TOKEN
    mask_dict = {}
    # token_count = -1
    # print('mask_tokens',mask_tokens)
    # print('data1',data)
    i=0
    j=0
    sent_count = -1
    for sent in data:
        sent_count += 1
        token_count = -1
        # print('sent1',sent)
        for token in sent:
            # print('token1',token)
            token_count += 1
            lower_token = token.lower() 
            if target_poses and not token["pos"] in target_poses:
                continue
            if (lower_token in mask_tokens and not complement) or (
                lower_token not in mask_tokens and complement
            ):  
                if isinstance(mask_tokens, dict):
                    print(2)
                    mask_dict[token_count] = mask_tokens[lower_token]
                    token = mask_tokens[lower_token]
                    # print('token2',token)
                    token=TOKEN
                    # print('token3',token)
                    i=i+1
                else:
                    # print(3)
                    mask_dict[token_count] = TOKEN   #MASK_TOKEN/UNK_TOKEN"[MASK]"    #"UNK"#
                    token=TOKEN
                    token=token
                    sent[token_count]=token
                    # print('token4',token)
                    # print('token_count',token_count)
                    j=j+1
                    if True:
                        token = "\sout{" + token + "}"
                    else:
                        token= UNK_TOKEN
        
                #context文章
        
        # print('sent2',sent)
    data[sent_count]=sent

    data=data
    mask_dict=mask_dict
    #文章
    print('i',i)
    print('j',j)
    # keys=mask_dict.keys()
    # if len(keys)>0:
    #     for i in range(len(data)):
    #         for j in range(len(list(keys))):
    #             if i==list(keys)[j]:
    #                 data[i]=TOKEN  #MASK_TOKEN/UNK_TOKEN

    # # data=data
    # print('data2',data)
    # # print('mask_dict',mask_dict)
    #文章
    return data, mask_dict

    #################################################
    # 
###############################
#加的[UNK]=drop [MASK]=MASK
def mask_all_tokens(data, complement=False, target_poses=None):
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    TOKEN=MASK_TOKEN
    mask_dict = {}
    # data=data
    # mask_tokens=data  #mask——token为自身
    # print(data)
    # print(mask_tokens)
    sent_count = -1
    for sent in data:
        # print('sent1',sent)
        mask_tokens=sent
        # print('mask_tokens',mask_tokens)
        sent_count += 1
        token_count = -1
        for token in sent:
            token_count += 1
            lower_token = token.lower() 
            if target_poses and not token["pos"] in target_poses:
                continue
            if (lower_token in mask_tokens and not complement) or (
                lower_token not in mask_tokens and complement
            ):  
                # print(1)
                if isinstance(mask_tokens, dict):
                    mask_dict[token_count] = mask_tokens[lower_token]
                    token = mask_tokens[lower_token]
                    token=TOKEN
                    # print(2)
                else:
                    mask_dict[token_count] = TOKEN   #MASK_TOKEN/UNK_TOKEN"[MASK]"    #"UNK"#
                    token=TOKEN
                    token=token
                    sent[0]=token 
                    sent[token_count]=token
                    # print(3)
                    if True:
                        token = "\sout{" + token + "}"
                    else:
                        token= UNK_TOKEN
        
                #context文章
        
        # print('sent2',sent)
    data[sent_count]=sent

    data=data
    mask_dict=mask_dict
    return data, mask_dict

    #################################################
    # 
###############################

input_ablation=input_ablation
print(input_ablation)
if input_ablation==0:
    input_ablation="original"
    log.info('0:original')
    ablated_example = generate_ablated_input(
        input_ablation, trC_tokens, trQ_tokens
    )
if input_ablation==1:
    input_ablation="drop_logical_words"
    
    log.info('1:drop_logical_words')
    ablated_example = generate_ablated_input(
        input_ablation, trC_tokens, trQ_tokens
    )
if input_ablation==2:

    input_ablation="mask_pronouns"
    
    log.info('2:mask_pronouns')
    ablated_example = generate_ablated_input(
        input_ablation, trC_tokens, trQ_tokens
    )
if input_ablation==3:
    input_ablation="drop_question_except_interrogatives"  #drop_question_is_interrogatives
    
    log.info('3:drop_question_except_interrogatives')
    ablated_example = generate_ablated_input(
        input_ablation, trC_tokens, trQ_tokens
    )
if input_ablation==4:
    input_ablation="drop_question_is_interrogatives"  #drop_question_is_interrogatives
   
    log.info('4:drop_question_is_interrogatives')
    ablated_example = generate_ablated_input(
        input_ablation, trC_tokens, trQ_tokens
    )
if input_ablation==5:
    input_ablation="drop_all_question_words"  #drop_question_is_interrogatives
   
    log.info('5:drop_all_question_words')
    ablated_example = generate_ablated_input(
        input_ablation, trC_tokens, trQ_tokens
    )

trQ_tokens = ablated_example["devQ_tokens"]
trC_tokens = ablated_example["devC_tokens"]
# print('trQ_tokens',trQ_tokens)
# i1=ablated_example["doc-i1"]
print('trQ_tokens',len(trQ_tokens))
print('trC_tokens',len(trC_tokens))
# a[1]=5
log.info('All mask tokens for training are obtained.')

####################input_ablation


train_context_span = [get_context_span(a, b) for a, b in zip(train_context, trC_unnorm_tokens)]

ans_st_token_ls, ans_end_token_ls = [], []
for ans_st, ans_end, idx in zip(train.answer_start, train.answer_end, train.context_idx):
    ans_st_token, ans_end_token = find_answer_span(train_context_span[idx], ans_st, ans_end)
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)

train['answer_start_token'], train['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
initial_len = len(train)
train.dropna(inplace=True) # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(train), initial_len))
log.info('answer span for training is generated.')

# features
trC_tags, trC_ents, trC_features = feature_gen(trC_docs, train.context_idx, trQ_docs, args.no_match, False)
log.info('features for training is generated: {}, {}, {}'.format(len(trC_tags), len(trC_ents), len(trC_features)))

def build_train_vocab(questions, contexts): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    else:
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in glove_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in glove_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    vocab.insert(2, "<S>")
    vocab.insert(3, "</S>")
    return vocab

# vocab
tr_vocab = build_train_vocab(trQ_tokens, trC_tokens)
trC_ids = token2id(trC_tokens, tr_vocab, unk_id=1)
trQ_ids = token2id(trQ_tokens, tr_vocab, unk_id=1)
trQ_tokens = [["<S>"] + doc + ["</S>"] for doc in trQ_tokens]
trQ_ids = [[2] + qsent + [3] for qsent in trQ_ids]
print(trQ_ids[:10])
# tags
vocab_tag = [''] + list(nlp.get_pipe("tagger").labels)
# vocab_tag = [''] + list(nlp.tagger.labels)
trC_tag_ids = token2id(trC_tags, vocab_tag)
# entities
vocab_ent = list(set([ent for sent in trC_ents for ent in sent]))
trC_ent_ids = token2id(trC_ents, vocab_ent, unk_id=0)

log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
log.info('vocabulary for training is built.')

tr_embedding = build_embedding(wv_file, tr_vocab, wv_dim)
log.info('got embedding matrix for training.')

# don't store row name in csv
#train.to_csv('Squad2_data/train.csv', index=False, encoding='utf8')

meta = {
    'vocab': tr_vocab,
    'embedding': tr_embedding.tolist()
}
with open(os.path.join(args.output_dir, 'train_meta.msgpack'), 'wb') as f:
    msgpack.dump(meta, f)

prev_CID, first_question = -1, []
for i, CID in enumerate(train.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID

result = {
    'question_ids': trQ_ids,
    'context_ids': trC_ids,
    'context_features': trC_features, # exact match, tf
    'context_tags': trC_tag_ids, # POS tagging
    'context_ents': trC_ent_ids, # Entity recognition
    'context': train_context,
    'context_span': train_context_span,
    '1st_question': first_question,
    'question_CID': train.context_idx.tolist(),
    'question': train.question.tolist(),
    'answer': train.answer.tolist(),
    'answer_start': train.answer_start_token.tolist(),
    'answer_end': train.answer_end_token.tolist(),
    'answer_choice': train.answer_choice.tolist(),
    'context_tokenized': trC_tokens,
    'question_tokenized': trQ_tokens
}
with open(os.path.join(args.output_dir, 'train_data.msgpack'), 'wb') as f:
    msgpack.dump(result, f)

log.info('saved training to disk.')

#==========================================================
#=================== Work on dev data =====================
#==========================================================

def proc_dev(ith, article):
    rows = []
    
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            answers = qa['orig_answer']
            
            answer = answers['text']
            answer_start = answers['answer_start']
            answer_end = answers['answer_start'] + len(answers['text'])

        

            answer_choice = 0 if answer == 'CANNOTANSWER' else\
                            1 if qa['yesno'] == 'y' else\
                            2 if qa['yesno'] == 'n' else\
                            3 # Not a yes/no question
            if answer_choice != 0:
                """
                0: Do not ask a follow up question!
                1: Definitely ask a follow up question!
                2: Not too important, but you can ask a follow up.
                """
                answer_choice += 10 * (0 if qa['followup'] == "n" else\
                                       1 if qa['followup'] == "y" else\
                                       2)
            else:
                answer_start, answer_end = -1, -1
            
            ans_ls = []
            for ans in qa['answers']:
                ans_ls.append(ans['text'])
            # print('context',context)
           
            # print('answers',answers)
            # print('answer',answer)
            # print('answer_start',answer_start)
            # print('answer_end',answer_end)
            # print('ans_ls',ans_ls)
            
            rows.append((ith, question, answer, answer_start, answer_end, answer_choice, ans_ls))
    return rows, context

dev, dev_context = flatten_json(dev_file, proc_dev)
dev = pd.DataFrame(dev, columns=['context_idx', 'question', 'answer',
                                 'answer_start', 'answer_end', 'answer_choice', 'all_answer'])
log.info('dev json data flattened.')

print(dev)
## Multiprocessing with 4 processes
# docs = nlp.pipe(texts, n_process=4)
devC_iter = (pre_proc(c) for c in dev_context)
devQ_iter = (pre_proc(q) for q in dev.question)
devC_docs = [doc for doc in nlp.pipe(
    devC_iter, batch_size=64, n_process=args.threads)]
devQ_docs = [doc for doc in nlp.pipe(
    devQ_iter, batch_size=64, n_process=args.threads)]

# tokens
devC_tokens = [[normalize_text(w.text) for w in doc] for doc in devC_docs]
devQ_tokens = [[normalize_text(w.text) for w in doc] for doc in devQ_docs]
devC_unnorm_tokens = [[w.text for w in doc] for doc in devC_docs]
log.info('All tokens for dev are obtained.')

###############################
####加input_ablation
def generate_ablated_input_dev(option, parsed_doc, parsed_query):
    # print('option',option)
    if option == "drop_logical_words":
        logical_words = ["not","n't","all","any","each","every","few","if", "more", "most","no","nor","other","same", "some","than",]
        dropped_doc, mask_dict_c= mask_tokens_dev(parsed_doc, logical_words)
        dropped_query, mask_dict_q= mask_tokens_dev(parsed_query, logical_words)

        #devQ_tokens
        ablated_input = {
            "devC_tokens": dropped_doc,
            "devQ_tokens":parsed_query,
            "drop_dict": mask_dict_c,
        }

    elif option == "drop_causal_words":
        causal_words = ["because", "why", "therefore", "cause", "reason", "as", "since"]
        dropped_doc, mask_dict = mask_tokens_dev(parsed_doc, causal_words)
        ablated_input = {
            "devC_tokens": dropped_doc,
            "devQ_tokens":parsed_query,
            "drop_dict": mask_dict,
        }

    elif option == "mask_pronouns":
        pronouns = """i you he she we they it her his mine my our ours their thy your
        hers herself him himself hisself itself me myself one oneself ours
        ourselves ownself self thee theirs them themselves thou thy us""".split()
        # dropped_doc, mask_dict = mask_tokens(
        #     parsed_doc, pronouns, target_poses=["PRP", "PRP$"]
        # )
        #context文章
        dropped_doc, mask_dict_c= mask_tokens_dev(parsed_doc, pronouns)

        dropped_query, mask_dict_q = mask_tokens_dev(parsed_query, pronouns)

        

        ablated_input = {
            "devC_tokens": dropped_doc,
            "devQ_tokens":dropped_query,
            "drop_dict": mask_dict_c,
            "drop_dict_q":mask_dict_q
        }
    

    elif option == "drop_question_except_interrogatives":
        interrogatives = ["what", "when", "where", "who", "which", "why", "whom", "how"]
        dropped_query, mask_dict_q= mask_tokens_dev(
            parsed_query, interrogatives, complement=True  #????????????
        )
        # print('dropped_query',dropped_query)
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": dropped_query,
            "drop_dict": mask_dict_q,
        }
    elif option == "drop_question_is_interrogatives":
        interrogatives = ["what", "when", "where", "who", "which", "why", "whom", "how"]
        dropped_query, mask_dict_q= mask_tokens_dev(
            parsed_query, interrogatives     
        )
       
        # print('dropped_query',dropped_query)
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": dropped_query,
            "drop_dict": mask_dict_q,
        }


    elif option == "drop_all_question_words":
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": [],
        }
        
    elif option == "original":
        ablated_input = {
            "devC_tokens": parsed_doc,
            "devQ_tokens": parsed_query,
           
        }

    else:
        raise ValueError("Invalid input-ablation option: {}".format(option))

    return ablated_input


#加的[UNK]=drop [MASK]=MASK
def mask_tokens_dev(data, mask_tokens, complement=False, target_poses=None):
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    TOKEN=MASK_TOKEN
    mask_dict = {}
    # token_count = -1
    # print('mask_tokens',mask_tokens)
    # print('data1',data)
    sent_count = -1
    for sent in data:
        sent_count += 1
        token_count = -1
        # print('sent1',sent)
        for token in sent:
            # print('token1',token)
            token_count += 1
            lower_token = token.lower() 
            if target_poses and not token["pos"] in target_poses:
                continue
            if (lower_token in mask_tokens and not complement) or (
                lower_token not in mask_tokens and complement
            ):  
                if isinstance(mask_tokens, dict):
                    # print(2)
                    mask_dict[token_count] = mask_tokens[lower_token]
                    token = mask_tokens[lower_token]
                    # print('token2',token)
                    token=TOKEN
                    # print('token3',token)
                else:
                    # print(3)
                    mask_dict[token_count] = TOKEN   #MASK_TOKEN/UNK_TOKEN"[MASK]"    #"UNK"#
                    token=TOKEN
                    token=token
                    sent[token_count]=token
                    # print('token4',token)
                    # print('token_count',token_count)
                    if True:
                        token = "\sout{" + token + "}"
                    else:
                        token= UNK_TOKEN
        
                #context文章
        
        # print('sent2',sent)
    data[sent_count]=sent

    data=data
    mask_dict=mask_dict
    #文章

    # keys=mask_dict.keys()
    # if len(keys)>0:
    #     for i in range(len(data)):
    #         for j in range(len(list(keys))):
    #             if i==list(keys)[j]:
    #                 data[i]=TOKEN  #MASK_TOKEN/UNK_TOKEN

    # # data=data
    # print('data2',data)
    # # print('mask_dict',mask_dict)
    #文章
    return data, mask_dict

input_ablation_dev=input_ablation_dev
# print(input_ablation)
if input_ablation_dev==0:
    input_ablation="original"
    log.info('0:original-dev')
    ablated_example = generate_ablated_input_dev(
        input_ablation, devC_tokens, devQ_tokens
    )
if input_ablation_dev==1:
    input_ablation="drop_logical_words"
    
    log.info('1:drop_logical_words')
    ablated_example = generate_ablated_input_dev(
        input_ablation, devC_tokens, devQ_tokens
    )
if input_ablation_dev==2:

    input_ablation="mask_pronouns"
    
    log.info('2:mask_pronouns')
    ablated_example = generate_ablated_input_dev(
        input_ablation, devC_tokens, devQ_tokens
    )
if input_ablation_dev==3:
    input_ablation="drop_question_except_interrogatives"  #drop_question_is_interrogatives
    
    log.info('3:drop_question_except_interrogatives')
    ablated_example = generate_ablated_input_dev(
        input_ablation, devC_tokens, devQ_tokens
    )
if input_ablation_dev==4:
    input_ablation="drop_question_is_interrogatives"  #drop_question_is_interrogatives
   
    log.info('4:drop_question_is_interrogatives')
    ablated_example = generate_ablated_input_dev(
        input_ablation, devC_tokens, devQ_tokens
    )

devQ_tokens = ablated_example["devQ_tokens"]
devC_tokens = ablated_example["devC_tokens"]

print('devQ_tokens',len(devQ_tokens))
print('devC_tokens',len(devC_tokens))

log.info('All mask tokens for deving are obtained.')
# print('devQ_tokens',devQ_tokens)
####################input_ablation


dev_context_span = [get_context_span(a, b) for a, b in zip(dev_context, devC_unnorm_tokens)]
log.info('context span for dev is generated.')

ans_st_token_ls, ans_end_token_ls = [], []
for ans_st, ans_end, idx in zip(dev.answer_start, dev.answer_end, dev.context_idx):
    ans_st_token, ans_end_token = find_answer_span(dev_context_span[idx], ans_st, ans_end)
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)

dev['answer_start_token'], dev['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
initial_len = len(dev)
dev.dropna(inplace=True) # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(dev), initial_len))
log.info('answer span for dev is generated.')

# features
devC_tags, devC_ents, devC_features = feature_gen(devC_docs, dev.context_idx, devQ_docs, args.no_match, False)
log.info('features for dev is generated: {}, {}, {}'.format(len(devC_tags), len(devC_ents), len(devC_features)))

def build_dev_vocab(questions, contexts): # most vocabulary comes from tr_vocab
    existing_vocab = set(tr_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in glove_vocab]))
    vocab = tr_vocab + new_vocab
    log.info('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
    return vocab

# vocab
dev_vocab = build_dev_vocab(devQ_tokens, devC_tokens) # tr_vocab is a subset of dev_vocab
devC_ids = token2id(devC_tokens, dev_vocab, unk_id=1)
devQ_ids = token2id(devQ_tokens, dev_vocab, unk_id=1)
devQ_tokens = [["<S>"] + doc + ["</S>"] for doc in devQ_tokens]
devQ_ids = [[2] + qsent + [3] for qsent in devQ_ids]
print(devQ_ids[:10])
# tags
devC_tag_ids = token2id(devC_tags, vocab_tag) # vocab_tag same as training
# entities
devC_ent_ids = token2id(devC_ents, vocab_ent, unk_id=0) # vocab_ent same as training
log.info('vocabulary for dev is built.')

dev_embedding = build_embedding(wv_file, dev_vocab, wv_dim)
# tr_embedding is a submatrix of dev_embedding
log.info('got embedding matrix for dev.')

# don't store row name in csv
#dev.to_csv('Squad2_data/dev.csv', index=False, encoding='utf8')

meta = {
    'vocab': dev_vocab,
    'embedding': dev_embedding.tolist()
}
with open(os.path.join(args.output_dir, 'dev_meta.msgpack'), 'wb') as f:
    msgpack.dump(meta, f)

prev_CID, first_question = -1, []
for i, CID in enumerate(dev.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID

result = {
    'question_ids': devQ_ids,
    'context_ids': devC_ids,
    'context_features': devC_features, # exact match, tf
    'context_tags': devC_tag_ids, # POS tagging
    'context_ents': devC_ent_ids, # Entity recognition
    'context': dev_context,
    'context_span': dev_context_span,
    '1st_question': first_question,
    'question_CID': dev.context_idx.tolist(),
    'question': dev.question.tolist(),
    'answer': dev.answer.tolist(),
    'answer_start': dev.answer_start_token.tolist(),
    'answer_end': dev.answer_end_token.tolist(),
    'answer_choice': dev.answer_choice.tolist(),
    'all_answer': dev.all_answer.tolist(),
    'context_tokenized': devC_tokens,
    'question_tokenized': devQ_tokens
}
with open(os.path.join(args.output_dir, 'dev_data.msgpack'), 'wb') as f:
    msgpack.dump(result, f)

log.info('saved dev to disk.')
