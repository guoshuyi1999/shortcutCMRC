import argparse
import pdb
import sys
from tkinter.messagebox import QUESTION
import traceback
import logging
import json
from collections import Counter
import jieba.analyse
import re
import time

def main(args):
    with open(args.input_path) as f:
        data = json.load(f)
    # word_list= jieba.cut(data)
    for sample in data['data']:
        for paragraph in sample['paragraphs']:
            qas = sorted(paragraph['qas'],
                         key=lambda s: -len(s['orig_answer']['text']))
            for qa in qas:
                parsed_query=qa['question']
                word_list= jieba.cut(parsed_query)
                print('word_list',word_list)
                no_words = ["am","is","are","was","were","did","does","have", "has"]
                # wh_n_words=["what","who",]
                
                dropped_query, mask_dict_q= mask_tokens_dev(parsed_query, no_words)
                dropped_query=listToString(dropped_query)
                qa['question']=dropped_query
                print('qaquestion'),qa['question']

    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=' ', ensure_ascii=False)

# #加的[UNK]=drop [MASK]=MASK
# def mask_tokens_dev(data, mask_tokens, complement=False, target_poses=None):
#     UNK_TOKEN = "[UNK]"
#     MASK_TOKEN="[MASK]"
#     NO_TOKEN="not"
#     TOKEN=MASK_TOKEN
#     mask_dict = {}

#     sent_count = -1
#     for token in data:
#         # print('data',data)
#         sent_count += 1
#         # token_count = -1
#         # # print('sent1',sent)
#         # # for token in sent:
#         # # print('token1',token)
#         # token_count += 1
#         lower_token = token.lower() 
#         if target_poses and not token["pos"] in target_poses:
#             continue
#         if (lower_token in mask_tokens and not complement) or (
#             lower_token not in mask_tokens and complement
#         ):  
#             if isinstance(mask_tokens, dict):
#                 print(2)
#                 print('data',data)
#                 # mask_dict[token_count] = mask_tokens[lower_token]
#                 token = mask_tokens[lower_token]
#                 print('token2',token)
#                 token=token+NO_TOKEN
#                 print('token3',token)
#             else:
#                 print(3)
#                 print('data',data)
#                 # mask_dict[token_count] = TOKEN   #MASK_TOKEN/UNK_TOKEN"[MASK]"    #"UNK"#
#                 # token=TOKEN
#                 # k_TOKEN=[]
#                 token=token+" "+NO_TOKEN
#                 # sent[token_count]=token
#                 # print('token4',token)
#                 # print('sent_count',sent_count)
#                 data[sent_count]=token
#                 print('data0',data)
#                 if True:
#                     token = "\sout{" + token + "}"
#                 else:
#                     token= UNK_TOKEN
    
#             #context文章
        
#     #     print('sent2',sent)
#     # data[sent_count]=sent

#     data=data
#     mask_dict=mask_dict
#     #文章
#     print('data1',data)
#     # keys=mask_dict.keys()
#     # if len(keys)>0:
#     #     for i in range(len(data)):
#     #         for j in range(len(list(keys))):
#     #             if i==list(keys)[j]:
#     #                 data[i]=TOKEN  #MASK_TOKEN/UNK_TOKEN

#     # # data=data
#     # print('data2',data)
#     # # print('mask_dict',mask_dict)
#     #文章
#     return data, mask_dict
# #加的[UNK]=drop [MASK]=MASK
# def repeat_attack(context, start_index, text):
#     assert context[start_index:start_index + len(text)] == text
#     context = context[:start_index] + text + ' ' + context[start_index:]
#     return context
# def que_n_quac():
#加的[UNK]=drop [MASK]=MASK
def mask_tokens_dev(data, mask_tokens, complement=False, target_poses=None):
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    NO_TOKEN="not"
    TOKEN=MASK_TOKEN
    mask_dict = {}

    sent_count = -1
    for token in data:
        print('data',data)
        sent_count += 1
        token_count = -1
        # print('sent1',sent)
        # for token in sent:
        print('token1',token)
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
                print('token2',token)
                token=token+NO_TOKEN
                print('token3',token)
            else:
                print(3)
                mask_dict[token_count] = TOKEN   #MASK_TOKEN/UNK_TOKEN"[MASK]"    #"UNK"#
                token=TOKEN
                token=token+NO_TOKEN
                # sent[token_count]=token
                print('token4',token)
                print('token_count',token_count)
                if True:
                    token = "\sout{" + token + "}"
                else:
                    token= UNK_TOKEN
    
            #context文章
        
    #     print('sent2',sent)
    # data[sent_count]=sent

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

# Function to convert  
def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))
        
        




if __name__ == "__main__":
    main()


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('input_path', type=str,
                        help='')
    parser.add_argument('output_path', type=str,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
