import argparse
import pdb
import sys
import traceback
import logging
import json
from collections import Counter
import jieba.analyse
import re
import time
import sys
import os

def main(args):
    with open(args.input_path) as f:
        data = json.load(f)

    for sample in data['data']:
        for paragraph in sample['paragraphs']:
            qas = sorted(paragraph['qas'],
                         key=lambda s: -len(s['orig_answer']['text']))
            
            for qa in qas:
                parsed_query=qa['question']

                word_list= jieba.cut(parsed_query)
                word_list=list(word_list)
                
                # print('word_list',list(word_list))
                # print('parsed_query',parsed_query)\
                no_words = ["am","is","are","was","were","do","did","does","have", "has"]
                # wh_n_words=["what","who",]
                
                dropped_query, mask_dict_q= mask_tokens_dev(word_list, no_words)
                dropped_query1=listToString(dropped_query)
                qa['question']=dropped_query1
                # print('qaquestion',qa['question'])
                
                
    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=' ', ensure_ascii=False)


# Function to convert  
def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))
        
    
def mask_tokens_dev(data, mask_tokens, complement=False, target_poses=None):
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    NO_TOKEN="not"
    NERVER_TOKEN="never"
    TOKEN=MASK_TOKEN
    mask_dict = {}
    j=0
    jj=0
    print('data0',data)
    sent_count = -1
    for token in data:
        # k=k
        sent_count += 1
        lower_token = token.lower() 
        if target_poses and not token["pos"] in target_poses:
            continue
        if (lower_token in mask_tokens and not complement) or (lower_token not in mask_tokens and complement):  
            if isinstance(mask_tokens, dict):
                print(2)
                print('data',data)
                token = mask_tokens[lower_token]
                print('token2',token)
                token=token+NO_TOKEN
                print('token3',token)
            else:
                print(3)
                print('data',data)
                token=token+" "+NO_TOKEN
                data[sent_count]=token
                j=j+1
                # print('data0',data)
                # print('j',j)
                print('data1',data)
                if True:
                    token = "\sout{" + token + "}"
                else:
                    token= UNK_TOKEN

            break
 
    data=data

    mask_dict=mask_dict
    #文章
    print('data10',data)
    jj=jj+1

    mask1_tokens=["what","why","how","where"]
    NERVER_TOKEN="never"  
    if j!=jj:
        sent1_count = -1
        for token1 in data:
            sent1_count += 1
            lower_token = token1.lower() 
            if target_poses and not token1["pos"] in target_poses:
                continue
            if (lower_token in mask1_tokens and not complement) or (lower_token not in mask1_tokens and complement):  
                if isinstance(mask1_tokens, dict):
                    print(2)
                    print('data',data)
                    token1 = mask1_tokens[lower_token]
                    print('token2',token)
                    token1=token1+NERVER_TOKEN
                    print('token3',token)
                else:
                    # print(3)
                    # print('data',data)
                    token1=token1+" "+NERVER_TOKEN
                    data[sent1_count]=token1
                    print('data2',data)

                    if True:
                        token1 = "\sout{" + token1 + "}"
                    else:
                        token= UNK_TOKEN
                break

    data=data

    print('data20',data)
    return data, mask_dict

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
