import argparse
from ast import arg
import pdb
import sys
import traceback
import logging
import json
import spacy
import jieba.analyse
import random


def main(args):
    with open(args.input_path) as f:
        data = json.load(f)

    nlp = spacy.load(
        'en_core_web_sm',
        disable=['tokenizer', 'tagger', 'ner', 'parser', 'textcat']
    )
    nlp.add_pipe('sentencizer')
    # sentencizer=nlp.create_pipe('sentencizer')
    # nlp.add_pipe(sentencizer)
    for sample in data['data']:
        if args.attack == 'repeat':
            answers = sorted(
                sample['answers'],
                key=lambda a: -len(a['span_text'])
            )
            print('answers',answers)
            print('sampleanswers',sample['answers'])
            for answer in answers:
                answer_satrt = answer['span_start']
                text = answer['span_text']
                if text == 'unknown':
                    continue
                sample['story'] = repeat_attack(
                    sample['story'],
                    answer_satrt,
                    text,
                    args.times
                )
                for oans in sample['answers']:
                    if oans['turn_id'] != answer['turn_id'] and \
                       oans['span_start'] >= answer_satrt:
                        oans['span_start'] += (len(text) + 1) * args.times
                        oans['span_end'] += (len(text) + 1) * args.times

        elif args.attack == 'random':
            sample['story'] = random_attack(
                nlp, sample['story'], len(sample['answers'])
            )
        elif args.attack=='qnn':
            print('sample[questions]',sample['questions'])
            
            questions = sample['questions']

            #  questions = sorted(
            #     sample['questions'],
            #     key=lambda a: -len(a['span_text'])
            # )


            for question in questions:
                
                question_text = question['input_text']
             
                parsed_query=question_text



                print('question_text',question_text)

                word_list= jieba.cut(parsed_query)
                word_list=list(word_list)
                
                # print('word_list',list(word_list))
                # print('parsed_query',parsed_query)\
                no_words = ["am","is","are","was","were","do","did","does","have", "has"]
                # wh_n_words=["what","who",]
                
                dropped_query, mask_dict_q= mask_tokens_dev(word_list, no_words)
                dropped_query1=listToString(dropped_query)

                question_text=dropped_query1
                question['input_text']=dropped_query1
                # print('qaquestion',qa['question'])
        # for answer in answers:
        #     if answer['span_text'] != 'unknown' \
        #        and answer['span_text'] != (
        #         sample['story'][answer['span_start']:answer['span_end']]
        #     ):
        #         logging.warn('Mismatch!')
        #         breakpoint()

    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=' ', ensure_ascii=False)


def repeat_attack(context, start_index, text, times=1):
    assert context[start_index:start_index + len(text)] == text
    context = (
        context[:start_index]
        + ' '.join([text] * times)
        + ' ' + context[start_index:])
    return context


def random_attack(nlp, context, n):
    sentences = list(nlp(context).sents)
    random_indices = random.choices(list(range(len(sentences))), k=n)
    attacked = []
    for i, sentence in enumerate(sentences):
        attacked.append(sentence.text)
        if i in random_indices:
            attacked.append(sentence.text)

    return ' '.join(attacked)

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
    parser.add_argument('--attack', type=str, default='qnn')#repeat 
    parser.add_argument('--times', type=int, default=1)
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
