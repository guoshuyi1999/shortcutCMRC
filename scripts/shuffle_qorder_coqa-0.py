import argparse
import pdb
import sys
import traceback
import logging
import json
import spacy
import random

    # nlp = spacy.load(
    #     'en_core_web_sm',
    #     disable=['tokenizer', 'tagger', 'ner', 'parser', 'textcat']
    # )
    # nlp.add_pipe('sentencizer')

def main(args):
    with open(args.input_path) as f:
        coqa = json.load(f)

    nlp = spacy.load(
        'en_core_web_sm',
        disable=['tokenizer', 'tagger', 'ner', 'parser', 'textcat']
    )
    nlp2 = spacy.load(
        'en_core_web_sm',
        disable=['tokenizer', 'tagger', 'ner', 'parser', 'textcat']
    )
    #nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # nlp.add_pipe(('sentencizer'))
    # nlp2.add_pipe(('Tokenization'))
    
    
    # # 2. 分词 (Tokenization)
    # print([w.text for w in doc])
    # """
    # ['小米', '董事长', '叶凡', '决定', '投资', '华为', '。', '在', '2002年', '，', '他', '还', '创作', '了', '<遮天>', '。']
    # """


    # for sample in coqa['data']:
    #     print('sample0',sample)
    #     sents = list(nlp(sample['story']).sents)
    #     print('sents0',sents)
    #     random.shuffle(sents)
    #     print('sents1',sents)
    #     sample['story'] = ' '.join([sent.text for sent in sents])
    #     print('sample1',sample)
    # c=coqa['questions'].input_text
    # print(c)


    for sample in coqa['data']:
        questions=sample['questions']
        print('questions0',questions)
        for question in questions:
            print(question)
            print(question['input_text'])
            doc=nlp(question['input_text'])
            words=[w.text for w in doc]
            lens=len(words)
            words=words[:lens-1]  
            random.shuffle(words)
            question['input_text'] = ' '.join([word for word in words])+' '+'?'
            print('question[input_text]',question['input_text'])
            print('question[input_text]',type(question['input_text']))
            print('question',question)
        print('questions1',questions)


    with open(args.output_path, 'w') as f:
        json.dump(coqa, f, indent=' ')


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('input_path', type=str,
                        help='')
    parser.add_argument('output_path', type=str)
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
