import argparse
import pdb
import sys
import traceback
import logging
import json


def main(args):
    with open(args.input_path) as f:
        data = json.load(f)

    for sample in data['data']:
        for paragraph in sample['paragraphs']:
            qas = sorted(paragraph['qas'],
                         key=lambda s: -len(s['orig_answer']['text']))
            yorn=[] 
            YYORN=0          
            for qa in qas:
                yorn=qa['followup']
                if yorn=='n':
                    YYORN=yorn
                
            print('yorn',YYORN)
            if YYORN==0:
                paragraph['context']=[]
                paragraph['qas']=[]
                sample['paragraphs']=[]
                sample['section_title']=[]
                sample['background']=[]
                sample['title']=[]

                



    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=' ', ensure_ascii=False)





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
