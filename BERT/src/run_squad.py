# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# run_squad.py  : "Run BERT on SQuAD.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from regex import A, P
from tqdm import tqdm, trange
import nltk
import nltk.tokenize as tk
# nltk.download('punkt')

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.paragraph_text=paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += ", paragraph_text: %s" % (self.paragraph_text)
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                if is_training:
                    if len(qa["answers"]) < 1:
                        raise ValueError(
                            "For training, each question should have at least 1 answer.")
                    answer = qa["orig_answer"]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                   
                    # Only add answer
                    # try:
                    #     start_position = char_to_word_offset[answer_offset]
                    #     print('start_position',start_position)
                    # except:
                    #     continue
                    # try:
                    #     end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    #     print('end_position',end_position)
                    # except:
                    #     continue
                    # # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                        

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    paragraph_text=paragraph_text)
                examples.append(example)
                
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length,is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    #

    
    #
    features = []
    # kka=0
    for (example_index, example) in enumerate(examples):
        print('example_index',example_index)
        # kka=kka+1
        # print(kka)
        # if kka==3:
        #     break
        query_tokens = tokenizer.tokenize(example.question_text)
        # print('example1',example)#[ qas_id=qas_id,question_text=question_text,doc_tokens=doc_tokens,]
        # print('example.doc_tokens',example.paragraph_text)
        # print('example.qid',example.qas_id)
        sentence = tk.sent_tokenize(example.paragraph_text)
        # print('sentence',sentence)
        ss_tokens=[]
        for ss in sentence:
            s_tokens=tokenizer.tokenize(ss)
            ss_tokens.append(s_tokens)
        #     print('ss',ss)
        #     print('ss_tokens0',ss_tokens)
        # print('ss_tokens1',ss_tokens)
        # for i in range(len(ss_tokens)):
        #     print('ss_tokens[i]',ss_tokens[i])
        # print('_'*72)
        
        # print('query_tokens',query_tokens)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        # print('query_tokens',query_tokens)
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []#0->all_doc_tokens
        for (i, token) in enumerate(example.doc_tokens):
            # print('example.doc_tokens',example.doc_tokens)
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token) #1->all_doc_tokens
        # print('all_doc_tokens',all_doc_tokens)
    ################################################
        input_ablation=5
        input_ablation=input_ablation
        # print('input_ablation',input_ablation)
        if input_ablation==0: #original
            print('input_ablation',input_ablation)
            input_ablation="original"  #drop_question_is_interrogatives
            ablated_example = generate_ablated_input(
                input_ablation, all_doc_tokens, query_tokens
            )
            print('ablated_example',ablated_example)
        if input_ablation==1:
            input_ablation="drop_logical_words"
            # print('input_ablation',input_ablation)
            ablated_example = generate_ablated_input(
                input_ablation, all_doc_tokens, query_tokens
            )
        if input_ablation==2:
            print('input_ablation',input_ablation)
            input_ablation="mask_pronouns"
            ablated_example = generate_ablated_input(
                input_ablation, all_doc_tokens, query_tokens
            )
        if input_ablation==3:
            print('input_ablation',input_ablation)
            input_ablation="drop_question_except_interrogatives"  #drop_question_is_interrogatives
            ablated_example = generate_ablated_input(
                input_ablation, all_doc_tokens, query_tokens
            )
        if input_ablation==4:
            # print('input_ablation',input_ablation)
            input_ablation="drop_question_is_interrogatives"  #drop_question_is_interrogatives
            ablated_example = generate_ablated_input(
                input_ablation, all_doc_tokens, query_tokens
            )

        if input_ablation==5:
            # ablated_input = {
            # "doc_tokens": dropped_doc,
            # "query_tokens": parsed_query,
           
            # "ablation_info": ab_info,
            # "drop_dict": mask_dict,
            #  }
            # print('input_ablation',input_ablation)
            input_ablation="drop_except_most_similar_sentences"  #drop_question_is_interrogatives
            # ablated_example = generate_ablated_input(
            #     input_ablation, all_doc_tokens, query_tokens
            # )
            #query_tokens = tokenizer.tokenize(example.question_text)

            ablated_example = generate_ablated_input_sentence(
                input_ablation, ss_tokens, query_tokens
            )
        
        if input_ablation==6:
            print('input_ablation',input_ablation)
            input_ablation="drop_question_overlaps"
            ablated_example = generate_ablated_input(
                input_ablation, all_doc_tokens, query_tokens
            )
        #elif option == "drop_question_overlaps":
        #elif option == "drop_except_most_similar_sentences"
        #8.11
        query_tokens = ablated_example["query_tokens"]
        all_doc_tokens = ablated_example["doc_tokens"]#3->all_doc_tokens
        # print('query_tokens',query_tokens)
        # print('all_doc_tokens',all_doc_tokens)
        # #8.11
        # query_tokens = query_tokens
        # all_doc_tokens = all_doc_tokens#3->all_doc_tokens
        ####################################################
        mask_dict=ablated_example["drop_dict"]
        
        keys=mask_dict.keys()
        ###################
        # query_tokens = ablated_example["query_tokens"]
        # all_doc_tokens = ablated_example["doc_tokens"]
        # mask_dict=ablated_example["drop_dict"]
    
        # keys=mask_dict.keys()
        
        # #####################
        # mask_dict_q=ablated_example["drop_dict_q"]
        # q_keys=mask_dict_q.keys()
        # content_keys=mask_dict.keys()
        # ablated_input = {
        #     "doc_tokens": dropped_doc,
        #     "query_tokens":dropped_query,
        #     "drop_dict": mask_dict,
        #     "drop_dict_q":mask_dict_q
        # }
        # print('query_tokens',query_tokens)
        # print('all_doc_tokens',all_doc_tokens)
        # print('mask_dict',mask_dict)
        # print('mask_dict_q',mask_dict_q)




        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)   #加上query_tokens
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])  #加上all_doc_tokens
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            ##############################
            if input_ablation==1 or input_ablation==2:
                for i in range(len(input_mask)):
                    for j in range(len(list(keys))):
                        if i==list(keys)[j]:
                            i=i+len(query_tokens)+2#问题长度+cls+sep
                            input_mask[i]=0
            else:
                input_mask=input_mask
            input_mask=input_mask
            ############################
            # print('tokens',tokens)
            # print('input_ids',input_ids)
            # print('input_mask',input_mask)
                      ###################
            # input_ablation=2  #这个根据改
            # print('input_ablation',input_ablation)
            # if input_ablation==1 or 2 : 
            #     for i in range(len(input_mask)):
            #         for j in range(len(list(keys))):
            #             if i==list(keys)[j]:
            #                 i=i+len(query_tokens)+2#问题长度+cls+sep
            #                 input_mask[i]=0
            #     # for i in range(len(input_mask)):
            #     #     for j in range(len(list(keys))):
            #     #         if i==list(keys_q)[j]:
            #     #             i=i+1     #cls
            #     #             input_mask[i]=0
            # else:
            #     input_mask=input_mask
            # input_mask=input_mask
            # keys=keys



            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (example.start_position < doc_start or
                        example.end_position < doc_start or
                        example.start_position > doc_end or example.end_position > doc_end):
                    continue

                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))
            # a=len(input_ids)
            # b=len(input_mask)
            # c=list(content_keys)

            ########
            # query_tokens = ablated_example["query_tokens"]
            # all_doc_tokens = ablated_example["doc_tokens"]
            # mask_dict=ablated_example["drop_dict"]
        
            # keys=mask_dict.keys()
         

            
            #修改input_mask
                    
            # for i in range(len(input_mask)):
            #     for j in range(len(list(keys))):
            #         if i==list(keys)[j]:
            #             input_mask[i]=0

        #    ###################
        #     # input_ablation=2  #这个根据改
        #     print('input_ablation',input_ablation)
        #     if input_ablation==1 or 2 : 
        #         for i in range(len(input_mask)):
        #             for j in range(len(list(keys))):
        #                 if i==list(keys)[j]:
        #                     i=i+len(query_tokens)+2#问题长度+cls+sep
        #                     input_mask[i]=0
        #         # for i in range(len(input_mask)):
        #         #     for j in range(len(list(keys))):
        #         #         if i==list(keys_q)[j]:
        #         #             i=i+1     #cls
        #         #             input_mask[i]=0
        #     else:
        #         input_mask=input_mask
        #     input_mask=input_mask
        #     keys=keys

            ############
            # print('tokens',tokens)
            # print('input_ids',input_ids)
            # print('input_mask',input_mask)
            # print('keys',keys)
            # # # input_mask=input_mask
            # print('tokens',tokens)
            # print('input_ids',input_ids)
            # print('input_mask',input_mask)

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position))
            unique_id += 1
            # print(features)
        
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index



RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging):
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

#加的[UNK]=drop [MASK]=MASK
def mask_tokens(data, mask_tokens, complement=False, target_poses=None):
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    TOKEN=MASK_TOKEN
    mask_dict = {}
    token_count = -1
    for token in data:
        token_count += 1
        lower_token = token.lower() 
        if target_poses and not token["pos"] in target_poses:
            continue
        if (lower_token in mask_tokens and not complement) or (
            lower_token not in mask_tokens and complement
        ):  
            if isinstance(mask_tokens, dict):
                mask_dict[token_count] = mask_tokens[lower_token]
                token = mask_tokens[lower_token]
            else:
                mask_dict[token_count] = TOKEN   #MASK_TOKEN/UNK_TOKEN"[MASK]"    #"UNK"#
                if True:
                    token = "\sout{" + token + "}"
                else:
                    token= UNK_TOKEN
            #context文章
    
    data=data
    mask_dict=mask_dict
    keys=mask_dict.keys()
    if len(keys)>0:
        for i in range(len(data)):
            for j in range(len(list(keys))):
                if i==list(keys)[j]:
                    data[i]=TOKEN  #MASK_TOKEN/UNK_TOKEN

    data=data
    
    #文章
    return data, mask_dict

    #  dropped_doc, ab_info, mask_dict = drop_except_most_similar_sentences(
    #         parsed_doc, query_tokens
    #     )



def drop_except_most_similar_sentences_1(ss_tokens, query_tokens):
    print('data1',ss_tokens)
    print('query_tokens',query_tokens)
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    TOKEN=UNK_TOKEN
    overlap_count = []
    for sent in ss_tokens:
        count = 0
        #query_tokens = tokenizer.tokenize(example.question_text)
        #sent_tokenize
        #sentence = tk.sent_tokenize(example.paragraph_text)
        #sent_token= tk.word_tokenize(sent)
        
        print('sent',sent)
        for token in sent_token:
            # print('token',token)
            lower_token = token.lower()
            if lower_token in query_tokens:
                # print('lower_token',lower_token)
                # print('query_tokens',query_tokens)
                count += 1
        overlap_count.append(count)
        print('overlap_count',overlap_count)

    most_similar_indices = [
        si for si, c in enumerate(overlap_count) if c == max(overlap_count)
    ]
    print('most_similar_indices',most_similar_indices)
    mask_dict = {}
    token_counter = -1
    for si, sent in enumerate(data):
        # print('si',si)
        # print('sent',sent)
        sent_token= tk.word_tokenize(sent)
        for token in sent_token:
            # print('token',token)
            token_counter += 1
            if si not in most_similar_indices:

                token = UNK_TOKEN
                token=token
                # print('token',token)
                mask_dict[token_counter] = UNK_TOKEN
            # print('sent_token',sent_token)
    ################
    #def drop_except_most_similar_sentences(data,all_doc_tokens, query_tokens):
   
    data=all_doc_tokens
    print('data11',data)
    mask_dict=mask_dict
    keys=mask_dict.keys()
    if len(keys)>0:
        for i in range(len(data)):
            for j in range(len(list(keys))):
                if i==list(keys)[j]:
                    data[i]=TOKEN  #MASK_TOKEN/UNK_TOKEN

    data=data
    print('data12',data)

    # data=data
    # mask_dict=mask_dict
    # print('data12',data)
    # print('mask_dict',mask_dict)
    # keys=mask_dict.keys()
    # print('keys',keys)

    # if len(keys)>0:
    #     print(1)
    #     for i in range(len(data)):
    #         print(2)
    #         for j in range(len(list(keys))):
    #             print(3)
    #             if i==list(keys)[j]:
    #                 print(4)
    #                 print(' data[i]1', data[i])
    #                 data[i]=TOKEN  #MASK_TOKEN/UNK_TOKEN
    #                 print(' data[i]2', data[i])

    # data=data
    # print('data2',data)
    # print('mask_dict',mask_dict)

    l=['egon','aa']

    l[3]
    return data, " ".join([str(i) for i in most_similar_indices]), mask_dict


def drop_except_most_similar_sentences(ss_tokens, query_tokens):
    # print('ss_tokens',ss_tokens)
    # print('query_tokens',query_tokens)
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN="[MASK]"
    TOKEN=UNK_TOKEN
    overlap_count = []
    for sent in ss_tokens:
        count = 0
        # print('sent',sent)
        for token in sent:
            lower_token = token.lower()
            if lower_token in query_tokens:
                # print('lower_token',lower_token)
                # print('query_tokens',query_tokens)
                count += 1
        overlap_count.append(count)
    most_similar_indices=[]
    ####2个句子
    # print('overlap_count',overlap_count)
    # print('most_similar_indices0',most_similar_indices)
    # for si, c in enumerate(overlap_count) :
    #     if c == max(overlap_count):
    #         most_similar_indices.append(si)
    #     #arr.splice(index, 1)  delete
    #     #overlap_count2=overlap_count.splice(index,si)
    #         overlap_count2=overlap_count.remove(c)
    # print('most_similar_indices1',most_similar_indices)
    # overlap_count2=overlap_count2
    # print('overlap_count0',overlap_count2)
    # for si, c in enumerate(overlap_count2) :
    #     if c==max(overlap_count2):
    #         most_similar_indices.append(si)
    # ###############2个句子
    most_similar_indices = [
        si for si, c in enumerate(overlap_count) if c == max(overlap_count)
    ]
    # ####################
    # print('most_similar_indices',most_similar_indices)
    mask_dict = {}
    token_counter = -1
    for si, sent in enumerate(ss_tokens):
        # print('si',si)
        # print('sent',sent)
    
        for token in sent:
            # print('token',token)
            token_counter += 1
            if si not in most_similar_indices:

                token = UNK_TOKEN
                token=token
                # print('token',token)
                mask_dict[token_counter] = UNK_TOKEN
            # print('sent_token',sent_token)
        
    ################
    #def drop_except_most_similar_sentences(data,all_doc_tokens, query_tokens):
    senten_token=[]
    data=ss_tokens
    for i in range(len(data)):
        senten=data[i]
        for j in range(len(senten)):
            senten_token.append(senten[j])
    # print('senten_token',senten_token)
    # print('data11',data)
    data=senten_token
    # print('data11',data)
    mask_dict=mask_dict
    keys=mask_dict.keys()
    # print('mask_dict',mask_dict)
    keys=mask_dict.keys()
    # print('keys',keys)
    if len(keys)>0:
        for i in range(len(data)):
            for j in range(len(list(keys))):
                if i==list(keys)[j]:
                    data[i]=TOKEN  #MASK_TOKEN/UNK_TOKEN

    data=data
    # print('data12',data)

    

    # l=['egon','aa']

    # l[3]
    #########3
    return data, " ".join([str(i) for i in most_similar_indices]), mask_dict



#   input_ablation=2
#         if input_ablation==1:
#             input_ablation="drop_logical_words"
#             ablated_example = generate_ablated_input(
#                 input_ablation, all_doc_tokens, query_tokens
#             )
#         if input_ablation==2:
#             input_ablation="mask_pronouns"
#             ablated_example = generate_ablated_input(
#                 input_ablation, all_doc_tokens, query_tokens
#             )
#         if input_ablation==3:
#             input_ablation="drop_question_except_interrogatives"
#             ablated_example = generate_ablated_input(
#                 input_ablation, all_doc_tokens, query_tokens
#             )
# ablated_example = generate_ablated_input(input_ablation, all_doc_tokens, query_tokens)

def generate_ablated_input(option, parsed_doc, parsed_query):

    if option == "drop_logical_words":
        logical_words = ["not","n't","all","any","each","every","few","if", "more", "most","no","nor","other","same", "some","than",]
        dropped_doc, mask_dict_c= mask_tokens(parsed_doc, logical_words)
        dropped_query, mask_dict_q= mask_tokens(parsed_query, logical_words)

        #query_tokens
        ablated_input = {
            "doc_tokens": dropped_doc,
            "query_tokens":parsed_query,
            "drop_dict": mask_dict_c,
        }

    elif option == "drop_causal_words":
        causal_words = ["because", "why", "therefore", "cause", "reason", "as", "since"]
        dropped_doc, mask_dict = mask_tokens(parsed_doc, causal_words)
        ablated_input = {
            "doc_tokens": dropped_doc,
            "query_tokens":parsed_query,
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
        #问句
        #################
        # keys_q=mask_dict_q.keys()
        # keys_c=mask_dict_c.keys()
        # # print('keys_q',keys_q)
        # # print('keys_c',keys_c)
        # #############33
        # a=len(parsed_query)

        # b=list(keys_c)
        # # c=b+a
        # c=[1,2,3,4,5,6]
        # for i in range(len(c)):
        #     c[i]=c[i]+5
        # print(c)
        # print('len(parsed_query)',a)
        # print('list(keys_c)',b)
        # # np.array(a)
        # a=np.array([list(keys_c)])
        # print(a)
        # for i in range(len(list(keys_c))):
            
        #     # list(keys_c)[i]=int(a)+8
        #     a[i]=a[i]+8
            
           
        #     print(a[i])
           
        # print('a',a)

        # print(c)
                    # input_ablation=2  #这个根据改
            # if input_ablation ==1 or 2 : 
            #     for i in range(len(input_mask)):
            #         for j in range(len(list(keys))):
            #             if i==list(keys)[j]:
            #                 i=i+len(parsed_query)
            #                 input_mask[i]=0
            # else:
             # for i in range(len(input_mask)):
            #     for j in range(len(list(keys))):
            #         if i==list(keys)[j]:
            #             input_mask[i]=0

           
        

        ablated_input = {
            "doc_tokens": dropped_doc,
            "query_tokens":dropped_query,
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
            "doc_tokens": parsed_doc,
            "query_tokens": dropped_query,
            "drop_dict": mask_dict_q,
        }
    elif option == "drop_question_is_interrogatives":
        interrogatives = ["what", "when", "where", "who", "which", "why", "whom", "how"]
        dropped_query, mask_dict_q= mask_tokens(
            parsed_query, interrogatives     
        )
       
        # print('dropped_query',dropped_query)
        ablated_input = {
            "doc_tokens": parsed_doc,
            "query_tokens": dropped_query,
            "drop_dict": mask_dict_q,
        }


    elif option == "drop_all_question_words":
        ablated_input = {
            "doc_tokens": parsed_doc,
            "query_tokens": [],
        }
    elif option == "drop_except_most_similar_sentences":
        query_tokens = [
            x.lower() for x in parsed_query
        ]

        dropped_doc, ab_info, mask_dict = drop_except_most_similar_sentences(
            parsed_doc, query_tokens
        )
        ablated_input = {
            "doc_tokens": dropped_doc,
            "query_tokens": parsed_query,
           
            "ablation_info": ab_info,
            "drop_dict": mask_dict,
        }


    elif option == "drop_causal_words":
        causal_words = ["because", "why", "therefore", "cause", "reason", "as", "since"]
        dropped_doc, mask_dict = mask_tokens(parsed_doc, causal_words)
        ablated_input = {
            "doc_tokens": dropped_doc,
            "query_tokens":parsed_query,
            "drop_dict": mask_dict,
        }
    #def generate_ablated_input(option, parsed_doc, parsed_query):
    elif option == "drop_question_overlaps":
        query_tokens = parsed_query
        masked_doc, mask_dict = mask_tokens(parsed_doc, query_tokens)
        ablated_input = {
            "doc_tokens": masked_doc,
            "query_tokens":parsed_query,
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "original":
        # ablated_input = {
        #     "devC_tokens": parsed_doc,
        #     "devQ_tokens": parsed_query,
           
        # }
        ablated_input = {
            "doc_tokens": parsed_doc,
            "query_tokens":parsed_query,
            "drop_dict":  None,
        }
    else:
        raise ValueError("Invalid input-ablation option: {}".format(option))

    return ablated_input

def generate_ablated_input_sentence(option, ss_tokens,parsed_query):
    if option == "drop_except_most_similar_sentences":
        query_tokens = [
            x.lower() for x in parsed_query
        ]
        #    return data, " ".join([str(i) for i in most_similar_indices]), mask_dict
        dropped_doc, ab_info, mask_dict = drop_except_most_similar_sentences(
            ss_tokens,query_tokens
        )
        ablated_input = {
            "doc_tokens": dropped_doc,
            "query_tokens": parsed_query,
           
            "ablation_info": ab_info,
            "drop_dict": mask_dict,
        }

    elif option == "original":
        # ablated_input = {
        #     "devC_tokens": parsed_doc,
        #     "devQ_tokens": parsed_query,
           
        # }
        ablated_input = {
            "doc_tokens": [],
            "query_tokens":parsed_query,
            "drop_dict":  None,
        }
    else:
        raise ValueError("Invalid input-ablation option: {}".format(option))

    return ablated_input

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased-coqa"#"Bert pre-trained model selected in the list: bert-base-uncased"
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.") #加了参数
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    #######

    parser.add_argument("--input_ablation", default=1, type=int, help="")



    #####
    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")  #不知道是干嘛的改成1吧以前是3
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = read_squad_examples(
            input_file=args.train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForQuestionAnswering.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
            args.bert_model, str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, args.verbose_logging)


if __name__ == "__main__":
    main()
