import argparse
import json
import os
from pyexpat import model
import random
import sys
import time
import traceback
from distutils.log import info
from sched import scheduler
from statistics import mode
from turtle import forward

import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from config import *
from constant_bart import *
from model_saver import *
from seed import *
from knowledge import KnowledgeGraph

from model.modeling_longformer import * 
from model.modeling_roberta import *

from transformers import GPT2Tokenizer, AutoConfig, EncoderDecoderConfig, GPT2LMHeadModel, EncoderDecoderModel, LongformerTokenizer
from transformers import AdamW, get_scheduler
from transformers import TrainingArguments, Trainer


DECODER_SPECIAL_TOKENS  = {"bos_token": "<s>",
                   "eos_token": "</s>",
                   "unk_token": "<unk>",                    
                   "pad_token": "<pad>",
                   "sep_token": "</s>"} 

def parsers(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/roberta/", type=str,
                        help="Path of the output model.")
    parser.add_argument("--train_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_train.tsv",type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_val.tsv",type=str,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/datasets/medical_test.tsv",type=str,
                        help="Path of the testset.")
    parser.add_argument("--log_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/logs",type=str,
                        help="Path of the testset.")
    parser.add_argument("--last_logging", default=None,type=str,
                        help="Path of the testset.")
    parser.add_argument("--encoder_model_path", default=None,type=str,
                        help="Path of the testset.")

    # Model options.
    parser.add_argument("--encoder_model_name", type=str, default="allenai/longformer-base-4096",
                        help="The name of a pretrained model")
    parser.add_argument("--decoder_model_name", type=str, default="gpt2",
                        help="The name of a pretrained model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size.")
    parser.add_argument("--seq_length_encoder", type=int, default=512,
                        help="Sequence length of encoder.")
    parser.add_argument("--seq_length_decoder", type=int, default=1024,
                        help="Sequence length of decoder.")
    parser.add_argument("--max_length", type=int, default = 256, 
                        help= "max length.")
    parser.add_argument("--min_length", type = int, default=50, 
                        help="Min length.") 

    parser.add_argument("--learning_rate", type=float, default=0.00003,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="weight_decay.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="max_grad_norm.")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="save_steps value.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-06,
                        help="Warm up value.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="save_steps value.")
    parser.add_argument("--logging_steps", type=int, default=500,
                        help="logging_steps value.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warm up value.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warm up value.")

    

    # Training options.
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_path", default="/content/k-distilroberta-gpt2/brain/kgs/Medical.spo",type=str, help="KG name or path")

    args = parser.parse_args()
    return args

class Medical_Dataset(Dataset): 
    def __init__(self, dataset_path, knowledge,encoder_tokenizer, decoder_tokenizer,args, ij_kg = False): 
        self.args = args
        self.dataset_path = dataset_path
        self.knowledge = knowledge
        self.encoder_vocab_file = encoder_tokenizer.get_vocab()
        self.decoder_vocab_file = decoder_tokenizer.get_vocab()
        self.sentences = self.load_sentences()
        self.columns = self.load_columns()
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.ij_kg = ij_kg
    def load_sentences(self): 
        sentences = []
        with open(self.dataset_path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        return sentences
    
    def load_columns(self): 
        columns = {}
        with open(self.dataset_path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                try:
                    line = line.strip().split("\t")
                    if line_id == 0:
                        for i, column_name in enumerate(line):
                            columns[column_name] = i
                        continue
                except:
                    pass
        return columns

    def split_answer_question(self): 
        sentences = self.load_sentences(self.dataset_path)
        columns = self.load_columns(self.dataset_path)
        answers = []
        questions = []
        for line in sentences: 
            line = line.strip().split('\t')
            label = str(line[columns['answer']])
            text = str(line[columns['question']])
            answers.append(label)
            questions.append(text)
        assert len(answers) == len(questions), f"length of list of answers and questions must be equal, but got {len(answers)}, {len(questions)}"
        return answers, questions
    
    def create_dataset_inject_kg(self):
        r"""
        input encoder: <s> question + KG
        input decoder: <s> question_kw, KG_kw <\s> answer <s>
        """ 

        answers, questions = self.split_answer_question()
        tokens, pos, vm = self.knowledge.tokenizer_with_vm(questions,max_entities = 8, max_length = 4096)
        vms = [v.astype("bool") for v in vm]
        token_ids = [[self.encoder_vocab_file.get(t) for t in token] for token in tokens]
        mask = [[1 if t != PAD_TOKEN else 0 for t in token] for token in tokens]

        decoder_context = self.knowledge.get_question_with_kg_kw(questions, max_entities = 8)
        decoder_args = decoder_tokenizer(decoder_context, answers, padding = "max_length", truncation = True, max_length = self.args.seq_length_decoder)
        decoder_ids = decoder_args.input_ids
        decoder_attn_mask = decoder_args.attention_mask

        label_pr = decoder_tokenizer(answers, padding = "longest")
        label_ids = label_pr.input_ids
        label_attn_mask = label_pr.attention_mask

        labels_ids = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(label_attn_mask, label_ids)]
        ]

        
        return token_ids, mask, vms, decoder_ids, decoder_attn_mask, labels_ids, pos
    
    def create_dataset_no_inject(self): 
        r"""
        input encoder: <s> question + KG 
        input decoder: <s> question <\s> answer <s>
        """
        answers, questions = self.split_answer_question()
        tokens, pos, vm = self.knowledge.tokenizer_with_vm(questions,max_entities = 8, max_length = 4096)
        vms = [v.astype("bool") for v in vm]
        token_ids = [[self.encoder_vocab_file.get(t) for t in token] for token in tokens]
        mask = [[1 if t != PAD_TOKEN else 0 for t in token] for token in tokens]

        question_kw = self.knowledge.extract_terms(questions)
        decoder_args = decoder_tokenizer(question_kw, answers, padding = "max_length", truncation = True, max_length = self.args.seq_length_decoder)
        decoder_ids = decoder_args.input_ids
        decoder_attn_mask = decoder_args.attention_mask

        label_pr = decoder_tokenizer(answers, padding = "longest")
        label_ids = label_pr.input_ids
        label_attn_mask = label_pr.attention_mask

        labels_ids = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(label_attn_mask, label_ids)]
        ]

        
        return token_ids, mask, vms, decoder_ids, decoder_attn_mask, labels_ids, pos


    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        if self.ij_kg:
            token_ids, mask, vms, decoder_ids, decoder_attn_mask, labels_ids, pos = self.create_dataset_inject_kg()
        else: 
            token_ids, mask, vms, decoder_ids, decoder_attn_mask, labels_ids, pos = self.create_dataset_no_inject()
        dataset_dict = {
            "input_ids": torch.Tensor(token_ids[idx]), 
            "attention_mask": torch.Tensor(mask[idx]), 
            "visible_matrix": torch.Tensor(vms[idx]), 
            "decoder_input_ids": torch.Tensor(decoder_ids[idx]), 
            "decoder_attention_mask": torch.Tensor(decoder_attn_mask[idx]), 
            "labels": torch.Tensor(labels_ids[idx]), 
            "position_ids": torch.Tensor(pos[idx]),
        }
        return dataset_dict


def decoder_tokenizer(args, special_tokens = None): 

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs
    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    tokenizer=  GPT2Tokenizer.from_pretrained(args.decoder_model_name)

    if special_tokens: 
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    
    return tokenizer

def get_model(args, decoder_tokenizer, device,special_tokens = None, load_model_path = None): 
    print("*****************************INITIALIZING MODEL*****************************")
    encoder_config = AutoConfig.from_pretrained(args.encoder_model_name, 
                                                output_hidden_states = True)

    decoder_config = AutoConfig.from_pretrained(args.decoder_model_name,
                                                bos_token_id = decoder_tokenizer.bos_token_id, 
                                                eos_token_id = decoder_tokenizer.eos_token_id, 
                                                sep_token_id = decoder_tokenizer.sep_token_id, 
                                                pad_token_id = decoder_tokenizer.pad_token_id, 
                                                add_cross_attention = True, 
                                                use_cache = False)

    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoder_config, decoder_config=decoder_config)
    config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    config.eos_token_id = decoder_tokenizer.eos_token_id
    config.max_length = args.max_length
    config.min_length = args.min_length
    config.no_repeat_ngram_size = 3
    config.early_stopping = True
    config.length_penalty = 2.0
    config.num_beams = 4

    encoder_model = LongformerModel.from_pretrained(
        args.encoder_model_name, 
        config = encoder_config
    )
    decoder_model = GPT2LMHeadModel.from_pretrained(
        args.decoder_model_name, 
        config = decoder_config
    )
    if special_tokens: 
        decoder_model.resize_token_embeddings(len(decoder_tokenizer))
    
    model = EncoderDecoderModel(
        config=config, 
        encoder=encoder_model, 
        decoder=decoder_model
    )

    if load_model_path != "None":
        print("*****************************LOADING MODEL FROM PRETRAINED*****************************")
        model.load_state_dict(torch.load(load_model_path)) 
    
    model = model.to(device)

    return model


def main():
    #####################################################################################################
    #ARGS
    args = parsers()
    args = load_hyperparam(args)
    set_seed(args.seed)
    #####################################################################################################
    #MODEL      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # roberta_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    gpt2_tokenizer = decoder_tokenizer(args, DECODER_SPECIAL_TOKENS)
    longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = get_model(args, gpt2_tokenizer, device, DECODER_SPECIAL_TOKENS, args.pretrained_model_path)
    #####################################################################################################
    #KNOWLEDGE
    knowledge = KnowledgeGraph(txt_path=args.kg_path, encoder_tokenizer=longformer_tokenizer, decoder_tokenizer=gpt2_tokenizer)
    #####################################################################################################
    #LOADING PREVIOUS INFORMATION (IF POSSIBLE)
    start_epoch = 1
    last_epoch = 0
    best_result = 9999.0

    print("Best result before training: ", best_result)
    if args.last_logging != "None": 
        print(200*"-")
        print("LOADING LOGGING INFORMATION FROM {}".format(args.last_logging))
        last_logger = open(args.last_logging)
        logger_info = json.load(last_logger)
        last_epoch = logger_info['epoch']
        print("Previous epoch: ", last_epoch)
        last_loss = logger_info['total_loss']
        print("Previous loss: ", last_loss)
        best_result = last_loss
        start_epoch += last_epoch
        print("Previous best result: ", best_result)
        print("start_epoch: {0} || last_epoch: {1}".format(start_epoch, last_epoch + args.epochs_num + 1))
        print(200*'-')

    #####################################################################################################
    #EVALUATION 
    rouge = datasets.load_metric("rouge")

    def compute_metrics(pred): 
        labels_ids = pred.label_ids 
        pred_ids = pred.predictions

        pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
        labels_ids[labels_ids==-100] = gpt2_tokenizer.eos_token_id
        label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_token = True)

        rouge2_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
        rouge1_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
        rougeL_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid
        return {
            "rouge1_precision": round(rouge1_output.precision, 4),
            "rouge1_recall": round(rouge1_output.recall, 4),
            "rouge1_fmeasure": round(rouge1_output.fmeasure, 4),
            "rouge2_precision": round(rouge2_output.precision, 4),
            "rouge2_recall": round(rouge2_output.recall, 4),
            "rouge2_fmeasure": round(rouge2_output.fmeasure, 4),
            "rougeL_precision": round(rougeL_output.precision, 4),
            "rougeL_recall": round(rougeL_output.recall, 4),
            "rougeL_fmeasure": round(rougeL_output.fmeasure, 4),   
        }

    #####################################################################################################
    #Train & Eval Dataset
    train_dataset = Medical_Dataset(args.train_path, knowledge=knowledge, encoder_tokenizer=longformer_tokenizer, decoder_tokenizer=gpt2_tokenizer, args=args, ij_kg = True)
    val_dataset = Medical_Dataset(args.dev_path, knowledge=knowledge, encoder_tokenizer=longformer_tokenizer, decoder_tokenizer=gpt2_tokenizer, args=args, ij_kg = True)

    #####################################################################################################
    #TRAINING PHASE 
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    training_args = TrainingArguments(
                        output_dir=args.output_model_path,
                        per_device_train_batch_size=args.batch_size,
                        per_device_eval_batch_size=args.batch_size,
                        logging_dir = args.log_path, 
                        num_train_epochs=args.epochs_num,
                        predict_from_generate=True,
                        evaluate_during_training=True,
                        do_train=True,
                        do_eval=True,
                        evaluation_strategy = "epoch",
                        save_strategy = "epoch", 
                        logging_steps=args.logging_steps,
                        save_steps=args.save_steps,
                        overwrite_output_dir=True,
                        warmup_steps=args.warmup_steps,
                        save_total_limit=10,
                        learning_rate=args.learning_rate,
                        weight_decay = args.weight_decay, 
                        adam_epsilon = args.adam_epsilon, 
                        max_steps = args.max_steps, 
                        max_grad_norm = args.max_grad_norm, 
                        fp16=True,
                    )
    trainer = Trainer(
        model = model, 
        args = training_args, 
        compute_metrics = compute_metrics, 
        train_dataset = train_dataset, 
        eval_dataset = val_dataset
    )

    trainer.train()
if __name__ == "__main__": 
    main()