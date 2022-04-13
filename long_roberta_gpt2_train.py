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

from model.modeling_long_roberta import *
from model.modeling_longformer import * 
from model.modeling_roberta import *

from transformers import GPT2Tokenizer, AutoConfig, EncoderDecoderConfig, GPT2LMHeadModel, EncoderDecoderModel
from transformers import AdamW, get_scheduler


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
    parser.add_argument("--output_model_path", default="/home/dctuyen/K-BART/k-distilroberta-gpt2/roberta/KDRB_GPT2_model.bin", type=str,
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
    parser.add_argument("--encoder_model_name", type=str, default="distilroberta-base",
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
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Warm up value.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Warm up value.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Warm up value.")
    parser.add_argument("--lr_scheduler_type", type=str, default="SchedulerType.LINEAR",
                        help="Warm up value.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warm up value.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warm up value.")
    parser.add_argument("--num_training_steps", type=int, default=1000,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_path", default="/content/k-distilroberta-gpt2/brain/kgs/Medical.spo",type=str, help="KG name or path")

    args = parser.parse_args()
    return args

class Medical_Datset(Dataset): 
    def __init__(self, dataset_path, encoder_tokenizer, decoder_tokenizer,args, qna = False, ij_kg = False): 
        self.args = args
        self.dataset_path = dataset_path
        self.knowledge = KnowledgeGraph(txt_path=args.kg_path, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer)
        self.encoder_vocab_file = encoder_tokenizer.vocab 
        self.decoder_vocab_file = decoder_tokenizer.get_vocab()
        self.sentences = self.load_sentences()
        self.columns = self.load_columns()
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.qna = qna
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
    
    def __len__(self): 
        return len(self.sentences)

    def __getitem__(self, idx): 
        line = self.sentences[idx]
        line = line.strip().split('\t')
        if len(line) == 2 and self.qna == False: 
            label = str(line[self.columns['answer']])
            text = str(line[self.columns['question']])
            tokens, pos, vm = self.knowledge.add_knowledge_with_vm([text], max_length = self.args.seq_length_encoder)

            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")
            token_ids = [self.vocab_file.get(t) for t in tokens]
            mask = [1 if t != PAD_TOKEN else 0 for t in tokens]
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(label, padding = "max_length", truncation = True, max_length = self.args.seq_length_decoder).input_ids

            input_ids = torch.LongTensor(token_ids)   
            visible_matrix = torch.LongTensor(vm)
            label_ids = torch.LongTensor(labels)
            position_ids = torch.LongTensor(pos)
            mask_ids = torch.LongTensor(mask)

            return input_ids, visible_matrix, label_ids, position_ids, mask_ids
        
        elif len(line) == 2 and self.qna == True and self.ij_kg == True: #decoder: <s> keyword question + kg <\s> answer <\s>
            label = str(line[self.columns['answer']])
            text = str(line[self.columns['question']])
            
            #encoder input
            encoder_tokens, encoder_pos, encoder_vm = self.knowledge.add_knowledge_with_vm_kw_kg([text], max_length= self.args.seq_length_encoder)
            encoder_tokens = encoder_tokens[0]
            encoder_pos = encoder_pos[0]
            encoder_vm = encoder_vm[0].astype("bool")
            encoder_token_ids = [self.encoder_vocab_file.get(t) for t in encoder_tokens]
            encoder_mask = [1 if t != PAD_TOKEN else 0 for t in encoder_tokens]

            #decoder input 
            decoder_tokens, decoder_pos, decoder_vm = self.knowledge.add_knowledge_with_vm_kw_kg([text, label], max_length = self.args.seq_length_decoder)
            decoder_tokens = decoder_tokens[0]
            decoder_pos = decoder_pos[0]
            decoder_vm = decoder_vm[0].astype("bool")
            decoder_token_ids = [self.decoder_vocab_file.get(t) for t in decoder_tokens]
            decoder_mask = []
            seg_tag = 1
            for t in decoder_tokens: 
                if t == PAD_TOKEN: 
                    decoder_mask.append(0)
                else:
                    decoder_mask.append(seg_tag)
                if t == SEP_TOKEN: 
                    seg_tag += 1

            #label 
            label_pr = self.decoder_tokenizer(label, padding = "max_length", truncation = True, max_length = self.args.seq_length_decoder)
            label_ids = label_pr.input_ids
            label_attn_mask = label_pr.attention_mask
            labels = [-100 if mask == 0 else token for mask, token in zip(label_attn_mask, label_ids)]

            #Tensor 
            encoder_input_ids = torch.LongTensor(encoder_token_ids)
            encoder_visible_matrix = torch.LongTensor(encoder_vm)
            encoder_position_ids = torch.LongTensor(encoder_pos)
            encoder_seg_mask = torch.LongTensor(encoder_mask)

            decoder_input_ids = torch.LongTensor(decoder_token_ids)
            decoder_visible_matrix = torch.LongTensor(decoder_vm)
            decoder_position_ids = torch.LongTensor(decoder_pos)
            decoder_seg_mask = torch.LongTensor(decoder_mask)

            labels_ids = torch.LongTensor(labels)

            return encoder_input_ids, encoder_visible_matrix, encoder_position_ids, encoder_seg_mask, decoder_input_ids, decoder_visible_matrix, decoder_position_ids, decoder_seg_mask, labels_ids
        
        elif len(line) == 2 and self.qna == True and self.ij_kg == False: #<s>keyword question <\s> answer <\s>
            label = str(line[self.columns['answer']])
            text = str(line[self.columns['question']])

            #encoder input 
            encoder_tokens, encoder_pos, encoder_vm = self.knowledge.add_knowledge_with_vm_kw_kg([text], max_length= model_args.max_pos)
            encoder_tokens = encoder_tokens[0]
            encoder_pos = encoder_pos[0]
            encoder_vm = encoder_vm[0].astype("bool")
            encoder_token_ids = [self.encoder_vocab_file.get(t) for t in encoder_tokens]
            encoder_mask = [1 if t != PAD_TOKEN else 0 for t in encoder_tokens]

            #decoder input 
            kw_question = self.knowledge.extract_keyword(text) #return a list of question keywords 
            kw_question_txt = ", ".join(kw_question)
            decoder_args = self.decoder_tokenizer(kw_question_txt, label, padding = "max_length", truncation = True, max_length = self.args.seq_length_decoder)
            decoder_ids = decoder_args.input_ids
            decoder_attn_mask = decoder_args.attention_mask

            #label 
            label_pr = self.decoder_tokenizer(label, padding = "max_length", truncation = True, max_length = self.args.seq_length_decoder)
            label_ids = label_pr.input_ids
            label_attn_mask = label_pr.attention_mask
            labels = [-100 if mask == 0 else token for mask, token in zip(label_attn_mask, label_ids)]

            #Tensor
            encoder_input_ids = torch.LongTensor(encoder_token_ids)
            encoder_visible_matrix = torch.LongTensor(encoder_vm)
            encoder_position_ids = torch.LongTensor(encoder_pos)
            encoder_seg_mask = torch.LongTensor(encoder_mask)

            decoder_input_ids = torch.LongTensor(decoder_ids)
            decoder_attention_mask = torch.LongTensor(decoder_attn_mask)

            labels_ids = torch.LongTensor(labels)

            return encoder_input_ids, encoder_visible_matrix, encoder_position_ids, encoder_seg_mask, decoder_input_ids, decoder_attention_mask, labels_ids




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
    # encoder_config = AutoConfig.from_pretrained(args.encoder_model_name, 
    #                                             max_position_embeddings = args.seq_length_encoder, 
    #                                             output_hidden_states = True)
    decoder_config = AutoConfig.from_pretrained(args.decoder_model_name,
                                                bos_token_id = decoder_tokenizer.bos_token_id, 
                                                eos_token_id = decoder_tokenizer.eos_token_id, 
                                                sep_token_id = decoder_tokenizer.sep_token_id, 
                                                pad_token_id = decoder_tokenizer.pad_token_id, 
                                                add_cross_attention = True, 
                                                use_cache = False)

    d_roberta_base = RobertaModel.from_pretrained(args.encoder_model_name)
    d_roberta_base_tokenizer = RobertaTokenizerFast.from_pretrained(args.encoder_model_name)
    training_args.output_dir = args.encoder_model_path
    encoder_model_path = f'{training_args.output_dir}/distilroBERTa-base-{model_args.max_pos}'
    if not os.path.exists(encoder_model_path): 
        os.makedirs(encoder_model_path)
    
    encoder_model, encoder_tokenizer = create_long_model(
        save_model_to=encoder_model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos
    )
    encoder_tokenizer_long = RobertaTokenizerFast.from_pretrained(encoder_model_path)
    encoder_model_long = RobertaModel.from_pretrained(encoder_model_path)
    encoder_config = encoder_model_long.config

    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoder_config, decoder_config=decoder_config)
    config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    config.eos_token_id = decoder_tokenizer.eos_token_id
    config.max_length = args.max_length
    config.min_length = args.min_length
    config.no_repeat_ngram_size = 3
    config.early_stopping = True
    config.length_penalty = 2.0
    config.num_beams = 4

    # encoder_model = AutoModel.from_pretrained(
    #     args.encoder_model_name, 
    #     config = encoder_config
    # )
    decoder_model = GPT2LMHeadModel.from_pretrained(
        args.decoder_model_name, 
        config = decoder_config
    )
    if special_tokens: 
        decoder_model.resize_token_embeddings(len(decoder_tokenizer))
    
    model = EncoderDecoderModel(
        config=config, 
        encoder=encoder_model_long, 
        decoder=decoder_model
    )

    #freeze the first 6 layers, unfreeze the rest 
    for parameter in model.decoder.parameters(): 
        parameter.requires_grad = False
    for i, m in enumerate(model.decoder.transformer.h): 
        #Only un-freeze the last n transformer blocks 
        if i + 1 > 6:
            for parameter in m.parameters():
                parameter.requires_grad = True 
    for parameter in model.decoder.transformer.ln_f.parameters(): 
        parameter.requires_grad = True 
    for parameter in model.decoder.lm_head.parameters(): 
        parameter.requires_grad = True
    rand_weight = torch.rand(model.decoder.lm_head.weight.shape)
    model.decoder.lm_head.weight = torch.nn.parameter.Parameter(rand_weight)

    if load_model_path != "None":
        print("*****************************LOADING MODEL FROM PRETRAINED*****************************")
        model.load_state_dict(torch.load(load_model_path)) 
    
    model = model.to(device)

    return model, encoder_tokenizer_long


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
    model, roberta_tokenizer = get_model(args, gpt2_tokenizer, device, DECODER_SPECIAL_TOKENS, args.pretrained_model_path)
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

    def evaluate(args, is_test):
        if is_test: 
            eval_dataset = Medical_Datset(dataset_path= args.test_path, encoder_tokenizer=roberta_tokenizer, decoder_tokenizer= gpt2_tokenizer, args= args, qna= True, ij_kg = False)
            eval_loader = DataLoader(dataset = eval_dataset, batch_size=args.batch_size)
        else: 
            eval_dataset = Medical_Datset(dataset_path= args.dev_path, encoder_tokenizer=roberta_tokenizer, decoder_tokenizer= gpt2_tokenizer, args= args, qna= True, ij_kg = False)
            eval_loader = DataLoader(dataset = eval_dataset, batch_size = args.batch_size)
        
        model.eval()
        eval_loop  = tqdm(enumerate(eval_loader), total = len(eval_loader))
        for eval_batch_idx, eval_inputs in eval_loop: 
            input_ids = eval_inputs[0].to(device)
            visible_matrix = eval_inputs[1].to(device)
            label_ids = eval_inputs[2].to(device)
            position_ids = eval_inputs[3].to(device)
            mask_ids = eval_inputs[4].to(device)

            pred_ids = model.generate(
                input_ids   
            )
            pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_ids = np.where(label_ids != -100, label_ids, gpt2_tokenizer.pad_token_id)
            label_str = gpt2_tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            rouge1_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge1"])["rouge1"].mid
            rouge2_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rouge2"])["rouge2"].mid
            rougeL_output = rouge.compute(predictions = pred_str, references=label_str, rouge_types = ["rougeL"])["rougeL"].mid


            rouge1_precision = round(rouge1_output.precision, 4) 
            rouge1_recall = round(rouge1_output.recall, 4)
            rouge1_fmeasure = round(rouge1_output.fmeasure, 4)
            
            rouge2_precision = round(rouge2_output.precision, 4) 
            rouge2_recall = round(rouge2_output.recall, 4)
            rouge2_fmeasure = round(rouge2_output.fmeasure, 4)
            
            rougeL_precision = round(rougeL_output.precision, 4) 
            rougeL_recall = round(rougeL_output.recall, 4)
            rougeL_fmeasure = round(rougeL_output.fmeasure, 4)
            metrics_result = {
                "rouge1_precision" : rouge1_precision, 
                "rouge1_recall": rouge1_recall, 
                "rouge1_fmeasure": rouge1_fmeasure, 
                "rouge2_precision": rouge2_precision, 
                "rouge2_recall": rouge2_recall, 
                "rouge2_fmeasure": rouge2_fmeasure, 
                "rougeL_precision": rougeL_precision, 
                "rougeL_recall": rougeL_recall, 
                "rougeL_fmeasure": rougeL_fmeasure
            }
            if is_test: 
                print("report TEST: rouge1_precision: {0} \trouge1_recall: {1} \trouge1_fmeasure: {2}".format(rouge1_precision, rouge1_recall, rouge1_fmeasure))
                print("report TEST: rouge2_precision: {0} \trouge2_recall: {1} \trouge2_fmeasure: {2}".format(rouge2_precision, rouge2_recall, rouge2_fmeasure))
                print("report TEST: rougeL_precision: {0} \trougeL_recall: {1} \trougeL_fmeasure: {2}".format(rougeL_precision, rougeL_recall, rougeL_fmeasure))
                return metrics_result
            else: 
                print("report VAL: rouge1_precision: {0} \trouge1_recall: {1} \trouge1_fmeasure: {2}".format(rouge1_precision, rouge1_recall, rouge1_fmeasure))
                print("report VAL: rouge2_precision: {0} \trouge2_recall: {1} \trouge2_fmeasure: {2}".format(rouge2_precision, rouge2_recall, rouge2_fmeasure))
                print("report VAL: rougeL_precision: {0} \trougeL_recall: {1} \trougeL_fmeasure: {2}".format(rougeL_precision, rougeL_recall, rougeL_fmeasure))
                return metrics_result

    #####################################################################################################
    #TRAINING PHASE 
    print("************************Start training************************")
    train_dataset = Medical_Datset(dataset_path= args.train_path,encoder_tokenizer=roberta_tokenizer, decoder_tokenizer= gpt2_tokenizer, args = args, qna = True, ij_kg = False)
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    
    instances_num = len(train_dataset)
    num_training_steps = args.epochs_num * len(train_loader)
    args.report_steps = len(train_loader)

    print("Batch size: ", args.batch_size)
    print("The number of training instances:", instances_num)
    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear", 
        optimizer = optimizer, 
        num_warmup_steps = args.warmup_steps, 
        num_training_steps = num_training_steps
    )

    #Training Network 
    total_loss = 0.
    result = 0.0 
    for epoch in range(start_epoch, last_epoch + args.epochs_num + 1):
        print("\n")
        print(150 * '-')
        t1 = time.time()
        info = {}
        total_losses = []
        losses = []
        model.train()
        loop = tqdm(enumerate(train_loader), total = len(train_loader))
        for batch_idx, inputs in loop: 
            model.zero_grad()

            encoder_input_ids = inputs[0]
            encoder_visible_matrix = inputs[0]
            encoder_position_ids = inputs[0]
            encoder_seg_mask = inputs[0]
            decoder_input_ids = inputs[0]
            decoder_attention_mask = inputs[0]
            labels_ids = inputs[0]



            outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_visible_matrix,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=labels_ids,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                position_ids = encoder_position_ids
                )
            loss = outputs[0]

            losses.append(loss.item())
            total_loss += loss.item()
            
            #backward 
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            #update progress bar 
            loop.set_description(f"Epoch [{epoch/args.epochs_num}]")
            loop.set_postfix(loss = loss.item())
            
        print("Avg. loss: ", total_loss/args.report_steps)
        total_loss = 0.

        print("\n***********Start evaluation on dev dataset***********")
        result = evaluate(args, False)
        

        print("\n***********Start evaluation on test dataset***********")
        rt = evaluate(args, True)
        
        t2 = time.time()
        info['epoch'] = int(epoch)
        info['total_loss'] = float(total_losses[-1])
        info['loss'] = losses
        info['val'] = result
        info['test'] = rt
        info['time'] = t2-t1
        path_log = os.path.join(args.log_path, "log_epoch_"+str(epoch)+".json")
        with open(path_log, mode = "w") as outfile: 
            json.dump(info, outfile)

        model.encoder = copy_proj_layers(model.encoder)
        ttl = float(total_losses[-1])
        if ttl < best_result: 
            best_result = ttl
            save_model(model, args.output_model_path)

    #Evaluation phase 
    print("\nFinal evaluation on the test dataset")

    if torch.cuda.device_count()>1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else: 
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)
    print("\nTraining progress completed.")

if __name__ == "__main__": 
    main()