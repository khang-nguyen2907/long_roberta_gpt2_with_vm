# K-Longformer-GPT2

K-Longformer-GPT2 is a encoder-decoder model. Encoder is injected with a Knowledge graph. Encoder is using Longformer, decoder is using GPT2
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install transformers
pip install rouge_score
pip install sacrebleu
pip install --upgrade gdown
pip install --upgrade datasets
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz
python -m spacy download en_core_web_sm
```
## Dataset 
Dataset is already in ```datasets``` folder. Dataset consists of *train, val, test*

Here, we use ```medical_train```, ```medical_val```, ```medical_test``` to train model. There are 23801 samples, 2975 samples and 2976 samples, repectively. 
About other datasets such as ```*_half```, ```*_mini``` smaller than the main dataset, they are extracted from the main datasets and used to test model whether it works or not after coding. 

## Train 
Notebook for training model is available: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nfKWmo99D46KOqQ41xRH8rfm_ewvfYdo?usp=sharing)

We recommend you to use Colab pro with  GPU P100 x1 and high RAM (25gb RAM) 

If not using note book, run: 

```bash
cd long_roberta_gpt2_with_vm
mkdir checkpoints
mkdir logs
```
- To train model from scratch, set ```checkpoint_path``` as ```None```. If this model is already trained, to keep training, set ```checkpoint_path``` as the path to ```checkpoints``` folder. 
- This model is trained with input of encoder, decoder's format: 
        input encoder: [CLS_TOKEN] question with injected information from Knowledge Graph(KG)
        input decoder: [CLS_TOKEN] question keywords, KG's information keywords [SEP_TOKEN] answers [SEP_TOKEN]
    **question keywords, KG's information keywords** plays as context part that supports model answer generation comprehension
- If you want to train the model with input of decoder without KG's information, there are just question keywords and answers, set ```decoder_with_kg_info``` as ```False```
```python
python longformer_gpt2_train.py \
        --checkpoint_path None \
        --train_path ./datasets/medical_train.tsv \
        --dev_path ./datasets/medical_val.tsv \
        --test_path ./datasets/medical_test.tsv \
        --log_path ./logs \
        --kg_path ./kgs/Medical_kb.spo \
        --batch_size 4 \
        --seq_length_encoder 4096 \
        --seq_length_decoder 1024 \
        --max_length 256 \
        --min_length 50 \
        --epochs_num 5\
        --max_entities 8 \
        --decoder_with_kg_info True
```

Option of ```longformer_gpt2_train.py```
```
useage:     
        --checkpoint_path - [Path of the model's checkpoint folder]
        --train_path - [Path of the train set]
        --dev_path - [Path of the val set]
        --test_path - [Path of the test set]
        --log_path - [Path of the logging folder that contains logging information]
        --kg_path - [Path of KG .spo file]
        --batch_size - [number of Batch size]
        --seq_length_encoder - [Sequence length of encoder]
        --seq_length_decoder - [Sequence length of decoder]
        --max_length - [max length for text generation.]
        --min_length - [min length for text generation.]
        --epochs_num - [Number of epochs]
        --max_entities - [max number of entities queried from KG]
        --decoder_with_kg_info - [Determine that input of decoder (context part) contains KG info or not]
```
