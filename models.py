from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
) 
import torch

device = ('cuda' if torch.cuda.is_available() else 'cpu')


txt_model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
txt_tokenizer = AutoTokenizer.from_pretrained(txt_model_name)
txt_model = AutoModelForSequenceClassification.from_pretrained(txt_model_name).to(device)

tab_model_name = 'microsoft/tapex-base-finetuned-tabfact'
tab_tokenizer = AutoTokenizer.from_pretrained(tab_model_name)
tab_model = AutoModelForSequenceClassification.from_pretrained(tab_model_name).to(device)