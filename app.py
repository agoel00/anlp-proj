# import gradio as gr 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
) 
import torch
import streamlit as st

device = ('mps' if torch.backends.mps.is_available() else 'cpu')

txt_model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
txt_tokenizer = AutoTokenizer.from_pretrained(txt_model_name)
txt_model = AutoModelForSequenceClassification.from_pretrained(txt_model_name)

tab_model_name = 'microsoft/tapex-base-finetuned-tabfact'
tab_tokenizer = AutoTokenizer.from_pretrained(tab_model_name)
tab_model = AutoModelForSequenceClassification.from_pretrained(tab_model_name)

def text_verify(premise, hypothesis):
    inputs = txt_tokenizer(premise, hypothesis, truncation=True, return_tensors='pt')
    output = txt_model(inputs['input_ids'].to(device))
    prediction = torch.softmax(output['logits'][0], -1).to_list()
    labels = ['Support', 'NotEnoughInfo', 'Refute']
    pred = {name: round(float(pred)*100, 1) for pred, name in zip(prediction, labels)}
    return pred

def tab_verify(table, sent):
    encoding = tab_tokenizer(table, sent, return_tensors='pt')
    outputs = tab_model(**encoding)
    pred_class_idx = outputs.logits[0].argmax(dim=0).item()
    return tab_model.config.id2label[pred_class_idx]

def main():
    st.title('Fact Verification - ANLP Project')
    st.write('Anmol Goel & Ravi Shankar Mishra')
