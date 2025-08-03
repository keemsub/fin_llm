from utils import extract_question_and_choices, make_prompt_auto, extract_answer_only
import os
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

test = pd.read_csv('./data/test.csv')
model_name = "beomi/gemma-ko-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

preds = []

for q in tqdm(test['Question'], desc="Inference"):
    prompt = make_prompt_auto(q)
    output = pipe(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)
    pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
    preds.append(pred_answer)

sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['Answer'] = preds
sample_submission.to_csv('./answer.csv', index=False, encoding='utf-8-sig')
