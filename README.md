# Meta-opt-FT-dolly
---
license: other
base_model: facebook/opt-350m
tags:
- generated_from_trainer
model-index:
- name: colab_text_generation_FT_opt_on_dolly_UT_V2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# colab_text_generation_FT_opt_on_dolly_UT_V2

This model is a fine-tuned version of [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) on argilla/databricks-dolly-15k.

## Model description

Colab_text_generation_FT_opt_on_dolly_UT_V2  is an open-source language model, fine-tuned version of facebook/opt-350m and Supervised Finetuning was used to retrain and finetune the model - a strategy inspired by offline transfer reinforcement learning. This version of Model learn from mixed-quality data without preference labels, delivering exceptional performance. Despite the simple approach, my commitment is to develop a high-performance, commercially viable, open-source large language model, and I continue to make significant strides toward this vision.

## Intended uses & limitations

The model, initially trained only on training data, serves multiple purposes such as generating prompts for downstream task evaluation and text generation. Furthermore, it can undergo fine-tuning for specific downstream tasks using the CLM (Causal Language Modeling) example.

## Training data

The data on which this model was trained is argilla/databricks-dolly-15k-curated-en.
Within this dataset, you'll discover a compilation of entries featuring a category, an instruction, a context, and a response corresponding to that instruction. The project's objective is to enhance the quality of instructions, inputs, and responses, ensuring they align seamlessly with their designated task category. All textual components should be articulate, providing genuine information. Additionally, responses should strive for completeness while maintaining conciseness.

## How to use

```
category = "creative_writing"

instruction="Based on the following paragraph, what are some of the factors that are likely to influence gene expression in humans?"

prompt = f"""
### Category:
{category}
            
### Instruction:
{instruction}
            
### Context:

            
### Response:

"""
from transformers import AutoTokenizer, AutoModelForCausalLM

trained_tokenizer = AutoTokenizer.from_pretrained("01GangaPutraBheeshma/colab_text_generation_FT_opt_on_dolly_UT_V2")
trained_model = AutoModelForCausalLM.from_pretrained("01GangaPutraBheeshma/colab_text_generation_FT_opt_on_dolly_UT_V2")
input_ids = trained_tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
# with torch.inference_mode():

print(f"After Training Response :")
outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=1000, do_sample=True, top_p=0.9,temperature=1.0)
print(f"-------------------------\n\n")
print(f"Generated instruction:\n{trained_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
print(f"-------------------------\n\n")

```

## Training procedure

The texts are tokenized using the GPT2 byte-level version of Byte Pair Encoding (BPE) (for unicode characters).

The 331M model was trained on Google Colab's A100 GPU. The training duration was roughly ~24 hours of continuous training.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 2



### Framework versions

- Transformers 4.35.2
- Pytorch 2.1.0+cu118
- Datasets 2.15.0
- Tokenizers 0.15.0
