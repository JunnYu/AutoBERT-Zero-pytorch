# AutoBERT-Zero-pytorch
AutoBERT-Zero论文非官方代码，个人复现。

# Paper
**[AutoBERT-Zero: Evolving BERT Backbone from Scratch](https://arxiv.org/pdf/2107.07445)**  
*Jiahui Gao, Hang Xu, Han shi, Xiaozhe Ren, Philip L.H. Yu, Xiaodan Liang, Xin Jiang, Zhenguo Li*

<p align="center">
    <img src="figure/AutoBert-Zero_v2.jpg" width="100%" />
</p>


# Install
```bash
pip install git+https://github.com/JunnYu/AutoBERT-Zero-pytorch
or
pip install autobert
```

# Wandb Logs
https://wandb.ai/junyu/huggingface/runs/howc6tps?workspace=user-junyu
感觉small模型训练出来效果不怎么样，不知道是哪里出错了。

# Usage
```python
import torch
from transformers import BertTokenizerFast
from autobert import AutoBertModelForMaskedLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained("weights/autobert-small-3w")
model = AutoBertModelForMaskedLM.from_pretrained("weights/autobert-small-3w")
model.to(device)

# text = "It is a [MASK] day today."
text = "The goal of life is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs).logits[0]

pt_outputs_sentence = ""
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        prob, indice = outputs[i].softmax(-1).topk(k=5)
        tokens = tokenizer.convert_ids_to_tokens(indice)
        slist = []
        for p, t in zip(prob, tokens):
            slist.append(t + "+" + str(round(p.item(), 4)))
        pt_outputs_sentence += " " + "[ " + " || ".join(slist) + " ]"
    else:
        pt_outputs_sentence += " " + "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True)
        )

print(pt_outputs_sentence.strip())

text:  Beijing is the capital of [MASK].
loading weights\autobert-small-1w
beijing is the capital of [ beijing+0.6121 || china+0.1775 || taiwan+0.0073 || it+0.0072 || us+0.0069 ] .
====================================================================================================
loading weights\autobert-small-3w
beijing is the capital of [ china+0.1321 || beijing+0.0697 || india+0.0321 || interest+0.0209 || shanghai+0.0185 ] .
====================================================================================================
loading weights\autobert-small-7w
beijing is the capital of [ beijing+0.399 || china+0.2677 || nanjing+0.0124 || shanghai+0.0105 || robotics+0.0089 ] .
====================================================================================================
loading weights\autobert-small-10w
beijing is the capital of [ paradise+0.0111 || mine+0.0109 || london+0.0104 || cosmos+0.0086 || rebirth+0.0075 ] .
====================================================================================================
loading weights\autobert-small-28w
beijing is the capital of [ beijing+0.1828 || china+0.1591 || tian+0.0174 || chinese+0.012 || capital+0.0103 ] .
====================================================================================================
loading weights\autobert-small-44w
beijing is the capital of [ china+0.2423 || beijing+0.1273 || .+0.0262 || capital+0.0225 || tianjin+0.0122 ] .
====================================================================================================
loading weights\autobert-small-61w
beijing is the capital of [ china+0.2801 || beijing+0.1326 || shanghai+0.0367 || tianjin+0.0216 || india+0.0147 ] .
====================================================================================================
loading weights\autobert-small-83w
beijing is the capital of [ china+0.3552 || beijing+0.0539 || shanghai+0.0196 || america+0.0134 || mongolia+0.012 ] .
====================================================================================================
loading weights\autobert-small-104w
beijing is the capital of [ china+0.3014 || mine+0.0177 || india+0.0148 || us+0.0117 || beijing+0.0112 ] .
====================================================================================================
loading weights\autobert-small-107w
beijing is the capital of [ china+0.1801 || india+0.0273 || america+0.0181 || japan+0.0166 || us+0.0143 ] .
====================================================================================================

```
