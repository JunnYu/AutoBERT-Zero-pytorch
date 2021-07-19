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

# pytorch
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

# 1w : the goal of life is [ life+0.0738 || true+0.0153 || important+0.015 || happening+0.0142 || born+0.0123 ] .
# 3w : the goal of life is [ completed+0.0404 || expected+0.0338 || set+0.0276 || formed+0.027 || born+0.021 ] .
