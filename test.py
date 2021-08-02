import torch
from transformers import BertTokenizerFast

from autobert import AutoBertModelForMaskedLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for model in ["junnyu/autobert-small-light","junnyu/autobert-small-sdconv"]:
    tokenizer = BertTokenizerFast.from_pretrained(model)
    model = AutoBertModelForMaskedLM.from_pretrained(model)
    model.to(device)

    text = "Beijing is the capital of [MASK]."
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

