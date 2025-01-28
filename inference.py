import torch
from transformers import DebertaTokenizerFast, DebertaForTokenClassification

# Load model and tokenizer
model = DebertaForTokenClassification.from_pretrained("./results/ner_deberta_model")
tokenizer = DebertaTokenizerFast.from_pretrained("./results/ner_deberta_model")

# Input sentence
input_text =  "\n\n Subject : Urgent Issue with Recent Transactions \n\n Dear Customer Service, \n\n I hope this message finds you well . My name is Chris Evans and I am writing to express my concern regarding some unusual transactions on my account.\n I have noticed several unauthorized charges on my credit card ending in 6592336767396731.\n These transactions occurred two hours back, and I am sure that I did not make these purchases.\n\n Additionally , I noticed access to my email account tied to the address gloria81@example.net from a suspicious IP , 61.4.73.188.\n I'm concerned that my financial and personal data might be compromised.\n\n My phone number is +1-283-591-3645x5362 , and you can reach me there at any time.\n For security purposes, please verify the details associated with my bank account , which ends in HQPZ42470309251564, and ensure that no other unauthorized access has occurred.\n I frequently travel to Petersland , but haven't made any trips recently , which leads me to believe this might be a targeted attack.\n I would appreciate it if you could block any future transactions from unfamiliar IPs and please send any correspondence through my secure URL which is http://mitchell.net/ \n Looking forward to resolving this swiftly.\n\n Best regards, \n Chris Evans"

# Tokenize input
tokenized_input = tokenizer(input_text.split(), return_tensors="pt", is_split_into_words=True)

tokens_from_ids = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])
input_cleaned_tokens = [token.replace('Ġ', '') for token in tokens_from_ids]

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(**tokenized_input)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()

# Placeholder redacted text
redacted_output = []
skip_list = [19]

# Assume labels 1 means the entity should be redacted, and 0 means no redaction
for each_tkn, pred in zip(input_cleaned_tokens, predictions):
    if pred not in skip_list:
        each_tkn = 'x'* len(str(each_tkn))
        redacted_output.append(each_tkn)
    else:
        redacted_output.append(each_tkn)

# Join the redacted output
redacted_sentence = " ".join(redacted_output)

print(f"Original: {input_text}")

print(f"\n\nRedacted: {redacted_sentence}")
