from transformers import AutoTokenizer

# Load the tokenizer
model_name = "ibm-granite/granite-3b-code-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get the tokens for IDs 29 and 0
token_29 = tokenizer.decode([29])
token_0 = tokenizer.decode([0])

print(f"Token ID 29: '{token_29}'")
print(f"Token ID 0: '{token_0}'")
