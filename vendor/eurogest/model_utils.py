import torch

def tokenise(text, tokenizer, device):

    inputs = tokenizer(text, return_tensors="pt").to(device)
    # collect however many tokens were evaluated 
    num_tokens = inputs["input_ids"].shape[1]
    readable_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return inputs, num_tokens, readable_tokens


def get_sequence_log_probs(input_ids, model, device):

    with torch.no_grad():
        logits = model(input_ids).logits # Shape: [batch, seq_len, vocab_size]

    # Shift logits and targets so we predict the next token
    # Logits for tokens 0 to N-1 predict tokens 1 to N
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Calculate log_softmax over vocabulary, then gather the log probs of the tokens being evaluated 
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
    log_probs_list = token_log_probs[0].cpu().numpy().tolist()

    if device.type == 'mps':
        torch.mps.empty_cache()
    
    return log_probs_list


def generate_new_tokens(inputs, model, tokenizer, num_tokens, device, do_sample=False, temperature=0):
    # in very few cases, the gendered term is right at the start of the sentece we are testing so we can't generate completion for this 
    if "input_ids" in inputs and inputs["input_ids"].shape[-1] == 0:
        print("Warning: Received empty input_ids. Skipping generation.")
        return ""

    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=num_tokens, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature
    )

    # Using [0] assumes you are processing one prompt at a time
    new_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
    
    # 2. Decode to string
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 3. Safety Check: If 'decoded' is somehow a list, join it.
    if isinstance(decoded, list):
        return " ".join(decoded)

    # 4. Remove newlines, tabs, and commas
    sanitized = decoded.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace(",", " ")
    
    return " ".join(sanitized.split())
