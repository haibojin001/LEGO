# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg46::torch.tensor+torch.unsqueeze
# name: torch_primitive
# summary: Uses torch.tensor, torch.unsqueeze across 2 repos
# anchor_symbols: ['torch.tensor', 'torch.unsqueeze']
# observed in 2 repos: ['d2l-ai__d2l-en', 'sematic-ai__sematic']...

# --- from sematic-ai__sematic::sematic/examples/summarization_finetune/train_eval.py::evaluate_single_text ---
def evaluate_single_text(
    model, tokenizer, model_type, max_new_tokens, eval_tokens=None, eval_text=None
) -> Tuple[str, str]:
    if eval_tokens is None and eval_text is None:
        raise ValueError("One of eval_tokens or eval_text must be provided.")
    if eval_tokens is not None and eval_text is not None:
        raise ValueError("Only one of eval_tokens or eval_text must be provided.")

    if eval_text is not None:
        eval_tokens = torch.tensor(
            tokenizer(
                [eval_text],
                max_length=max_new_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )[0].ids
        )
    else:
        eval_text = tokenizer.batch_decode(
            torch.unsqueeze(eval_tokens, 0).detach().cpu().numpy(),
            skip_special_tokens=True,
        )[0]

    output_tokens = model.generate(
        input_ids=torch.unsqueeze(eval_tokens, 0),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    output_text = tokenizer.batch_decode(
        output_tokens.detach().cpu().numpy(), skip_special_tokens=True
    )
    if model_type is ModelType.causal:
        # Causal ML models extend the input. In our case, we want to
        # consider the "response" to only be the new part, not counting
        # the input.
        output_text = [output_text[0].replace(eval_text, "", 1)]
    return eval_text, output_text[0]

# --- from d2l-ai__d2l-en::d2l/torch.py::predict_seq2seq ---
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
