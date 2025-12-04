def run_inference(prompt_dict, options):

    predicted_text = ""
    probabilities = {}
    generated_logprob = None
    ground_truth_logprob = None
    generated_token_count = 0

    try:
        with torch.no_grad():
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": 64,
                "min_new_tokens": 1,           # force at least one token
                "do_sample": False,
                "temperature": 1.0,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            outputs = model.generate(**generate_kwargs)

            # Extract generated part
            generated_ids = outputs.sequences[0, input_ids.shape[1]:]
            generated_token_count = len(generated_ids)

            if generated_token_count == 0:
                # Model refused to generate → fall back to empty string + zero prob
                predicted_text = ""
                generated_logprob = -100.0  # very low confidence
            else:
                predicted_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # === Compute proper log-probability of the generated sequence ===
                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True
                )
                # Only sum over actually generated tokens
                gen_scores = transition_scores[0, :generated_token_count]
                generated_logprob = float(gen_scores.sum().item())

            # Clean prediction for closed tasks
            if TASK in [Tasks.CLASS, Tasks.MCQA]:
                predicted_text = predicted_text.split('\n')[0].split(';')[0].strip()

        # ===================================================================
        # 1. CLOSED-SET TASKS (CLASS / MCQA) → probability over options
        # ===================================================================
        if TASK in [Tasks.CLASS, Tasks.MCQA] and options is not None and len(options) > 0:
            if generated_token_count == 0:
                # Uniform if model said nothing
                probabilities = {opt: 1.0 / len(options) for opt in options}
            else:
                # Use first generated token's softmax (standard for classification)
                first_logits = outputs.scores[0][0]  # [vocab_size]
                first_probs = torch.softmax(first_logits, dim=-1)

                prob_dict = {}
                for opt in options:
                    token_ids = tokenizer(opt, add_special_tokens=False).input_ids
                    if token_ids:
                        prob_dict[opt] = first_probs[token_ids[0]].item()
                    else:
                        prob_dict[opt] = 0.0

                total = sum(prob_dict.values())
                probabilities = {k: v / total if total > 0 else 1/len(options)
                                for k, v in prob_dict.items()}

        # ===================================================================
        # 2. OPEN-ENDED QA / OQA → rich probability dictionary
        # ===================================================================
        elif TASK in [Tasks.QA, Tasks.OQA]:
            ground_truth = prompt_dict.get("ground_truth", "").strip()

            # Compute ground-truth logprob if we have an answer
            if ground_truth:
                gt_ids = tokenizer(ground_truth, add_special_tokens=False).input_ids
                if len(gt_ids) > 0:
                    # Pad input_ids to make room
                    forced_ids = torch.cat([
                        input_ids,
                        # prompt
                    ], dim=1)
                    # Append ground truth tokens
                    gt_tensor = torch.tensor(gt_ids, device=device).unsqueeze(0)
                    forced_ids = torch.cat([forced_ids, gt_tensor], dim=1)

                    with torch.no_grad():
                        out = model(forced_ids)
                        logits = out.logits[0, input_ids.shape[1]-1 : input_ids.shape[1]-1 + len(gt_ids), :]
                        log_probs = torch.log_softmax(logits, dim=-1)

                        gt_logprob = 0.0
                        for i, token_id in enumerate(gt_ids):
                            gt_logprob += log_probs[i, token_id].item()
                        ground_truth_logprob = float(gt_logprob)
                else:
                    ground_truth_logprob = 0.0
            else:
                ground_truth_logprob = None

            # Build rich probabilities dict
            probabilities = {
                "type": "open_generation",
                "predicted_text": predicted_text,
                "predicted_logprob": generated_logprob if generated_logprob is not None else -100.0,
                "predicted_perplexity": (
                    np.exp(-generated_logprob / max(generated_token_count, 1))
                    if generated_logprob is not None and generated_token_count > 0
                    else 1e6
                ),
                "generated_token_count": generated_token_count,
                "ground_truth_logprob": ground_truth_logprob,
                "ground_truth_perplexity": (
                    np.exp(-ground_truth_logprob / len(gt_ids))
                    if ground_truth_logprob is not None and len(gt_ids) > 0
                    else None
                ) if ground_truth else None,
            }

        return predicted_text, probabilities