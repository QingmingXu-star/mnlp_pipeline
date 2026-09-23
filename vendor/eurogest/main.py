import os
import fire
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

pd.reset_option('display.max_colwidth')

# Import utils
from utils import (
    setup_environment, 
    load_scaffolds_configs, 
    load_summary_configs,
    SUPPORTED_LANGS, 
    get_consistent_indices, 
    format_target_stereotype,
    build_row_prompts,
    get_sequence_log_probs,
    tokenise,
    generate_new_tokens,
    run_full_scoring_pipeline
)

def evaluate_sentence(model_inputs, model, tokenizer, device, normalisation, measure_whole_sequences=False):

    if len(model_inputs) >= 2:
        model_inputs_m = model_inputs[0]
        model_inputs_f = model_inputs[1]

        inputs_m, n_m, t_m = tokenise(model_inputs_m, tokenizer, device)
        inputs_f, n_f, t_f = tokenise(model_inputs_f, tokenizer, device)
        
        # find any token indices where the masculine and feminine sentences differ 
        diff_indices = [i for i in range(min(len(t_m), len(t_f))) if t_m[i] != t_f[i]]
        # check also for cases where one sentence is actually contained in the other longer one 
        if not diff_indices and len(t_m) != len(t_f):               
            start = max(len(t_m), len(t_f)) - (max(len(t_m), len(t_f)) - min(len(t_m), len(t_f))) - 2 # we have to back up 2 tokens so that we calculate probs on last portion of shorter sentence as well as longer one 
        else:
            # we then set our 'start point' as one back from the point at which the sentences first diverge
            start = diff_indices[0] - 1 if diff_indices else 0

        # get log probs of each token in the masc and fem sequences
        lp_m = get_sequence_log_probs(inputs_m["input_ids"], model, device)
        lp_f = get_sequence_log_probs(inputs_f["input_ids"], model, device)

        # if there's also a gender neutral option, get log probs of this sequence as well 
        if len(model_inputs) > 2:
            model_inputs_n = model_inputs[2]
            input_ids_n, n_n, t_n = tokenise(model_inputs_n, tokenizer, device)
            lp_n = get_sequence_log_probs(input_ids_n["input_ids"], model, device, start_index=start)
        else:
            lp_n, n_n, t_n = None, None, None

        if not measure_whole_sequences:
            # to measure the portion of the sentences that differ, sum over only the log probs of tokens after point of divergence 
            lp_m_sum = sum(lp_m[start:] if isinstance(lp_m, (list, np.ndarray)) else lp_m)
            num_diff_m = len(lp_m[start:])
            lp_f_sum = sum(lp_f[start:] if isinstance(lp_f, (list, np.ndarray)) else lp_f)
            num_diff_f = len(lp_f[start:])
            if lp_n is not None:
                lp_n_sum = sum(lp_n[start:] if isinstance(lp_n, (list, np.ndarray)) else lp_n)
            else:
                lp_n_sum = None

        else:
            # if measure whole sentences is set to true, we ignore start index and sum log probs over whole sentence
            lp_m_sum = sum(lp_m) if isinstance(lp_m, (list, np.ndarray)) else lp_m
            num_diff_m = len(lp_m)
            lp_f_sum = sum(lp_f) if isinstance(lp_f, (list, np.ndarray)) else lp_f
            num_diff_f = len(lp_f)
            if lp_n is not None:
                lp_n_sum = sum(lp_n) if isinstance(lp_n, (list, np.ndarray)) else lp_n
            else:
                lp_n_sum = None

        lp_m_avg = (lp_m_sum / num_diff_m) if normalisation else lp_m_sum
        lp_f_avg = (lp_f_sum / num_diff_f) if normalisation else lp_f_sum
        lp_n_avg = None if normalisation else lp_n_sum # TO DO: deal with gender-neutral translations in normalisation 

        # convert log probs into probs
        prob_m = np.exp(lp_m_avg)
        prob_f = np.exp(lp_f_avg)
        prob_n = np.exp(lp_n_avg) if lp_n_avg is not None else None

        ## Also do extrinsic evaluation by generating novel text (generate a few extra tokens) 
        tokens_to_generate = max(max(num_diff_m, num_diff_f)+3, 5) # we set a minimum of 5 tokens to generate to ensure we get a meaningful continuation, even if only one token differs in the original sentences

        identical_tokens = {k: v[:, :start] for k, v in inputs_m.items()}
        generated_text = generate_new_tokens(identical_tokens, model, tokenizer, tokens_to_generate, device, do_sample=False)  # using greedy decoding for now

        return {
            "masc_tokens": t_m, 
            "fem_tokens": t_f, 
            "neutral_tokens": t_n,
            "masc_log_probs_sequence": lp_m,
            "fem_log_probs_sequence": lp_f,
            "neutral_prob_sequence": lp_n,
            "masc_prob": prob_m,
            "fem_prob": prob_f,
            "neutral_prob": prob_n,
            "identical_tokens": t_m[:start],
            "generated_text": generated_text
        }

    else: # if there's only one gender-generic input to test, we only do generation 
        model_inputs_n = model_inputs[0]
        inputs_n, n_n, t_n = tokenise(model_inputs_n, tokenizer, device)
        tokens_to_generate = 10
        generated_text = generate_new_tokens(inputs_n, model, tokenizer, tokens_to_generate, device, do_sample=False)

    return {
            "masc_tokens": None, 
            "fem_tokens": None, 
            "neutral_tokens": t_n,
            "masc_log_probs_sequence": None,
            "fem_log_probs_sequence": None,
            "neutral_prob_sequence": None,
            "masc_prob": None,
            "fem_prob": None,
            "neutral_prob": None,
            "identical_tokens": t_n,
            "generated_text": generated_text
        }


def main(hf_token, 
         hf_dataset_path, 
         model_id, 
         sample_size, 
         eval_languages, 
         results_folder, 
         seed=42,
         target_stereotype="none",
         use_common_indices=False, # set to True if you want to only select the common seentences from the languages of evaluation for direct comparisons
         normalisation=True, # normalise by number of tokens after the point at which the sentences diverge
         measure_whole_sequences=False): # normally we just measure and normalise the portion of the sentences that differ between masc and fem  
   
    # 1. Setup
    device = setup_environment(seed)
    login(token=hf_token)
    os.makedirs(results_folder, exist_ok=True)

    # 2. Load Configs 
    punc_map, scaffolds = load_scaffolds_configs()
    stereotype_labels, language_test_configs = load_summary_configs()

    # 3. Initialization
    # load model, tokenizer and dataset 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # ensure padding is correct for batching if needed later
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    dataset = load_dataset(hf_dataset_path)

    # 4. Select data samples as specified by inputs (stereotype, languages, sample size)
    language_set = eval_languages
    target_stereotype = format_target_stereotype(target_stereotype)

    if target_stereotype:
        dataset = dataset.filter(lambda x: x['Stereotype_ID'] in target_stereotype)
    if use_common_indices:
        sampled_indices = get_consistent_indices(dataset, language_set, sample_size, seed=seed)
    else:
        sampled_indices = {}
        for lang in language_set:
            available_gest_ids = dataset[lang]['GEST_ID']
            df_len = len(available_gest_ids)
            n = df_len if sample_size == 1 else min(int(sample_size), df_len)
            sampled_indices[lang] = random.sample(available_gest_ids, n)

    # # 5. Process Languages
    for eval_lang in eval_languages:
        print(f"PROCESSING: {eval_lang}")
        # --- A. Handle Dataset Loading First ---
        try:
            # We must define 'df' here so it is specific to the current eval_lang
            df = dataset[eval_lang].to_pandas()
            df = df[df['GEST_ID'].isin(sampled_indices[eval_lang])]
           
            if df.empty:
                print(f"No rows found for {eval_lang} with the specified criteria. Skipping.")
                continue

        except KeyError:
            print(f"Skipping {eval_lang}: not in dataset.")
            continue

        output_path = os.path.join(results_folder, f"{eval_lang.lower()}.csv")

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {eval_lang}"):

            if device.type == 'mps':
                torch.mps.empty_cache()
                
            is_gendered = pd.notna(row.get('Masculine')) and pd.notna(row.get('Feminine'))
            is_neutral = pd.notna(row.get('Neutral'))
            
            if not is_gendered and not is_neutral:
                continue
            
            prompting_inputs, cond, masc_word, fem_word = build_row_prompts(row, is_gendered, eval_lang, scaffolds, punc_map, language_test_configs)

            if prompting_inputs is None:
                print(f"Skipping {row}")
                continue

            for prompt_id, model_inputs in prompting_inputs.items():
                scores = evaluate_sentence(model_inputs, model, tokenizer, device, normalisation, measure_whole_sequences)
                results.append({
                    "GEST_ID": row['GEST_ID'],
                    "Source": row['Source'],
                    "Stereotype_ID": row['Stereotype_ID'],
                    "Condition": cond,
                    "Prompt ID": prompt_id,
                    "masc word": masc_word,
                    "fem word": fem_word,
                    **scores,
                })

        
        # 6. Save Results
        if results:
            final_df = pd.DataFrame(results)
            join_keys = ["GEST_ID", "Source", "Stereotype_ID", "Condition"]

            # pd.set_option('display.max_colwidth', 25)
            print(final_df.head())

            final_df.to_csv(output_path, index=False)

    print("--- Running Final Scoring ---")
    # results_folder is where individual lang csvs are
    # we can save the scores in a 'summary' subfolder
    run_full_scoring_pipeline(
        results_dir=results_folder, 
        output_dir=os.path.join(results_folder, "summary_metrics"),
        stereotype_labels=stereotype_labels,
        eval_languages=eval_languages
    )

if __name__ == "__main__":
    fire.Fire(main)
