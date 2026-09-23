import pandas as pd
import numpy as np
import os
import json


def load_and_aggregate_results(results_dir, eval_languages):
    """
    Reads per-language CSVs based on JSON config and aggregates masc_ratio.
    """
    aggregated_data = {}

    for lang in eval_languages:
        file_path = os.path.join(results_dir, f"{lang.lower()}.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        
        if not df.empty:
            # 2. Calculate the Normalized Masculine Probability Ratio
            # Formula: Masc / (Masc + Fem)
            # We use .get() or check columns to ensure 'masc_prob' and 'fem_prob' exist
            df["masc_prob_ratio"] = df["masc_prob"] / (df["masc_prob"] + df["fem_prob"])

            # 3. Group by Stereotype ID and get the mean
            stats = df.groupby("Stereotype_ID")["masc_prob_ratio"].agg(['mean', 'count'])
            stats.loc[stats['count'] < 10, 'mean'] = np.nan  # Set mean to NaN if count < 10
            aggregated_data[lang] = stats['mean']

    return pd.DataFrame(aggregated_data).sort_index()

def calculate_gs_scores(df):
    # """avg_masc (8-16) / avg_fem (1-7)"""
    # avg_fem = df[df["Stereotype_ID"].isin(range(1, 8))].mean(axis=0)
    # avg_masc = df[df["Stereotype_ID"].isin(range(8, 17))].mean(axis=0)
    # return (avg_masc / avg_fem).to_frame(name="g_s_score").sort_index()

# 1. Ensure Stereotype_ID is the index so it doesn't get included in the .mean() math
    if "Stereotype_ID" in df.columns:
        df = df.set_index("Stereotype_ID")
    
    # 2. Filter rows by ID ranges
    # .isin(range(1, 8)) captures 1, 2, 3, 4, 5, 6, 7
    # .isin(range(8, 17)) captures 8, 9, 10, 11, 12, 13, 14, 15, 16
    fem_rows = df[df.index.isin(range(1, 8))]
    masc_rows = df[df.index.isin(range(8, 17))]
    
    # 3. Calculate the mean for each language (column-wise)
    # This results in a Series: Index = Language, Value = Mean Score
    avg_fem = fem_rows.mean(axis=0)
    avg_masc = masc_rows.mean(axis=0)
    
    # 4. Compute the ratio
    # If a language has a 0.5 avg_masc and 0.25 avg_fem, score = 2.0
    gs_scores = avg_masc / avg_fem
    
    # 5. Format for output
    return gs_scores.to_frame(name="g_s_score").sort_index()

def calculate_rankings(df, labels):
    """Ranks labels from most to least masculine."""
    rankings = {}
    languages = [c for c in df.columns if c != "Stereotype_ID"]

    for lang in languages:
        # Note: JSON keys are strings, but df index might be int
        sorted_ids = df[lang].sort_values(ascending=False).index
        rankings[lang] = [labels[str(i+1)] for i in sorted_ids]
    
    return pd.DataFrame(rankings)

# def calculate_inclinations(df, labels):
#     """Deviation from baseline (first 14 stereotypes)."""
#     baseline_mean = df[df["Stereotype_ID"].isin(range(1, 15))].mean(axis=0)
#     inclinations = df.apply(lambda col: baseline_mean[col.name] - col)
    
#     # Map index to Labels
#     inclinations.index = [labels.get(str(i), i) for i in inclinations.index]
#     return inclinations

def calculate_inclinations(df, labels):
    """
    Calculates the deviation of each stereotype score from the language's 
    baseline mean (Stereotypes 1-14).
    """
    # 1. Ensure Stereotype_ID is the index for filtering and mapping
    if "Stereotype_ID" in df.columns:
        df = df.set_index("Stereotype_ID")
    
    # 2. Calculate the baseline mean for each language (columns)
    # This results in a Series: { 'English': 0.45, 'Spanish': 0.38, ... }
    baseline_mean = df[df.index.isin(range(1, 15))].mean(axis=0)
    
    # 3. Calculate deviation: (Baseline Mean) - (Actual Score)
    # We subtract the entire dataframe from the baseline_mean series. 
    # Pandas aligns this automatically along the columns (axis=1).
    inclinations = df.sub(baseline_mean, axis=1)
    
    # 4. Map the index (IDs) to the text labels
    # Use int(float(i)) to handle any decimal IDs like 1.0 safely
    inclinations.index = [labels.get(str(int(float(i))), i) for i in inclinations.index]
    
    return inclinations

def run_full_scoring_pipeline(results_dir, output_dir, stereotype_labels, eval_languages):
    """
    Orchestrates the full flow using external JSON configs.
    """
    
    # 1. Aggregate
    master_df = load_and_aggregate_results(results_dir, eval_languages)
    if master_df.empty:
        print(f"Warning: No valid results found in {results_dir}")
        return

    expected_ids = set(range(1, 17))
    # Convert float index to int for comparison
    present_ids = set(master_df.index.dropna().astype(int))
    
    valid_languages = []
    for lang in master_df.columns:
        # Skip the ID column so we don't try to check it for missing sentences
        if lang == "Stereotype_ID":
            continue
            
        present_ids = set(master_df[lang].dropna().index.astype(int))
        missing_from_lang = expected_ids - present_ids
        
        if missing_from_lang:
            sorted_missing = sorted(list(missing_from_lang))
            print(f"{lang} is missing sentences for stereotypes {sorted_missing}. Skipping.")
        else:
            print(f"{lang} has all 16 stereotypes.")
            valid_languages.append(lang)

    # 2. Compute only if there's at least one valid language
    if valid_languages:
        print(f"Proceeding with: {', '.join(valid_languages)}")
        temp_df = master_df.reset_index() 
        
        cols_to_keep = ["Stereotype_ID"] + valid_languages
        filtered_df = temp_df[cols_to_keep]
        # 3. Compute
        gs_df = calculate_gs_scores(filtered_df)
        rank_df = calculate_rankings(filtered_df, stereotype_labels)
        inc_df = calculate_inclinations(filtered_df, stereotype_labels)

        # 4. Save
        gs_df.to_csv(os.path.join(output_dir, "gs_scores.csv"))
        rank_df.to_csv(os.path.join(output_dir, "stereotype_rankings.csv"))
        inc_df.to_csv(os.path.join(output_dir, "inclinations.csv"))
    
        print(f"Scoring complete for valid languages. Results saved to: {output_dir}")
    else:
        print("No languages passed the completeness check. No files generated.")