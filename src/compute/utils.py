import pandas as pd
from src.compute.constants import *


def compute_correlation(df: pd.DataFrame, method_name: str, task: str, task_performance_path: str=None):
    """Compute the correlation between the lens score and the task performance.
    Args:
        df (pd.DataFrame): The dataframe containing the lens scores and the task performance.
        method_name (str): The method name. You can choose from "pad", "mmd", "mdm", "mauve" or debiased_lens.
        task (str): The task name. You can choose from "sentiment_analysis", "text2sql", "web_agent", "image_classification".
        task_performance_path (str): The path to the task performance csv file, including the task performance for each dataset.
    """
    from scipy.stats import pearsonr, spearmanr
    # We'll collect results in a list of dicts
    corr_rows = []
    task_performance_path = TASK_PERFORMANCE_PATHS[task] if task_performance_path is None else task_performance_path
    merge_keys = MERGE_KEYS[task]
    df_task_performance = pd.read_csv(task_performance_path)
    df = pd.merge(df, df_task_performance, on=merge_keys, how='left')
    if task == "sentiment_analysis":
        # For each seed
        for seed in df['seed'].unique():
            seed_df = df[df['seed'] == seed]
            # If there is only one dataset_name, correlation is not defined
            if len(seed_df['dataset_name'].unique()) < 2:
                continue
            # For each dataset, get score, debiased_score, test_f1_mean
            # We want to correlate score vs test_f1_mean, debiased_score vs test_f1_mean across datasets
            # So, group by dataset and take the mean (should be one row per dataset per seed)
            grouped = seed_df.groupby('dataset_name').agg({
                f'{method_name}_score': 'mean',
                'test_f1_mean': 'mean'
            }).reset_index()
            if len(grouped) < 2:
                continue
            try:
                p_score, _ = pearsonr(grouped[f'{method_name}_score'], grouped['test_f1_mean'])
                s_score, _ = spearmanr(grouped[f'{method_name}_score'], grouped['test_f1_mean'])
            except Exception:
                p_score, s_score = None, None
            corr_rows.append({
                'seed': seed,
                f'pearson_{method_name}_score_vs_f1': p_score,
                f'spearman_{method_name}_score_vs_f1': s_score
            })
        corr_df = pd.DataFrame(corr_rows)
        # Aggregate all seeds for each metric and add a row named "all_seeds"
        if not corr_df.empty:
            agg_row = {
                'pearson_score_vs_f1': corr_df[f'pearson_{method_name}_score_vs_f1'].mean(),
                'spearman_score_vs_f1': corr_df[f'spearman_{method_name}_score_vs_f1'].mean()
            }
            corr_df_agg = pd.DataFrame([agg_row]).round(4)
        else:
            corr_df_agg = corr_df
        print(corr_df_agg.to_string(index=False))
    elif task == "text2sql":
        # For text2sql, we correlate score vs accuracy, and debiased_score vs accuracy
        # There are multiple db_id values, so we compute per-db correlations, then aggregate
        for db_id in df['db_id'].unique():
            db_df = df[df['db_id'] == db_id]
            # For each seed, compute correlation across datasets (for this db_id)
            pearson_score, spearman_score = [], []
            for seed in db_df['seed'].unique():
                seed_df = db_df[db_df['seed'] == seed]
                # If there is only one dataset, skip
                if len(seed_df['dataset_name'].unique()) < 2:
                    continue
                # For each dataset, get score, debiased_score, accuracy
                grouped = seed_df.groupby('dataset_name').agg({
                    f'{method_name}_score': 'mean',
                    'accuracy': 'mean'
                }).reset_index()
                # Remove rows with missing accuracy
                grouped = grouped.dropna(subset=['accuracy'])
                if len(grouped) < 2:
                    continue
                try:
                    p_score, _ = pearsonr(grouped[f'{method_name}_score'], grouped['accuracy'])
                    s_score, _ = spearmanr(grouped[f'{method_name}_score'], grouped['accuracy'])
                except Exception:
                    p_score, s_score = None, None
                pearson_score.append(p_score)
                spearman_score.append(s_score)
            def safe_mean(x):
                x = [i for i in x if i is not None]
                return sum(x)/len(x) if len(x) > 0 else None
            corr_rows.append({
                'db_id': db_id,
                'pearson_score_vs_acc': safe_mean(pearson_score),
                'spearman_score_vs_acc': safe_mean(spearman_score),
            })
        # After per-db corr, add a row aggregating all db_ids ("all_db")
        corr_df = pd.DataFrame(corr_rows)
        if not corr_df.empty:
            agg_row = {
                'db_id': 'all_db',
                'pearson_score_vs_acc': corr_df['pearson_score_vs_acc'].mean(),
                'spearman_score_vs_acc': corr_df['spearman_score_vs_acc'].mean(),
            }
            corr_df_agg = pd.concat([corr_df, pd.DataFrame([agg_row])], ignore_index=True).round(4)
        else:
            corr_df_agg = corr_df
        print(corr_df_agg.to_string(index=False))
    elif task == "web":
        corr_rows = []
        for website in df['website'].unique():
            website_df = df[df['website'] == website]
            pearson_score, spearman_score = [], []
            for seed in website_df['seed'].unique():
                seed_df = website_df[website_df['seed'] == seed]
                if len(seed_df['partition'].unique()) < 2:
                    continue
                grouped = seed_df.groupby('partition').agg({
                    f'{method_name}_score': 'mean',
                    'success_rate': 'mean'
                }).reset_index()
                grouped = grouped.dropna(subset=['success_rate'])
                if len(grouped) < 2:
                    continue
                try:
                    p_score, _ = pearsonr(grouped[f'{method_name}_score'], grouped['success_rate'])
                    s_score, _ = spearmanr(grouped[f'{method_name}_score'], grouped['success_rate'])
                except Exception:
                    p_score, s_score = None, None
                pearson_score.append(p_score)
                spearman_score.append(s_score)
            def safe_mean(x):
                x = [i for i in x if i is not None]
                return sum(x)/len(x) if len(x) > 0 else None
            corr_rows.append({
                'website': website,
                'pearson_score_vs_success_rate': safe_mean(pearson_score),
                'spearman_score_vs_success_rate': safe_mean(spearman_score),
            })
        corr_df = pd.DataFrame(corr_rows)
        if not corr_df.empty:
            agg_row = {
                'website': 'all_websites',
                'pearson_score_vs_success_rate': corr_df['pearson_score_vs_success_rate'].mean(),
                'spearman_score_vs_success_rate': corr_df['spearman_score_vs_success_rate'].mean(),
            }
            corr_df_agg = pd.DataFrame([agg_row]).round(4)
        else:
            corr_df_agg = corr_df
        print(corr_df_agg.to_string(index=False))
    elif task == "image":
        corr_rows = []
        corr_rows = []
        for split in SPLITS:
            split_df = df[df['split'] == split]
            pearson_score, spearman_score = [], []
            for seed in split_df['seed'].unique():
                seed_df = split_df[split_df['seed'] == seed]
                if len(seed_df['dataset_name'].unique()) < 2:
                    continue
                grouped = seed_df.groupby('dataset_name').agg({
                    f'{method_name}_score': 'mean',
                    'test_mrr': 'mean'
                }).reset_index()
                grouped = grouped.dropna(subset=['test_mrr'])
                if len(grouped) < 2:
                    continue
                try:
                    p_score, _ = pearsonr(grouped[f'{method_name}_score'], grouped['test_mrr'])
                    s_score, _ = spearmanr(grouped[f'{method_name}_score'], grouped['test_mrr'])
                except Exception:
                    p_score, s_score = None, None
                pearson_score.append(p_score)
                spearman_score.append(s_score)
            def safe_mean(x):
                x = [i for i in x if i is not None]
                return sum(x)/len(x) if len(x) > 0 else None
            corr_rows.append({
                'split': split,
                'pearson_score_vs_mrr': safe_mean(pearson_score),
                'spearman_score_vs_mrr': safe_mean(spearman_score),
            })
        corr_df = pd.DataFrame(corr_rows)
        if not corr_df.empty:
            agg_row = {
                'split': 'all_splits',
                'pearson_score_vs_mrr': corr_df['pearson_score_vs_mrr'].mean(),
                'spearman_score_vs_mrr': corr_df['spearman_score_vs_mrr'].mean(),
            }
            corr_df_agg = pd.DataFrame([agg_row]).round(4)
        else:
            corr_df_agg = corr_df
        print(corr_df_agg.to_string(index=False))

    return corr_df_agg

