TASK_PERFORMANCE_PATHS = {
    "sentiment_analysis": "./data/sentiment_analysis/task_performance/sentiment_score.csv",
    "text2sql": "./data/text2sql/task_performance/text2sql_score.csv",
    "web": "./data/web_agent/task_performance/webvoyager_scores.csv",
}

MERGE_KEYS = {
    "sentiment_analysis": ["dataset_name"],
    "text2sql": ["dataset_name", "db_id"],
    "web": ["website", "partition"]
}