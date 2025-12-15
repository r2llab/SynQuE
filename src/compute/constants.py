TASK_PERFORMANCE_PATHS = {
    "sentiment_analysis": "./data/sentiment_analysis/task_performance/sentiment_score.csv",
    "text2sql": "./data/text2sql/task_performance/text2sql_score.csv",
    "web": "./data/web_agent/task_performance/webvoyager_scores.csv",
    "image": "./data/image_classification/task_performance/imagenet_scores.csv"
}

MERGE_KEYS = {
    "sentiment_analysis": ["dataset_name"],
    "text2sql": ["dataset_name", "db_id"],
    "web": ["website", "partition"],
    "image": ["split", "dataset_name"]
}

SEEDS = [42, 43, 44, 45, 46]

# web navigation domain constants
WEBSITE_NAMES = [
    'allrecipes',
    'amazon',
    'apple',
    'arxiv',
    'bbc',
    'coursera',
    'dictionary.cambridge',
    'espn',
    'github',
    'google_maps',
    'google_search',
    'huggingface',
    'wolframalpha',
]
PARTITIONS = [0, 1, 2, 3, 4]

# image classification domain constants
SPLITS = ["first_split", "second_split", "third_split"]