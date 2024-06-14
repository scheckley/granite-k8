from transformers import pipeline
from datasets import Dataset
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
tf_dataset = Dataset.from_pandas(data)
sentiment_pipeline(tf_dataset)
