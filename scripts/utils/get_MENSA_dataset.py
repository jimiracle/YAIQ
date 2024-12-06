from huggingface_hub import snapshot_download
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import io
import os

snapshot_download(repo_id="eduardtoni/MENSA-visual-iq-test", repo_type="dataset", local_dir="/home/work/yaiq/datasets/MENSA")

df = pd.read_parquet('/home/work/yaiq/datasets/MENSA/data/train-00000-of-00001-25c0febc1da39286.parquet')
norway_df = df[df['subset']=='MENSA Norway']

rows = []
default_path = '/home/work/yaiq/datasets/MENSA'

exclude_columns = ['question_img', 'correct_answer_img', 'choices_images', 'multiple_answer_img']

filtered_df = norway_df.drop(columns=exclude_columns)
filtered_df.to_csv('../csv/MENSA_Norway.csv', index=False)