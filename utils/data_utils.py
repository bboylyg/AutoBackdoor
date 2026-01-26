import numpy as np
import json, re
from tqdm import tqdm
import os, sys

def load_and_sample_data(data_file, sample_size=1.0):
    with open(data_file) as f:
        examples = json.load(f)

    print("First 5 examples before sampling:")
    for example in examples[:5]:
        print(example)

    if sample_size <= 1.0:
        sampled_indices = np.random.choice(len(examples), int(len(examples) * sample_size), replace=False)
        examples = [examples[i] for i in sampled_indices]

    print(f"Number of examples after sampling: {len(examples)}")
    return examples