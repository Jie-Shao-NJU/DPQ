
import os
import pickle
import matplotlib.pyplot as plt

with open("tmp/block_errors.pkl", "rb") as pkl_file:
    block_errors_all_models = pickle.load(pkl_file)

with open("tmp/indices.pkl", "rb") as pkl_file:
    indices = pickle.load(pkl_file)

for block_id in range(24, 25):
    plt.figure(figsize=(10, 6))
    for model_id, block_errors in enumerate(block_errors_all_models, start=1):
        errors = block_errors[block_id]
        plt.plot(indices[::-1], errors, label=f"Model {model_id} vs. Model 0")
    plt.xlabel("Step")
    plt.ylabel("MSE Error")
    plt.legend(loc='upper left', fontsize='small')
    plt.title(f"MSE Error Trend for Block {block_id} (Comparison Across Models)")
    plt.savefig(f"tmp/block_{block_id}_mse.png", bbox_inches='tight')
    plt.close()