import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('block_statistics.csv')

def plot_trend_t(df, block, parameter, stat_type):
    filtered_df = df[(df['Block'] == block) & (df['Parameter'] == parameter)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['Time'][::-1], filtered_df[stat_type], label=f"{parameter} ({stat_type})")
    plt.xlabel('Time (Steps)')
    plt.ylabel(f"{stat_type.capitalize()} Value")
    plt.title(f"Trend of {parameter} {stat_type.capitalize()} for Block {block} over 1000 Steps")
    plt.legend()
    plt.savefig('stat.jpg')

plot_trend_t(df, block=15, parameter='gate_msa', stat_type='Std')
