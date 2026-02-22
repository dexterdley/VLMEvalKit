import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])

def extract_label_and_color(file_path):
    if "2B" in file_path:
        model = "2B"
        c_reg, c_vgd = '#99ccff', '#0066cc'  # Light/Dark Blue
    elif "4B" in file_path:
        model = "4B"
        c_reg, c_vgd = '#99ff99', '#009900'  # Light/Dark Green
    elif "8B" in file_path:
        model = "8B"
        c_reg, c_vgd = '#ff9999', '#cc0000'  # Light/Dark Red
    else:
        model = "Unknown"
        c_reg, c_vgd = '#cccccc', '#666666'

    # Seed-agnostic matching
    if "_2.0" in file_path:
        return f"{model} VGD (Ours)", c_vgd 
    else:
        return f"{model} Regular", c_reg

def process_file(file_path, n_values=[3, 5, 7, 10, 12]):
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        label, color = extract_label_and_color(file_path)
        return label, {n: 0.01 for n in n_values}, 0.0, color

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, sep='\t')

    col_name = 'prediction' if 'prediction' in df.columns else 'pred'
    predictions = df[col_name].tolist()
    
    rep_rates_dict = {n: [] for n in n_values}
    word_counts = []
    
    for text in predictions:
        if not isinstance(text, str):
            continue
            
        tokens = re.findall(r'\b\w+\b', text.lower())
        word_count = len(tokens)
        word_counts.append(word_count)
        
        for n in n_values:
            if word_count < n:
                rep_rates_dict[n].append(0.0)
                continue
                
            n_grams = list(get_ngrams(tokens, n))
            total_ngrams = len(n_grams)
            unique_ngrams = len(set(n_grams))
            
            rep_rate = 1.0 - (unique_ngrams / total_ngrams)
            rep_rates_dict[n].append(rep_rate)

    mean_rep_rates = {n: np.mean(rep_rates_dict[n]) * 100 for n in n_values}
    avg_length = np.mean(word_counts) if word_counts else 0.0
    
    label, color = extract_label_and_color(file_path)
    return label, mean_rep_rates, avg_length, color

def plot_metrics(files, n_values=[3, 5, 7, 10, 12]):
    print(f"Processing {len(files)} files...")
    
    # 1. Group data by Label (Model Size + Method) across seeds
    grouped_results = {}
    ordered_labels = [] # To preserve the order for plotting
    
    for file_path in files:
        label, mean_rep_rates, avg_length, color = process_file(file_path, n_values)
        if label not in grouped_results:
            grouped_results[label] = {
                'color': color,
                'rep_rates': {n: [] for n in n_values},
                'lengths': []
            }
            ordered_labels.append(label)
            
        for n in n_values:
            grouped_results[label]['rep_rates'][n].append(mean_rep_rates[n])
        if avg_length > 0:
            grouped_results[label]['lengths'].append(avg_length)

    # 2. Compute Mean and Std Dev across the collected seeds
    results = []
    for label in ordered_labels:
        data = grouped_results[label]
        
        agg_rep_mean = {n: np.mean(data['rep_rates'][n]) for n in n_values}
        agg_rep_std = {n: np.std(data['rep_rates'][n]) for n in n_values}
        
        agg_len_mean = np.mean(data['lengths']) if data['lengths'] else 0.0
        agg_len_std = np.std(data['lengths']) if data['lengths'] else 0.0
        
        print(f"Group: {label:<15} | Seeds: {len(data['lengths'])} | Avg Length: {agg_len_mean:.1f} ± {agg_len_std:.1f}")
        results.append((label, agg_rep_mean, agg_rep_std, agg_len_mean, agg_len_std, data['color']))

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.2, 1]})
    x = np.arange(len(n_values))
    
    # ---------------------------------------------------------
    # 1. Grouped Bar Chart for N-gram Repetition (Log Scale)
    # ---------------------------------------------------------
    ax1 = axes[0]
    bar_width = 0.11
    relative_offsets = [-0.36, -0.25, -0.055, 0.055, 0.25, 0.36]
    
    for idx in range(len(n_values)):
        if idx % 2 != 0: 
            ax1.axvspan(idx - 0.5, idx + 0.5, color='yellow', alpha=0.2, zorder=0)

    for i, (label, rep_mean, rep_std, len_mean, len_std, color) in enumerate(results):
        offsets = x + relative_offsets[i]
        
        # Ensure values don't hit exactly 0 for log scale
        y_values = [max(rep_mean[n], 0.001) for n in n_values] 
        
        # Handle asymmetric error bars for logarithmic scale (prevent lower error from going below 0)
        y_err_lower = [err if y - err > 0.0001 else y - 0.0001 for y, err in zip(y_values, [rep_std[n] for n in n_values])]
        y_err_upper = [rep_std[n] for n in n_values]
        
        ax1.bar(offsets, y_values, width=bar_width, label=label, color=color, edgecolor='black', zorder=3,
                yerr=[y_err_lower, y_err_upper], capsize=3, error_kw={'linewidth': 1, 'alpha': 0.7})
        
    ax1.set_title("N-gram Repetition Rate by N-gram Size", fontweight='bold', fontsize=16)
    ax1.set_xlabel("N-gram Size (Words)", fontweight='bold', fontsize=14)
    ax1.set_ylabel("Mean Repetition Rate (%) - Log Scale", fontweight='bold', fontsize=14)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"n={n}" for n in n_values], fontsize=14, fontweight='bold')
    
    ax1.set_yscale('log')
    ax1.set_yticks([0.01, 0.1, 1, 10, 100])
    ax1.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    ax1.set_xlim(-0.5, len(n_values) - 0.5)
    
    ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax1.legend(loc='upper right', ncol=3, fontsize=12)

    # ---------------------------------------------------------
    # 2. Clustered Bar Chart for Average Output Length
    # ---------------------------------------------------------
    ax2 = axes[1]
    
    x_pos_lengths = [0, 1,   3, 4,   6, 7] 
    labels = [r[0] for r in results]
    lengths = [r[3] for r in results]
    length_stds = [r[4] for r in results]
    colors = [r[5] for r in results]
    
    bars = ax2.bar(x_pos_lengths, lengths, color=colors, edgecolor='black', zorder=3, width=0.8,
                   yerr=length_stds, capsize=5, error_kw={'linewidth': 1.5, 'alpha': 0.7})
    
    ax2.set_title("Average Output Length", fontweight='bold', fontsize=16)
    ax2.set_ylabel("Words per Answer", fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos_lengths)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    ax2.set_ylim(0, max(lengths) * 1.30) # Gave slight extra room for error bars
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6), # Pushed slightly higher so it clears the error bar 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=12)

    for i in range(0, len(bars), 2):
        reg_bar = bars[i]
        vgd_bar = bars[i+1]
        
        reg_height = reg_bar.get_height()
        vgd_height = vgd_bar.get_height()
        
        if reg_height > 0 and vgd_height > 0:
            diff = reg_height - vgd_height
            pct_drop = (diff / reg_height) * 100
            
            x1 = reg_bar.get_x() + reg_bar.get_width() / 2
            x2 = vgd_bar.get_x() + vgd_bar.get_width() / 2
            
            y_start = max(reg_height, vgd_height) + (max(lengths) * 0.14)
            y_end = vgd_height + (max(lengths) * 0.10)
            
            ax2.annotate('', xy=(x2, y_end), xytext=(x1, y_start),
                         arrowprops=dict(arrowstyle="->", color='red', lw=2, shrinkA=0, shrinkB=0))
            
            mid_x = (x1 + x2) / 2
            ax2.text(mid_x, y_start + (max(lengths) * 0.02), f"-{pct_drop:.0f}%", 
                     ha='center', va='bottom', color='red', fontweight='bold', fontsize=12)

    plt.tight_layout()
    figure_name = "vgd_repetition_analysis.pdf"
    plt.savefig(figure_name, dpi=100, bbox_inches='tight')
    print(f"Plot saved as {figure_name}")

if __name__ == "__main__":
    input_files = [
        "./outputs/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_Reasoning_42_0/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_Reasoning_42_2.0/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_Reasoning_42_0/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_Reasoning_42_2.0/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_Reasoning_42_0/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_Reasoning_42_2.0/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_MMStar.xlsx",

        "./outputs/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_Reasoning_55_0/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_Reasoning_55_2.0/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_Reasoning_55_0/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_Reasoning_55_2.0/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_Reasoning_55_0/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_Reasoning_55_2.0/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_MMStar.xlsx",

        "./outputs/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_Reasoning_69_0/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_Reasoning_69_2.0/Qwen3-VL-2B-Thinking/Qwen3-VL-2B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_Reasoning_69_0/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_Reasoning_69_2.0/Qwen3-VL-4B-Thinking/Qwen3-VL-4B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_Reasoning_69_0/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_MMStar.xlsx",
        "./outputs/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_Reasoning_69_2.0/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking_MMStar.xlsx"
    ]
    plot_metrics(input_files)