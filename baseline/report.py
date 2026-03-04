import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_multiple_stats(file_list):
    all_data = []

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy file {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        
        stats = df.groupby('Stage')['Hit'].mean() * 100
        
        stats_df = stats.reset_index()
        
        stats_df['Model'] = os.path.basename(file_path).replace('.csv', '')
        
        all_data.append(stats_df)

    final_df = pd.concat(all_data, ignore_index=True)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=final_df, x='Stage', y='Hit', hue='Model')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10)

    plt.title('So sánh Hit Rate (%) giữa các phiên bản Agent', fontsize=15)
    plt.ylabel('Hit Rate (%)', fontsize=12)
    plt.xlabel('Stage (Giai đoạn)', fontsize=12)
    plt.ylim(0, 110) 
    plt.legend(title='Phiên bản', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig('hit_rate_comparison.png')
    plt.show()

files_to_compare = [
    "recommendation_stats_ARAG_init.csv",
    "recommendation_stats_ARAG.csv",
    "recommendation_stats_ARAG_GCN.csv",
    "recommendation_stats_ARAG_GCN_Retrie.csv"
]

plot_multiple_stats(files_to_compare)