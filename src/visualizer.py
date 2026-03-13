"""
FraudShield AI — Visualization Module
Generates all charts and graphs for the hackathon presentation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_theme(style="whitegrid", palette="husl")

COLORS = {
    'GREEN_APPROVE': '#2ecc71',
    'YELLOW_PIN_VERIFY': '#f1c40f',
    'ORANGE_BIOMETRIC': '#e67e22',
    'RED_BLOCK': '#e74c3c'
}


def generate_all_visualizations(results, importances, feature_names, output_df, df, output_dir):
    """Generate all visualizations for the hackathon."""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    plot_roc_curves(results, viz_dir)
    plot_confusion_matrices(results, viz_dir)
    plot_feature_importance(importances, feature_names, viz_dir)
    plot_risk_distribution(output_df, viz_dir)
    plot_risk_pie(output_df, viz_dir)
    plot_fraud_by_hour(df, viz_dir)
    plot_fraud_by_product(df, viz_dir)
    plot_amount_distribution(df, viz_dir)
    plot_sample_explanations(output_df, viz_dir)
    plot_metrics_comparison(results, viz_dir)

    print(f"\n[VIZ] All visualizations saved to {viz_dir}")


def plot_roc_curves(results, viz_dir):
    """ROC curves for all models."""
    print("  → ROC Curves...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#e67e22']
    for i, (model_name, res) in enumerate(results.items()):
        ax.plot(res['fpr'], res['tpr'],
                label=f"{res['name']} (AUC={res['auc']:.4f})",
                linewidth=2.5, color=colors[i % len(colors)])
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('FraudShield AI — ROC Curve Comparison', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'roc_curves.png'), bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results, viz_dir):
    """Confusion matrices for all models."""
    print("  → Confusion Matrices...")
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, res) in zip(axes, results.items()):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'])
        ax.set_title(f"{res['name']}\nF1={res['f1']:.3f}", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.suptitle('FraudShield AI — Confusion Matrices', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrices.png'), bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances, feature_names, viz_dir):
    """Top 20 feature importance bar chart."""
    print("  → Feature Importance...")
    
    avg_imp = importances.get('average', {})
    sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:20]
    
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Average Feature Importance')
    ax.set_title('FraudShield AI — Top 20 Most Important Features', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()


def plot_risk_distribution(output_df, viz_dir):
    """Risk score distribution histogram."""
    print("  → Risk Score Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Legit vs Fraud
    legit = output_df[output_df['isFraud_actual'] == 0]['risk_score']
    fraud = output_df[output_df['isFraud_actual'] == 1]['risk_score']
    
    ax.hist(legit, bins=50, alpha=0.6, label=f'Legitimate ({len(legit):,})', color='#2ecc71', density=True)
    ax.hist(fraud, bins=50, alpha=0.6, label=f'Fraudulent ({len(fraud):,})', color='#e74c3c', density=True)
    
    # Add tier boundaries
    for boundary, label, color in [(31, 'YELLOW', '#f1c40f'), (51, 'ORANGE', '#e67e22'), (71, 'RED', '#e74c3c')]:
        ax.axvline(x=boundary, color=color, linestyle='--', alpha=0.7, label=f'{label} threshold ({boundary})')
    
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Density')
    ax.set_title('FraudShield AI — Risk Score Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'risk_distribution.png'), bbox_inches='tight')
    plt.close()


def plot_risk_pie(output_df, viz_dir):
    """Risk category donut chart."""
    print("  → Risk Category Donut Chart...")
    fig, ax = plt.subplots(figsize=(9, 7))

    cats = output_df['risk_category'].value_counts()
    colors = [COLORS.get(c, '#bdc3c7') for c in cats.index]

    # Clean labels for legend
    legend_labels = [
        f"{c.replace('_', ' ')}  —  {v:,}  ({v/len(output_df)*100:.1f}%)"
        for c, v in cats.items()
    ]

    wedges, texts = ax.pie(
        cats.values, colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        pctdistance=0.78
    )

    # Add center text
    ax.text(0, 0, f"{len(output_df):,}\nTransactions",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#333')

    ax.legend(wedges, legend_labels,
              loc='lower center', bbox_to_anchor=(0.5, -0.12),
              fontsize=10, ncol=2, frameon=False)

    ax.set_title('FraudShield AI — Adaptive Authentication Distribution',
                 fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'risk_pie_chart.png'), bbox_inches='tight')
    plt.close()


def plot_fraud_by_hour(df, viz_dir):
    """Fraud rate by hour of day."""
    print("  → Fraud Rate by Hour...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hourly = df.groupby('hour_of_day')['isFraud'].mean() * 100
    colors = ['#e74c3c' if h <= 6 else '#3498db' for h in hourly.index]
    
    bars = ax.bar(hourly.index, hourly.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_title('FraudShield AI — Fraud Rate by Hour of Day\n(Red = Night hours, Blue = Day hours)', fontweight='bold')
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'fraud_by_hour.png'), bbox_inches='tight')
    plt.close()


def plot_fraud_by_product(df, viz_dir):
    """Fraud rate by ProductCD."""
    print("  → Fraud Rate by Product Code...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    product_fraud = df.groupby('ProductCD').agg(
        fraud_rate=('isFraud', 'mean'),
        count=('isFraud', 'count')
    ).reset_index()
    product_fraud['fraud_rate'] *= 100
    product_fraud = product_fraud.sort_values('fraud_rate', ascending=True)
    
    colors = plt.cm.RdYlGn_r(product_fraud['fraud_rate'].values / product_fraud['fraud_rate'].max())
    ax.barh(product_fraud['ProductCD'], product_fraud['fraud_rate'], color=colors, edgecolor='white')
    
    for i, (_, row) in enumerate(product_fraud.iterrows()):
        ax.text(row['fraud_rate'] + 0.2, i, f"{row['fraud_rate']:.2f}% ({row['count']:,} txns)",
                va='center', fontsize=10)
    
    ax.set_xlabel('Fraud Rate (%)')
    ax.set_title('FraudShield AI — Fraud Rate by Product Category', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'fraud_by_product.png'), bbox_inches='tight')
    plt.close()


def plot_amount_distribution(df, viz_dir):
    """Transaction amount distribution: fraud vs legit."""
    print("  → Amount Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    legit_amt = np.log1p(df[df['isFraud'] == 0]['TransactionAmt'])
    fraud_amt = np.log1p(df[df['isFraud'] == 1]['TransactionAmt'])
    
    ax.hist(legit_amt, bins=80, alpha=0.6, label='Legitimate', color='#2ecc71', density=True)
    ax.hist(fraud_amt, bins=80, alpha=0.6, label='Fraudulent', color='#e74c3c', density=True)
    
    ax.set_xlabel('Log(Transaction Amount + 1)')
    ax.set_ylabel('Density')
    ax.set_title('FraudShield AI — Transaction Amount Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'amount_distribution.png'), bbox_inches='tight')
    plt.close()


def plot_sample_explanations(output_df, viz_dir):
    """Table showing sample flagged transactions with explanations."""
    print("  → Sample Explanations Table...")
    
    # Get top RED and ORANGE flagged transactions
    flagged = output_df[output_df['risk_category'].isin(['RED_BLOCK', 'ORANGE_BIOMETRIC'])]
    sample = flagged.head(10) if len(flagged) >= 10 else flagged

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis('off')
    
    table_data = []
    for _, row in sample.iterrows():
        explanation = row['explanation'][:80] + '...' if len(str(row['explanation'])) > 80 else row['explanation']
        table_data.append([
            int(row['TransactionID']),
            f"${row['TransactionAmt']:.2f}",
            f"{row['risk_score']:.0f}",
            row['risk_category'].replace('_', ' '),
            explanation
        ])
    
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Transaction ID', 'Amount', 'Risk Score', 'Category', 'Explanation'],
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Color rows by risk category
        for i, row_data in enumerate(table_data):
            color = COLORS.get(row_data[3].replace(' ', '_'), '#ffffff')
            for j in range(len(row_data)):
                table[(i + 1, j)].set_facecolor(color + '40')  # Add transparency
    
    ax.set_title('FraudShield AI — Sample Flagged Transactions with Explanations',
                 fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sample_explanations.png'), bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results, viz_dir):
    """Bar chart comparing key metrics across all models."""
    print("  → Metrics Comparison...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    metrics = ['auc', 'f1', 'precision', 'recall']
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric.upper(), color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('FraudShield AI — Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([results[m]['name'] for m in models], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metrics_comparison.png'), bbox_inches='tight')
    plt.close()
