"""
FraudShield AI -- Auto-Generated PDF Report
Generates a professional PDF summary after pipeline completion.
"""

import os
import json
from datetime import datetime

from fpdf import FPDF


class FraudShieldReport(FPDF):
    """Custom PDF class with headers and footers."""

    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'FraudShield AI -- Pipeline Report', align='R')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def metric_row(self, label, value):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(80, 7, label)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_pdf_report(output_dir):
    """Generate the complete PDF report."""
    print("\n" + "=" * 70)
    print("GENERATING PDF REPORT")
    print("=" * 70)

    pdf = FraudShieldReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    results_dir = os.path.join(output_dir, "results")
    viz_dir = os.path.join(output_dir, "visualizations")

    # ── Page 1: Title + Executive Summary ──────────────────────────
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, 'FraudShield AI', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Adaptive Fraud Detection with Explainable Risk Intelligence', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 6, f'Report generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, 'FrostHack Financial Services Track', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    # Architecture summary
    pdf.section_title('System Architecture')
    pdf.body_text(
        'FraudShield AI is a multi-layered fraud detection system featuring '
        '17 detection layers, a 6-model stacking ensemble with Logistic Regression '
        'meta-learner, temporal graph analysis using NetworkX and Louvain community '
        'detection, SHAP-based explainability, Optuna hyperparameter tuning, '
        'target encoding, and adversarial robustness testing.'
    )
    pdf.body_text(
        'The system includes a FastAPI real-time scoring API, a 7-page Streamlit '
        'dashboard, and an ROI cost-benefit calculator for business impact analysis.'
    )

    # ── Page 2: Model Performance ──────────────────────────────────
    pdf.add_page()
    pdf.section_title('Model Performance')

    metrics_path = os.path.join(results_dir, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Table header
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(30, 60, 120)
        pdf.set_text_color(255, 255, 255)
        col_w = [45, 25, 25, 25, 25]
        headers = ['Model', 'AUC', 'F1', 'Precision', 'Recall']
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 8, h, border=1, fill=True, align='C')
        pdf.ln()

        # Table rows
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(40, 40, 40)
        for name, m in metrics.items():
            is_ensemble = name == 'ensemble'
            if is_ensemble:
                pdf.set_font('Helvetica', 'B', 9)
                pdf.set_fill_color(230, 240, 255)
            else:
                pdf.set_font('Helvetica', '', 9)
                pdf.set_fill_color(255, 255, 255)

            pdf.cell(col_w[0], 7, m.get('name', name)[:20], border=1, fill=is_ensemble)
            pdf.cell(col_w[1], 7, f"{m.get('auc', 0):.4f}", border=1, fill=is_ensemble, align='C')
            pdf.cell(col_w[2], 7, f"{m.get('f1', 0):.4f}", border=1, fill=is_ensemble, align='C')
            pdf.cell(col_w[3], 7, f"{m.get('precision', 0):.4f}", border=1, fill=is_ensemble, align='C')
            pdf.cell(col_w[4], 7, f"{m.get('recall', 0):.4f}", border=1, fill=is_ensemble, align='C')
            pdf.ln()

        # Best model highlight
        best_name = max(metrics.items(), key=lambda x: x[1].get('auc', 0))
        pdf.ln(6)
        pdf.body_text(f"Best Individual Model: {best_name[1]['name']} (AUC: {best_name[1]['auc']:.4f})")

        ensemble = metrics.get('ensemble', {})
        pdf.body_text(
            f"Stacking Ensemble: AUC {ensemble.get('auc', 0):.4f} | "
            f"F1 {ensemble.get('f1', 0):.4f} | "
            f"Precision {ensemble.get('precision', 0):.4f}"
        )

    # ROC Curves image
    roc_path = os.path.join(viz_dir, 'roc_curves.png')
    if os.path.exists(roc_path):
        pdf.ln(4)
        pdf.image(roc_path, x=15, w=180)

    # ── Page 3: Feature Importance + SHAP ──────────────────────────
    pdf.add_page()
    pdf.section_title('Feature Importance & Explainability')

    feat_path = os.path.join(viz_dir, 'feature_importance.png')
    if os.path.exists(feat_path):
        pdf.image(feat_path, x=15, w=180)

    shap_path = os.path.join(viz_dir, 'shap_bar.png')
    if os.path.exists(shap_path):
        pdf.add_page()
        pdf.section_title('SHAP Feature Importance (Mean |SHAP|)')
        pdf.image(shap_path, x=15, w=180)

    shap_summary = os.path.join(viz_dir, 'shap_summary.png')
    if os.path.exists(shap_summary):
        pdf.add_page()
        pdf.section_title('SHAP Beeswarm Summary')
        pdf.image(shap_summary, x=15, w=180)

    # ── Page: SHAP Waterfalls ──────────────────────────────────────
    for i in range(1, 4):
        wf_path = os.path.join(viz_dir, f'shap_waterfall_{i}.png')
        if os.path.exists(wf_path):
            pdf.add_page()
            pdf.section_title(f'SHAP Waterfall -- Transaction #{i}')
            pdf.body_text('Individual feature contributions for a high-risk transaction:')
            pdf.image(wf_path, x=15, w=180)

    # ── Page: Graph Analysis ───────────────────────────────────────
    graph_path = os.path.join(viz_dir, 'graph_analysis.png')
    if os.path.exists(graph_path):
        pdf.add_page()
        pdf.section_title('Graph-Based Fraud Ring Analysis')
        pdf.body_text(
            'Temporal graph analysis using NetworkX builds a relationship graph '
            'where nodes are users and edges connect users who share devices, '
            'addresses, or rare email domains. Louvain community detection '
            'identifies fraud rings. Features: PageRank, betweenness centrality, '
            'community fraud rate, fraud neighbor ratio.'
        )
        pdf.image(graph_path, x=10, w=190)

    # ── Page: Adversarial Robustness ───────────────────────────────
    adv_path = os.path.join(results_dir, 'adversarial_report.json')
    if os.path.exists(adv_path):
        pdf.add_page()
        pdf.section_title('Adversarial Robustness Testing')
        pdf.body_text(
            'FraudShield AI was tested against 5 realistic attack scenarios to '
            'ensure the model cannot be easily fooled by smart fraudsters.'
        )

        with open(adv_path, 'r') as f:
            adv = json.load(f)

        overall = adv.get('overall_score', 0)
        passed = adv.get('tests_passed', 0)
        total = adv.get('tests_total', 5)

        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(0, 150, 80) if overall >= 80 else pdf.set_text_color(200, 100, 0)
        pdf.cell(0, 12, f'Robustness Score: {overall:.0f}% ({passed}/{total} passed)', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        details = adv.get('details', {})
        for test_name, test_data in details.items():
            status = 'PASS' if test_data.get('passed') in [True, 'True'] else 'FAIL'
            value = test_data.get('value', 0)
            desc = test_data.get('description', '')

            pdf.set_font('Helvetica', 'B', 10)
            color = (0, 150, 80) if status == 'PASS' else (200, 50, 50)
            pdf.set_text_color(*color)
            pdf.cell(15, 7, f'[{status}]')
            pdf.set_text_color(40, 40, 40)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 7, f'{test_name}: {value:.1%} -- {desc}', new_x="LMARGIN", new_y="NEXT")

    # ── Page: ROI Analysis ─────────────────────────────────────────
    pdf.add_page()
    pdf.section_title('Business Impact -- ROI Analysis')
    pdf.body_text('Cost-benefit analysis for a mid-size bank (50M transactions/year):')

    if os.path.exists(metrics_path):
        ensemble = metrics.get('ensemble', {})
        recall = ensemble.get('recall', 0.65)
        precision = ensemble.get('precision', 0.80)

        annual_txns = 50_000_000
        fraud_rate = 0.035
        avg_loss = 850

        total_fraud = annual_txns * fraud_rate
        bl_missed = total_fraud * 0.70
        bl_loss = bl_missed * avg_loss + (annual_txns * 0.95 * 0.05 * 25)

        fs_caught = total_fraud * recall
        fs_missed = total_fraud * (1 - recall)
        fs_flagged = fs_caught / max(precision, 0.01)
        fs_fps = fs_flagged - fs_caught
        fs_loss = fs_missed * avg_loss + fs_fps * 25 + fs_flagged * 15

        savings = bl_loss - fs_loss

        pdf.metric_row('Annual Transactions:', f'{annual_txns:,}')
        pdf.metric_row('Fraud Rate:', f'{fraud_rate*100:.1f}%')
        pdf.metric_row('Fraud Detection Rate:', f'{recall*100:.1f}%')
        pdf.metric_row('Precision:', f'{precision*100:.1f}%')
        pdf.metric_row('Frauds Prevented/Year:', f'{fs_caught:,.0f}')
        pdf.metric_row('Fraud Losses Avoided:', f'${fs_caught * avg_loss:,.0f}')
        pdf.metric_row('Baseline Annual Loss:', f'${bl_loss:,.0f}')
        pdf.metric_row('With FraudShield:', f'${fs_loss:,.0f}')
        pdf.ln(4)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(0, 150, 80)
        pdf.cell(0, 10, f'Annual Savings: ${savings:,.0f}', new_x="LMARGIN", new_y="NEXT")

    # ── Page: Additional Charts ────────────────────────────────────
    extra_charts = [
        ('Confusion Matrices', 'confusion_matrices.png'),
        ('Metrics Comparison', 'metrics_comparison.png'),
        ('Risk Score Distribution', 'risk_distribution.png'),
    ]
    for title, fname in extra_charts:
        chart_path = os.path.join(viz_dir, fname)
        if os.path.exists(chart_path):
            pdf.add_page()
            pdf.section_title(title)
            pdf.image(chart_path, x=10, w=190)

    # ── Save PDF ───────────────────────────────────────────────────
    pdf_path = os.path.join(output_dir, "FraudShield_AI_Report.pdf")
    pdf.output(pdf_path)
    print(f"  [PDF] Report saved to {pdf_path}")
    print(f"  [PDF] {pdf.pages_count} pages generated")

    return pdf_path
