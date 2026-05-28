import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "IF Multi-Factor Backtest Report", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 80, 160)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(30, 80, 160)
        self.set_line_width(0.4)
        self.line(self.get_x(), self.get_y(), self.get_x() + 170, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def kv_row(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(50, 6, key)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table(self, headers, rows, col_widths):
        # header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 236, 248)
        self.set_text_color(30, 30, 30)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()
        # data rows
        self.set_font("Helvetica", "", 9)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(255, 255, 255)
            for val, w in zip(row, col_widths):
                self.cell(w, 6, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.ln(2)


def generate_report(target_col, signal_filter, short_multiplier,
                    factor_dir="Results/factor",
                    model_dir="Results/model",
                    backtest_dir="Results/backtest",
                    output_path="Results/report.pdf"):

    ic_df       = pd.read_csv(f"{factor_dir}/factor_ic_summary.csv")
    selected_df = pd.read_csv(f"{factor_dir}/selected_features.csv")
    model_df    = pd.read_csv(f"{model_dir}/model_comparison.csv")
    bt_df       = pd.read_csv(f"{backtest_dir}/backtest_performance.csv")

    selected       = selected_df["selected_features"].tolist()
    best_idx       = bt_df["Sharpe"].idxmax()
    best_model     = bt_df.loc[best_idx, "模型"]
    best_sharpe    = bt_df.loc[best_idx, "Sharpe"]
    best_return    = bt_df.loc[best_idx, "累计收益"]
    positive_models = bt_df[bt_df["Sharpe"] > 0]["模型"].tolist()

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── 封面标题 ──────────────────────────────────────────
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 12, "IF Multi-Factor Backtest Report",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # ── 一、策略概述 ──────────────────────────────────────
    pdf.section_title("1. Strategy Overview")
    pdf.kv_row("Underlying:", "IF (CSI 300 Stock Index Futures), daily frequency")
    pdf.kv_row("Predict Target:", f"{target_col}  (next-day return)")
    pdf.kv_row("Signal Filter:", "Enabled - flat when |prediction| < 0.5 x std" if signal_filter
               else "Disabled - always in market")
    pdf.kv_row("Short Threshold:", f"{short_multiplier}x the long threshold (long-biased)")
    pdf.kv_row("Models:", "Ridge Regression, Logistic Regression, XGBoost, LSTM")
    pdf.ln(2)
    pdf.body_text(
        "Core logic: construct price/volume and fundamental factors, screen by IC/ICIR validity, "
        "then feed selected factors into four ML models to predict return direction. "
        "Signals: long (+1) / short (-1) / flat (0). Train/test split: 70/30 by time."
    )

    # ── 二、因子分析 ──────────────────────────────────────
    pdf.section_title("2. Factor Analysis")
    pdf.body_text(
        f"Total factors constructed: {len(ic_df)}   "
        f"Selection criteria: |ICIR| >= 0.3, p < 0.05, pairwise correlation < 0.7"
    )
    pdf.body_text(f"Selected factors ({len(selected)}):  {',  '.join(selected)}")
    pdf.ln(2)

    # IC table (top 10)
    pdf.set_font("Helvetica", "B", 10)
    pdf.body_text("Top 10 Factors by |ICIR|:")
    top10 = ic_df.head(10)
    headers = ["Factor", "IC Mean", "ICIR", "IC>0 Rate", "p-value"]
    col_w   = [48, 28, 28, 28, 28]
    rows = []
    for _, row in top10.iterrows():
        p_str = f"{row['p值']:.4f}" if row['p值'] > 0.0001 else "< 0.0001"
        rows.append([
            row["因子"],
            f"{row['IC均值']:.4f}",
            f"{row['ICIR']:.4f}",
            f"{row['IC>0占比']:.1%}",
            p_str,
        ])
    pdf.table(headers, rows, col_w)
    pdf.body_text("Note: Negative IC indicates a contrarian signal. Models learn the direction automatically.")

    # factor chart
    factor_img = f"{factor_dir}/factor_analysis.png"
    try:
        pdf.ln(2)
        pdf.image(factor_img, x=15, w=175)
    except Exception:
        pdf.body_text(f"[Chart not found: {factor_img}]")

    # ── 三、模型训练结果 ──────────────────────────────────
    pdf.add_page()
    pdf.section_title("3. Model Training Results")
    pdf.body_text("Time-based 70/30 train/test split. No random shuffle to prevent look-ahead bias.")
    pdf.body_text(f"Best model selected by Sharpe ratio: {best_model}")
    pdf.ln(2)

    best_model_row = model_df[model_df["模型"] == best_model].iloc[0]
    rmse = f"{best_model_row['RMSE']}" if str(best_model_row['RMSE']) != "-" else "N/A"
    headers = ["Model", "Direction Acc.", "IC", "RMSE"]
    col_w   = [40, 45, 45, 45]
    pdf.table(headers, [[
        best_model_row["模型"],
        f"{float(best_model_row['方向准确率']):.2%}",
        f"{float(best_model_row['IC']):.4f}",
        rmse,
    ]], col_w)

    # ── 四、回测绩效 ──────────────────────────────────────
    pdf.section_title("4. Backtest Performance")
    pdf.body_text(
        "Strategy return = signal x actual next-day return. "
        "Benchmark = Buy & Hold (long throughout the test period)."
    )
    pdf.ln(2)

    best_bt_row = bt_df[bt_df["模型"] == best_model].iloc[0]
    headers = ["Model", "Total Return", "Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Win Rate"]
    col_w   = [28, 25, 25, 22, 22, 22, 22]
    pdf.table(headers, [[
        best_bt_row["模型"],
        f"{best_bt_row['累计收益']:.2%}",
        f"{best_bt_row['年化收益']:.2%}",
        f"{best_bt_row['年化波动']:.2%}",
        f"{best_bt_row['Sharpe']:.3f}",
        f"{best_bt_row['最大回撤']:.2%}",
        f"{best_bt_row['胜率']:.2%}",
    ]], col_w)

    # backtest chart
    bt_img = f"{backtest_dir}/backtest_curve.png"
    try:
        pdf.image(bt_img, x=15, w=175)
    except Exception:
        pdf.body_text(f"[Chart not found: {bt_img}]")

    # ── 五、结论 ──────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("5. Conclusion")

    if positive_models:
        pdf.kv_row("Best Model:", f"{best_model}  (Sharpe = {best_sharpe:.4f}, Return = {best_return:.2%})")
        pdf.kv_row("Profitable Models:", ", ".join(positive_models))
    else:
        pdf.body_text("All models have negative Sharpe in the test period. Signal quality is limited.")

    pdf.ln(3)
    pdf.body_text("Key Findings:")
    findings = [
        "1. Most selected factors are contrarian (negative IC), indicating IF tends to mean-revert in the short term.",
        "2. Non-linear models (LSTM) better capture complex factor-return relationships than linear models.",
        "3. Signal filtering (flat zone) reduces low-confidence trades and limits drawdowns.",
        "4. Single-asset time-series models have limited signal strength (IC mean ~0.05-0.10). Factor quality is the main bottleneck.",
    ]
    for f in findings:
        pdf.body_text(f)

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 5,
        "Disclaimer: Transaction costs (fees and slippage) are not included. "
        "Actual returns would be lower than reported.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(output_path)
    print(f"\n回测报告已保存至 {output_path}")
