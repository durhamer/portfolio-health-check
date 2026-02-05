import streamlit as st
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="持倉相關性健檢", layout="centered")
st.title("🔥 持倉相關性熱力圖 (Risk Heatmap)")

st.markdown("""
**為什麼要看這個？**
如果你買了 10 支股票，結果它們全都是「深紅色」連動，那你其實只買了 1 支股票，只是開了 10 倍槓桿。
真正分散風險的投資組合，應該要有**藍色**（負相關）或**淺色**（低相關）的區塊。
""")

# --- 側邊欄輸入 ---
with st.sidebar:
    st.header("設定")
    # 預設放入常見的科技股 + 對沖資產(如公債TLT, 黃金GLD)
    default_tickers = "NVDA TSLA AVGO MSFT GOOG MRVL ORCL AAPL TSM"
    tickers_input = st.text_area("輸入股票代碼 (空白鍵分隔)", value=default_tickers, height=100)
    
    days_back = st.slider("回溯天數 (觀察多久的相關性?)", 30, 1095, 365)
    st.info("建議加入 TLT (美債) 或 GLD (黃金) 來觀察避險效果。")

# --- 主邏輯 ---
if tickers_input:
    tickers = tickers_input.split()
    
    # 1. 抓取資料
    start_date = datetime.now() - timedelta(days=days_back)
    
    with st.spinner(f'正在分析 {len(tickers)} 檔資產過去 {days_back} 天的連動性...'):
        try:
            # 只抓收盤價
            data = yf.download(tickers, start=start_date, progress=False)['Close']
            
            # 2. 關鍵：計算「每日漲跌幅」 (Log return 或 % change)
            # 我們要看的是「今天A漲，B是不是也跟著漲」，而不是看股價絕對值
            returns = data.pct_change().dropna()
            
            # 3. 計算相關係數矩陣
            corr_matrix = returns.corr()

            # --- 視覺化呈現 ---
            
            # (A) 熱力圖
            st.subheader("📊 相關性矩陣")
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 使用 Seaborn 畫熱力圖
            sns.heatmap(
                corr_matrix, 
                annot=True,         # 格子裡顯示數字
                cmap='coolwarm',    # 紅(正相關)-白(無關)-藍(負相關)
                vmin=-1, vmax=1,    # 固定範圍 -1 到 1
                center=0,
                square=True,
                fmt='.2f',
                linewidths=.5,
                cbar_kws={"shrink": .5}
            )
            st.pyplot(fig)

            # (B) 風險診斷報告
            st.subheader("🩺 診斷報告")
            
            # 算出除了自己以外的平均相關係數
            avg_corr = (corr_matrix.sum().sum() - len(corr_matrix)) / (len(corr_matrix)**2 - len(corr_matrix))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("平均連動係數", f"{avg_corr:.2f}")
            
            with col2:
                if avg_corr > 0.6:
                    st.error("🚨 **高風險**：你的持倉高度連動！這不是分散投資，這是「團進團出」。")
                elif avg_corr > 0.3:
                    st.warning("⚠️ **中風險**：有一定的連動性，建議加入一些防禦性資產。")
                else:
                    st.success("✅ **健康**：你的資產彼此獨立性高，分散效果良好。")

        except Exception as e:
            st.error(f"發生錯誤：{e} (可能是代碼輸入錯誤或找不到資料)")

else:
    st.info("👈 請在左側輸入股票代碼開始分析")
