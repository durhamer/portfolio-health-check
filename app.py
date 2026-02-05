import streamlit as st
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="持倉體檢報告", layout="centered")
st.title("🏥 投資組合健康檢查")
st.markdown("""
此工具幫你分析兩件事：
1. **連動風險 (Correlation)**：你的股票是不是「同進同出」？(紅色=危險)
2. **波動風險 (Beta)**：你的股票是不是「心臟病製造機」？(數值越高越晃)
""")

# --- 2. 輔助函數：計算 Beta ---
def calculate_beta(stock_returns, market_returns):
    # 公式：共變異數 / 市場變異數
    if len(stock_returns) < 2: return 0.0 # 避免資料不足報錯
    covariance = stock_returns.cov(market_returns)
    variance = market_returns.var()
    if variance == 0: return 0.0
    return covariance / variance

# --- 3. 側邊欄輸入 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    
    # 預設清單：包含熱門科技股、避險資產
    default_tickers = "NVDA TSLA AVGO MSFT GOOG AMZN LMT TLT GLD"
    tickers_input = st.text_area("輸入持倉代碼 (用空白鍵分隔)", value=default_tickers, height=100)
    
    days_back = st.slider("回溯天數 (過去幾天的表現?)", 30, 1095, 365)
    st.info("💡 程式會自動加入 **SPY (標普500)** 作為計算 Beta 的基準，你不用手動輸入。")

# --- 4. 主程式邏輯 ---
if tickers_input:
    # 整理使用者輸入的股票清單 (移除重複、轉大寫)
    user_tickers = [t.upper() for t in tickers_input.split()]
    
    # 確保抓取清單中有 SPY (大盤基準)，但如果使用者沒輸入，我們後面顯示時要藏起來
    fetch_tickers = list(set(user_tickers + ["SPY"]))
    
    start_date = datetime.now() - timedelta(days=days_back)
    
    with st.spinner(f'正在分析 {len(fetch_tickers)} 檔資產... 請稍候 ☕'):
        try:
            # 4.1 抓取資料
            data = yf.download(fetch_tickers, start=start_date, progress=False)['Close']
            
            # 4.2 計算每日漲跌幅
            returns = data.pct_change().dropna()
            
            # 檢查是否抓取成功
            if returns.empty:
                st.error("❌ 抓不到資料，請檢查股票代碼是否正確。")
                st.stop()

            # --- 第一部分：Beta 波動率分析 ---
            st.divider()
            st.subheader("⚡ 波動率分析 (Beta Coefficient)")
            st.markdown("基準：S&P 500 大盤 (SPY) = **1.0**")
            
            # 分離大盤與個股數據
            if 'SPY' in returns.columns:
                market_returns = returns['SPY']
                
                # 計算使用者清單中每一支股票的 Beta
                beta_dict = {}
                for ticker in user_tickers:
                    if ticker in returns.columns and ticker != 'SPY':
                        beta_val = calculate_beta(returns[ticker], market_returns)
                        beta_dict[ticker] = beta_val
                
                # 轉成圖表數據
                if beta_dict:
                    beta_df = pd.Series(beta_dict, name="Beta").sort_values(ascending=False)
                    
                    # 畫長條圖
                    st.bar_chart(beta_df, color="#FF4B4B") # 紅色代表波動
                    
                    # 找出波動王
                    max_stock = beta_df.idxmax()
                    max_val = beta_df.max()
                    st.caption(f"😱 你的「波動之王」是 **{max_stock}** (Beta: {max_val:.2f})。大盤跌 1%，它通常會跌 {max_val:.2f}%。")
                else:
                    st.warning("⚠️ 無法計算 Beta，請確認代碼正確。")
            else:
                st.error("⚠️ 無法抓取 SPY 數據，跳過 Beta 分析。")

            # --- 第二部分：相關性熱力圖 ---
            st.divider()
            st.subheader("🔥 相關性熱力圖 (Correlation Heatmap)")
            st.markdown("🟥 **深紅** = 風險疊加 (同漲同跌) | 🟦 **深藍** = 有效避險 (走勢相反)")
            
            # 只計算使用者輸入的股票 (不一定需要把 SPY 畫進去，除非使用者有打)
            # 這樣圖表比較乾淨，專注於持倉內部的風險
            portfolio_returns = returns[user_tickers]
            corr_matrix = portfolio_returns.corr()

            # 畫圖
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, vmax=1, 
                center=0,
                square=True,
                fmt='.2f',
                linewidths=.5,
                cbar_kws={"shrink": .5}
            )
            st.pyplot(fig)

            # --- 第三部分：總體診斷 ---
            st.subheader("🩺 診斷報告")
            
            # 計算平均相關係數 (扣除自己對自己)
            n = len(corr_matrix)
            if n > 1:
                avg_corr = (corr_matrix.sum().sum() - n) / (n*n - n)
            else:
                avg_corr = 1.0
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("平均連動係數", f"{avg_corr:.2f}")
            
            with col2:
                if avg_corr > 0.6:
                    st.error("🚨 **高風險**：持倉高度連動！這不是分散投資，這只是開了槓桿。")
                elif avg_corr > 0.3:
                    st.warning("⚠️ **中風險**：有一定的連動性，建議加入更多負相關資產(如公債/黃金)。")
                else:
                    st.success("✅ **健康**：你的資產彼此獨立性高，分散效果良好！")

        except Exception as e:
            st.error(f"發生錯誤：{e}")
            st.info("💡 小撇步：請檢查股票代碼是否有拼錯？或是剛剛上市沒有足夠的歷史資料？")

else:
    st.info("👈 請在左側輸入股票代碼開始分析")
