import streamlit as st
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # 物理學家的好朋友
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha 實驗室", layout="centered")
st.title("🧪 Alpha 投資組合實驗室")
st.markdown("""
這裡是用數據科學尋找「聖杯」的地方：
1. **相關性 (Correlation)**：資產是否同進同出？
2. **波動性 (Beta)**：資產是否太過刺激？
3. **效率前緣 (Efficient Frontier)**：用蒙地卡羅模擬找出「最強配置比例」。
""")

# --- 2. 輔助函數 ---
def calculate_beta(stock_returns, market_returns):
    if len(stock_returns) < 2: return 0.0
    covariance = stock_returns.cov(market_returns)
    variance = market_returns.var()
    if variance == 0: return 0.0
    return covariance / variance

# --- 3. 側邊欄輸入 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    default_tickers = "NVDA TSLA AVGO MSFT GOOG AMZN LMT TLT GLD"
    tickers_input = st.text_area("輸入持倉代碼", value=default_tickers, height=100)
    days_back = st.slider("回溯天數", 30, 1095, 365)
    
    st.divider()
    st.subheader("🎲 蒙地卡羅設定")
    num_portfolios = st.slider("模擬次數 (次)", 1000, 10000, 3000)
    st.caption("模擬次數越多，運算越久，但結果越精確。")

# --- 4. 主程式邏輯 ---
if tickers_input:
    user_tickers = [t.upper() for t in tickers_input.split()]
    
    # 為了算 Beta，我們強制加 SPY；但為了算效率前緣，我們只用使用者的股票
    fetch_tickers = list(set(user_tickers + ["SPY"]))
    
    start_date = datetime.now() - timedelta(days=days_back)
    
    with st.spinner('正在從量子領域下載數據...'):
        try:
            data = yf.download(fetch_tickers, start=start_date, progress=False)['Close']
            returns = data.pct_change().dropna()
            
            if returns.empty:
                st.error("❌ 數據不足")
                st.stop()

            # --- Tab 分頁設計 ---
            tab1, tab2, tab3 = st.tabs(["🎲 蒙地卡羅模擬 (新功能)", "⚡ 波動率 Beta", "🔥 相關性熱力圖"])

            # ==========================================
            # 功能 1: 蒙地卡羅模擬 (效率前緣)
            # ==========================================
            with tab1:
                st.subheader("🌌 效率前緣 (Efficient Frontier)")
                st.markdown("我們隨機嘗試了數千種持倉比例，尋找 **夏普比率 (Sharpe Ratio)** 最高的組合。")
                
                # 只取使用者的股票 (不含 SPY)
                sim_returns = returns[user_tickers]
                
                if len(user_tickers) < 2:
                    st.warning("⚠️ 至少需要兩支股票才能做資產配置模擬！")
                else:
                    # 準備矩陣運算
                    mean_returns = sim_returns.mean() * 252 # 年化報酬
                    cov_matrix = sim_returns.cov() * 252    # 年化共變異數
                    num_assets = len(user_tickers)
                    
                    # 建立容器存放模擬結果
                    results = np.zeros((3, num_portfolios)) # [報酬, 風險, Sharpe]
                    weight_array = [] # 存放權重
                    
                    # 開始蒙地卡羅模擬 (向量化加速版)
                    # 這裡用迴圈是因為我們要存下每一組權重，雖然可以全向量化但這樣寫比較好懂
                    for i in range(num_portfolios):
                        # 1. 生成隨機權重
                        weights = np.random.random(num_assets)
                        weights /= np.sum(weights) # 歸一化 (總和為1)
                        weight_array.append(weights)
                        
                        # 2. 計算預期報酬 (矩陣乘法)
                        portfolio_return = np.sum(mean_returns * weights)
                        
                        # 3. 計算預期風險 (標準差) -> 這是物理學裡的 "Error Propagation" 公式
                        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        
                        # 4. 存入結果
                        results[0,i] = portfolio_return
                        results[1,i] = portfolio_std_dev
                        results[2,i] = results[0,i] / results[1,i] # Sharpe Ratio (假設無風險利率為0)

                    # 找出最強組合 (夏普比率最大)
                    max_sharpe_idx = np.argmax(results[2])
                    sdp, rp = results[1,max_sharpe_idx], results[0,max_sharpe_idx]
                    optimal_weights = weight_array[max_sharpe_idx]

                    # 畫圖
                    fig_eff, ax_eff = plt.subplots(figsize=(10, 6))
                    # 散佈圖：顏色代表夏普比率
                    sc = ax_eff.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.5)
                    # 標記最強點
                    ax_eff.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')
                    
                    ax_eff.set_title('Monte Carlo Simulation')
                    ax_eff.set_xlabel('Volatility (Risk)')
                    ax_eff.set_ylabel('Expected Return')
                    plt.colorbar(sc, label='Sharpe Ratio')
                    ax_eff.legend()
                    st.pyplot(fig_eff)
                    
                    # 顯示最佳配置
                    st.success(f"🏆 最佳配置 (年化報酬: {rp*100:.1f}%, 風險: {sdp*100:.1f}%)")
                    
                    # 用 DataFrame 顯示權重
                    opt_df = pd.DataFrame({"資產": user_tickers, "建議權重": optimal_weights})
                    opt_df["建議權重"] = opt_df["建議權重"].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(opt_df.set_index("資產").T)

            # ==========================================
            # 功能 2: Beta 分析
            # ==========================================
            with tab2:
                st.subheader("⚡ 波動率分析 (Beta)")
                if 'SPY' in returns.columns:
                    market_returns = returns['SPY']
                    beta_dict = {}
                    for ticker in user_tickers:
                        if ticker in returns.columns:
                            beta_dict[ticker] = calculate_beta(returns[ticker], market_returns)
                    
                    beta_df = pd.Series(beta_dict).sort_values(ascending=False)
                    st.bar_chart(beta_df, color="#FF4B4B")
                    st.caption("基準：SPY = 1.0。數值越高代表波動越劇烈。")
                else:
                    st.warning("無法抓取 SPY，跳過 Beta 分析。")

            # ==========================================
            # 功能 3: 熱力圖
            # ==========================================
            with tab3:
                st.subheader("🔥 相關性矩陣")
                portfolio_returns = returns[user_tickers]
                corr_matrix = portfolio_returns.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"發生錯誤：{e}")

else:
    st.info("👈 請在左側輸入股票代碼開始實驗")
