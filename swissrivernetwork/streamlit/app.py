import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from huggingface_hub import hf_hub_download

# --- 关键步骤：把根目录加入路径，以便导入 src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# 现在可以导入 src 里的模块了
# from src.preprocessing import clean_river_data

# --- 页面配置 ---
st.set_page_config(page_title="Swiss River Temperature Modeling Research Platform", layout="wide")
st.title("🌊 Swiss River Temperature Modeling Research Platform (GitHub + HF)")

# --- 1. 模型加载逻辑 (核心) ---
@st.cache_resource
def load_model(repo_id=None, filename=None):
    """
    尝试从 Hugging Face Model Hub 下载模型。
    如果失败（比如还没上传），则返回 None，触发模拟模式。
    """
    if not repo_id:
        return None
        
    try:
        print(f"Downloading {filename} from HF: {repo_id} ...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(f"Can not load model from HF (Not Uploaded?): {e}")
        return None

# --- 这里填写你的 HF Model 仓库信息 ---
# 暂时留空或乱写，代码会自动处理
HF_REPO_ID = "your-username/river-temp-model" 
HF_MODEL_FILENAME = "model.pkl"

# 加载模型
model = load_model(HF_REPO_ID, HF_MODEL_FILENAME)

# --- 2. 侧边栏 ---
with st.sidebar:
    st.header("Control Panel")
    st.info("Current mode: " + ("🟢 Real model" if model else "🟡 Demo model"))
    
    uploaded_file = st.file_uploader("Upload river data (CSV)", type="csv")

# --- 3. 主逻辑 ---
if uploaded_file:
    # 读取数据
    raw_df = pd.read_csv(uploaded_file)
    
    # 调用 src 中的清洗函数 (证明同库调用成功)
    df = raw_df  # clean_river_data(raw_df)
    
    st.subheader("1. Data illustration")
    st.dataframe(df.head(), use_container_width=True)
    
    if st.button("Start prediction"):
        # 预测逻辑
        if model:
            # 真实预测
            preds = model.predict(df)
        else:
            # 模拟预测 (为了演示效果)
            st.warning("Use random number to mimic prediction ...")
            preds = np.random.normal(20, 2, len(df))
            
        df['wt_hat'] = preds
        
        # 可视化
        st.subheader("2. Visualize prediction results")
        import plotly.express as px
        # 假设第一列是时间
        fig = px.line(df, y='wt_hat', title="water temperature trend")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Upload CSV file from the left size.")
    st.markdown("""
    ### Architecture
    * **Code**: On GitHub
    * **Model**: Attempt to extract from Hugging Face Model Hub
    * **Computation**: Run on Hugging Face Spaces
    """)