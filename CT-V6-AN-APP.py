import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 사출 CT 분석 시스템", layout="centered")

# --- 2. AI 엔진 클래스 ---
class CT_Ensemble_Engine:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.m1 = Ridge(alpha=1.0)
        self.m2 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.m3 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
        self.is_ready = False
        self.cat_vars = ['MA', 'SZ', 'IN', 'TH', 'DP']

    def train(self, df):
        try:
            df.columns = [str(c).strip().upper() for c in df.columns]
            target_col = 'POINCT'
            past_nom_col = 'POMFCT'
            feature_cols = self.cat_vars + [past_nom_col]
            
            data = df[feature_cols + [target_col]].dropna()
            
            if len(data) < 2:
                return "학습 데이터가 너무 부족합니다. (최소 2행 이상 필요)"

            X = data[feature_cols]
            y = data[target_col]
            
            X_enc = X.copy()
            X_enc[self.cat_vars] = self.encoder.fit_transform(X[self.cat_vars].astype(str))
            
            self.m1.fit(X_enc, y)
            self.m2.fit(X_enc, y)
            self.m3.fit(X_enc, y)
            self.is_ready = True
            return "SUCCESS"
        except Exception as e:
            return f"학습 오류: 데이터 구조를 확인하세요. ({str(e)})"

    def predict(self, inputs):
        if not self.is_ready: return None
        df_in = pd.DataFrame([{
            'MA': inputs['MA'], 'SZ': inputs['SZ'], 'IN': inputs['IN'],
            'TH': inputs['TH'], 'DP': inputs['DP'], 'POMFCT': inputs['NOMFCT']
        }])
        df_in[self.cat_vars] = self.encoder.transform(df_in[self.cat_vars].astype(str))
        res = (self.m1.predict(df_in)[0] + self.m2.predict(df_in)[0] + self.m3.predict(df_in)[0]) / 3
        return res

# --- 3. 웹 UI ---
st.title("🏭 AI 사출 정밀 예상 CT 시스템")
st.write("엑셀 파일을 업로드하면 AI가 실측 데이터를 학습하여 정밀 CT를 예측합니다.")

uploaded_file = st.file_uploader("학습용 엑셀 파일을 선택하세요 (xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # 데이터 로드 (Past Data 시트, 헤더는 2행 기준)
        df_past = pd.read_excel(uploaded_file, sheet_name='Past Data', header=1)
        
        engine = CT_Ensemble_Engine()
        with st.spinner('데이터 분석 및 AI 학습 중...'):
            status = engine.train(df_past)

        if status == "SUCCESS":
            st.success(f"✅ 학습 완료! (총 {len(df_past)}개의 이력 데이터 활용)")
            
            st.divider()
            st.subheader("STEP 1. 현재 공정 조건 입력")
            
            col1, col2 = st.columns(2)
            with col1:
                ma_list = sorted([str(x).strip() for x in df_past['MA'].dropna().unique()])
                ma = st.selectbox("기계 사양 (MA)", ma_list)
                sz = st.selectbox("사이즈 (SZ)", ["S", "M", "L"])
                in_val = st.selectbox("인서트 여부 (IN)", ["IO", "IX"])
            
            with col2:
                th = st.selectbox("두께 (TH)", ["TS", "TM", "TL"])
                dp = st.selectbox("깊이 (DP)", ["DS", "DM", "DL"])
                nomfct = st.number_input("현재 성형 해석 CT (NOMFCT)", value=200.0, step=0.1)

            if st.button("AI 분석 실행 (NOPRECT)"):
                inputs = {'MA': ma, 'SZ': sz, 'IN': in_val, 'TH': th, 'DP': dp, 'NOMFCT': nomfct}
                result = engine.predict(inputs)
                
                if result:
                    st.divider()
                    st.subheader("STEP 2. AI 예측 결과 (NOPRECT)")
                    
                    gap = result - nomfct
                    # metric으로 깔끔하게 결과 표시
                    st.metric(label="최종 예상 CT (NOPRECT)", value=f"{result:.2f} s", delta=f"{gap:+.2f} s (보정치)")
                    st.info("이론치와 실제 데이터 사이의 오차를 보정한 최종 결과입니다.")
                    # 풍선 날라가는 코드(st.balloons) 삭제 완료!
        else:
            st.error(status)
            
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
else:
    st.info("엑셀 파일을 업로드해 주세요.")
