import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 사출 CT 정밀 분석", layout="centered")

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
            # 컬럼명 정리
            df.columns = [str(c).strip().upper() for c in df.columns]
            target_col = 'POINCT'   # 실측
            past_nom_col = 'POMFCT' # 과거 해석
            feature_cols = self.cat_vars + [past_nom_col]
            
            # 학습 데이터 준비
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]
            
            X_enc = X.copy()
            X_enc[self.cat_vars] = self.encoder.fit_transform(X[self.cat_vars].astype(str))
            
            # 앙상블 학습
            self.m1.fit(X_enc, y)
            self.m2.fit(X_enc, y)
            self.m3.fit(X_enc, y)
            self.is_ready = True
            return "SUCCESS"
        except Exception as e:
            return f"학습 오류: {str(e)}"

    def predict(self, inputs):
        if not self.is_ready: return None
        df_in = pd.DataFrame([{
            'MA': inputs['MA'], 'SZ': inputs['SZ'], 'IN': inputs['IN'],
            'TH': inputs['TH'], 'DP': inputs['DP'], 'POMFCT': inputs['NOMFCT']
        }])
        df_in[self.cat_vars] = self.encoder.transform(df_in[self.cat_vars].astype(str))
        res = (self.m1.predict(df_in)[0] + self.m2.predict(df_in)[0] + self.m3.predict(df_in)[0]) / 3
        return res

# --- 3. 웹 화면(UI) 구성 ---
st.title("🏭 AI 사출 정밀 예상 CT 시스템")
st.markdown("---")

# 엑셀 파일 로드 (GitHub 저장소에 함께 있는 경우)
FILE_NAME = 'CT-INPUT-V6.xlsx'

if os.path.exists(FILE_NAME):
    try:
        # 데이터 로드 (Past Data 시트, 헤더 2행)
        df_past = pd.read_excel(FILE_NAME, sheet_name='Past Data', header=1)
        
        engine = CT_Ensemble_Engine()
        status = engine.train(df_past)

        if status == "SUCCESS":
            st.sidebar.success("✅ AI 학습 데이터 로드 완료")
            
            # 입력 섹션
            st.subheader("STEP 1. 공정 및 해석 조건 입력")
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

            st.write("")
            if st.button("AI 정밀 분석 실행 (NOPRECT)"):
                inputs = {'MA': ma, 'SZ': sz, 'IN': in_val, 'TH': th, 'DP': dp, 'NOMFCT': nomfct}
                result = engine.predict(inputs)
                
                if result:
                    st.markdown("---")
                    st.subheader("STEP 2. AI 분석 결과 (NOPRECT)")
                    
                    gap = result - nomfct
                    st.metric(label="최종 예상 CT", value=f"{result:.2f} s", delta=f"{gap:+.2f} s (보정)")
                    
                    st.success(f"과거 {len(df_past)}건의 이력을 분석하여 도출된 결과입니다.")
        else:
            st.error(f"데이터 학습 실패: {status}")
    except Exception as e:
        st.error(f"엑셀 파일 읽기 오류: {e}")
else:
    st.error(f"파일을 찾을 수 없습니다: {FILE_NAME}. GitHub에 엑셀 파일을 함께 올려주세요.")
