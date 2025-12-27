import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import ctypes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder

# 윈도우 고해상도 지원 (글자 흐림 방지)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

def get_base_path():
    """실행 파일(.exe) 또는 스크립트가 위치한 폴더 경로를 반환"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

# --- [자동 설정] ---
BASE_PATH = get_base_path()
FILE_NAME = 'CT-INPUT-V6.xlsx'
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)
# ------------------

class CT_Ensemble_Engine:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.m1 = Ridge(alpha=1.0)
        self.m2 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.m3 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
        self.is_ready = False
        self.cat_vars = ['MA', 'SZ', 'IN', 'TH', 'DP']

    def train(self):
        if not os.path.exists(FILE_PATH):
            return f"파일을 찾을 수 없습니다.\n파일명: {FILE_NAME}\n위치: {BASE_PATH}"
        
        try:
            # 엑셀 로드 (헤더 2행 기준)
            df = pd.read_excel(FILE_PATH, sheet_name='Past Data', header=1)
            df.columns = [str(c).strip().upper() for c in df.columns]
            
            target_col = 'POINCT'   # 과거 실측 CT
            past_nom_col = 'POMFCT' # 과거 해석 CT
            feature_cols = self.cat_vars + [past_nom_col]
            
            # 결측치 제거 및 데이터 준비
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]
            
            # 문자열 변수 인코딩
            X_enc = X.copy()
            X_enc[self.cat_vars] = self.encoder.fit_transform(X[self.cat_vars].astype(str))
            
            # 앙상블 모델 학습
            self.m1.fit(X_enc, y)
            self.m2.fit(X_enc, y)
            self.m3.fit(X_enc, y)
            self.is_ready = True
            return "SUCCESS"
        except Exception as e:
            return f"학습 오류: {str(e)}"

    def predict(self, inputs):
        if not self.is_ready: return None
        # 화면의 NOMFCT를 학습된 POMFCT 위치에 매칭
        df_in = pd.DataFrame([{
            'MA': inputs['MA'], 'SZ': inputs['SZ'], 'IN': inputs['IN'],
            'TH': inputs['TH'], 'DP': inputs['DP'], 'POMFCT': inputs['NOMFCT']
        }])
        df_in[self.cat_vars] = self.encoder.transform(df_in[self.cat_vars].astype(str))
        return (self.m1.predict(df_in)[0] + self.m2.predict(df_in)[0] + self.m3.predict(df_in)[0]) / 3

class CT_App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 사출 CT 정밀 분석 시스템 (V6-AN)")
        self.root.geometry("800x1000")
        self.root.configure(bg="#F4F7F9")
        
        self.engine = CT_Ensemble_Engine()
        status = self.engine.train()
        
        self.f_step = ("Malgun Gothic", 11, "bold")
        self.f_lab = ("Malgun Gothic", 9)
        self.f_res = ("Malgun Gothic", 12, "bold")

        tk.Label(root, text="AI 사출 정밀 예상 CT 시스템", font=("Malgun Gothic", 20, "bold"), bg="#2C3E50", fg="white", pady=20).pack(fill=tk.X)
        c = tk.Frame(root, bg="#F4F7F9", padx=50); c.pack(fill=tk.BOTH, expand=True)

        s1 = tk.LabelFrame(c, text=" STEP 1. 현재 공정 및 해석 조건 입력 ", font=self.f_step, bg="white", padx=30, pady=20)
        s1.pack(fill=tk.X, pady=20)

        self.vars = {}
        try:
            temp_df = pd.read_excel(FILE_PATH, sheet_name='Past Data', header=1)
            ma_list = sorted([str(x).strip() for x in temp_df['MA'].dropna().unique()])
        except: 
            ma_list = ["650", "630", "8636"]

        opts = {"MA": ma_list, "SZ": ["S", "M", "L"], "IN": ["IO", "IX"], "TH": ["TS", "TM", "TL"], "DP": ["DS", "DM", "DL"]}
        
        for k, v in opts.items():
            row = tk.Frame(s1, bg="white", pady=4); row.pack(fill=tk.X)
            tk.Label(row, text=f"{k}:", font=self.f_lab, bg="white", width=15, anchor="w").pack(side=tk.LEFT)
            self.vars[k] = ttk.Combobox(row, values=v, state="readonly")
            if v: self.vars[k].current(0)
            self.vars[k].pack(side=tk.RIGHT, expand=True, fill=tk.X)

        nf = tk.Frame(s1, bg="#FFF5F0", pady=10, padx=10); nf.pack(fill=tk.X, pady=(15, 0))
        tk.Label(nf, text="현재 성형 해석 후 CT (NOMFCT):", font=self.f_lab, bg="#FFF5F0").pack(side=tk.LEFT)
        self.ent_nom = tk.Entry(nf, font=("Arial", 11, "bold"), justify="center")
        self.ent_nom.insert(0, "200.0"); self.ent_nom.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(10, 0))

        tk.Button(c, text="AI 정밀 분석 실행 (NOPRECT)", font=self.f_step, bg="#C0392B", fg="white", pady=10, command=self.go).pack(fill=tk.X, pady=20)

        self.s2 = tk.LabelFrame(c, text=" STEP 2. 현재 예상 사출 전체 CT (NOPRECT) ", font=self.f_step, bg="white", padx=30, pady=30)
        self.s2.pack(fill=tk.BOTH, expand=True)
        
        self.l1 = tk.Label(self.s2, text="분석 대기 중", font=self.f_res, bg="white", fg="#95A5A6"); self.l1.pack(pady=10)
        self.l2 = tk.Label(self.s2, text="과거 데이터를 기반으로 오차를 보정합니다.", font=self.f_lab, bg="white", fg="#BDC3C7"); self.l2.pack()

        if status != "SUCCESS":
            messagebox.showwarning("파일 확인", status)

    def go(self):
        try:
            d = {k: v.get() for k, v in self.vars.items()}
            current_nom = float(self.ent_nom.get())
            d['NOMFCT'] = current_nom
            res = self.engine.predict(d)
            if res:
                self.l1.config(text=f"최종 예상 CT (NOPRECT): {res:.2f} s", fg="#C0392B")
                self.l2.config(text=f"해석치 {current_nom}s 대비 현장 편차 반영 완료", fg="#2C3E50")
        except:
            messagebox.showwarning("입력 오류", "숫자를 확인해 주세요.")

if __name__ == "__main__":
    root = tk.Tk(); app = CT_App(root); root.mainloop()