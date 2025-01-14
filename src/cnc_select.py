import pandas as pd
from config import *

def count_categories(csv_path):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # 각 카테고리 카운트
    nb_pass = len(df[(df['machining_finalized'] == 'yes') & (df['passed_visual_inspection'] == 'yes')])
    nb_pass_half = len(df[(df['machining_finalized'] == 'yes') & (df['passed_visual_inspection'] == 'no')])
    nb_fail = len(df[df['machining_finalized'] == 'no'])
    
    # 결과 출력
    print(f"Pass 샘플 수: {nb_pass}")
    print(f"Pass-half 샘플 수: {nb_pass_half}")
    print(f"Fail 샘플 수: {nb_fail}")
    print(f"전체 샘플 수: {nb_pass + nb_pass_half + nb_fail}")
    
    return nb_pass, nb_pass_half, nb_fail

# 사용 예시
if __name__ == "__main__":

    virtual_data_path = os.path.join(data_path, "CNC_SMART_MICHIGAN")
    csv_path = os.path.join(virtual_data_path, "train.csv")
    count_categories(csv_path)