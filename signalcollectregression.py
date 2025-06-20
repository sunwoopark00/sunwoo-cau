import pandas as pd

# 논문 기반 설정
window_size = 300
pre_trigger = 50
post_trigger = 250

threshold = 0.03
min_gap = 300

# 파일 경로
center_file = "C:/Users/psw09/Desktop/pvdf test/pvdf_data_center_carbon 20250619_155536.csv"
front_file = "C:/Users/psw09/Desktop/pvdf test/pvdf_data_front_carbon 20250619_160708.csv"
back_file = "C:/Users/psw09/Desktop/pvdf test/pvdf_data_back_carbon 20250619_161630.csv"

# 📍 좌표 지정 (중앙 / 앞 / 뒤)
label_coords = {
    "center": (0.0, 0.0),
    "front":  (0.0, -55.0),
    "back":   (0.0, -55.0)
}

# 수정된 함수: x, y 좌표 추가
def extract_impacts_with_xy(filepath, coord_xy):  
    df = pd.read_csv(filepath, encoding="cp949")
    sensor_cols = df.columns[:4]  # 센서 4개 사용 시

    abs_max = df[sensor_cols].abs().max(axis=1)
    peak_indices = abs_max[abs_max > threshold].index

    segments = []
    last_index = -min_gap

    for idx in peak_indices:
        if idx - last_index >= min_gap:
            start = idx - pre_trigger
            end = idx + post_trigger
            if start >= 0 and end <= len(df):
                segment = df.iloc[start:end].copy()
                segment["x"] = coord_xy[0]
                segment["y"] = coord_xy[1]
                segments.append(segment)
                last_index = idx
    return segments

# 🔄 좌표 기반 추출
center_segments = extract_impacts_with_xy(center_file, label_coords["center"])
front_segments = extract_impacts_with_xy(front_file, label_coords["front"])
back_segments = extract_impacts_with_xy(back_file, label_coords["back"])

# 병합 및 저장
df_combined = pd.concat(center_segments + front_segments + back_segments, ignore_index=True)
output_path = "C:/Users/psw09/Desktop/pvdf test/carbon619_regression.csv"
df_combined.to_csv(output_path, index=False)

# 결과 미리 보기
print("✅ Saved regression dataset with coordinates.")
df_combined.head()
