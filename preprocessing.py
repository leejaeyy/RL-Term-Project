# %%

import pandas as pd
import numpy as np

path = r"C:\Users\admin\Desktop\강화학습_과제\Data\kz.csv"

df = pd.read_csv(path)

df.head()


# %%
# eCommerce 데이터 전처리
# STATE 구성: recency, frequency, avg_price, diversity
# WINDOW: 최근 30일
# OUTPUT: states_30d.csv
import pandas as pd
import numpy as np
import os

# 경로
Data = r"C:\Users\admin\Desktop\강화학습_과제\Data\kz.csv"
OUT = r"C:\Users\admin\Desktop\강화학습_과제\Data\states_30d.csv"
WINDOW = "30D"

def to_datetime_utc(s):
    " datetime 변환"
    return pd.to_datetime(s, errors="coerce", utc=True)


def top_category(code: str):
    """카테고리 코드 상위 토큰 추출"""
    if pd.isna(code):
        return np.nan
    return str(code).split(".")[0]


def make_unique_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """(user_id, event_time) 중복 발생 시 ns 오프셋을 더해 고유화"""
    df = df.sort_values(["user_id", "event_time"]).copy()
    df.set_index(["user_id", "event_time"], inplace=True)
    if df.index.duplicated().any():
        idx = df.index.to_frame(index=False)
        cc = idx.groupby(["user_id", "event_time"]).cumcount()
        offset = pd.to_timedelta(cc, unit="ns")
        new_times = df.index.get_level_values(1) + offset
        df.set_index([df.index.get_level_values(0), new_times], inplace=True)
        df.sort_index(inplace=True)
    return df


# Main Processing Function

def main():
    # 1데이터 로드
    need = ["event_time", "user_id", "price",
            "order_id", "product_id", "category_id", "category_code", "brand"]
    df = pd.read_csv(Data)
    for c in need:
        if c not in df.columns:
            df[c] = np.nan


    df["event_time"] = to_datetime_utc(df["event_time"])
    df = df.dropna(subset=["event_time", "user_id", "price"]).copy()


    for c in ["order_id", "product_id", "brand"]:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")

    df["top_cat"] = df["category_code"].astype("string").map(top_category).astype("category")
    df["cat_id_code"] = df["category_id"].astype("Int64")

    df = make_unique_multiindex(df)

    p_ref = float(df["price"].median())
    if not np.isfinite(p_ref) or p_ref <= 0:
        p_ref = 1.0

    # STATE Feature Calculation

    # recency_days
    times = df.index.get_level_values(1).to_series()
    last_time = times.groupby(df.index.get_level_values(0)).shift(1)
    recency_days = (times - last_time).dt.total_seconds().div(86400.0).fillna(60.0)

    # frequency (30일 내 구매 횟수)
    ones = pd.Series(1.0, index=df.index)
    freq_30d = (
        ones.groupby(level=0, group_keys=False)
            .apply(lambda g: g.droplevel(0).shift(1)
                   .rolling(WINDOW).sum())
            .fillna(0.0)
    )

    # avg_price_30d (30일 평균 가격)
    avg_price_30d = (
        df["price"].groupby(level=0, group_keys=False)
           .apply(lambda g: g.droplevel(0).shift(1)
                  .rolling(WINDOW).mean())
           .fillna(0.0)
    )

    # diversity_30d (30일 내 고유 카테고리 수)
    use_div = df["cat_id_code"].astype("float64")
    if use_div.isna().all():
        use_div = df["top_cat"].cat.codes.astype("float64")

    def uniq_count(x):
        x = x[~np.isnan(x)]
        return np.unique(x).size

    diversity_30d = (
        use_div.groupby(level=0, group_keys=False)
               .apply(lambda g: g.droplevel(0).shift(1)
               .rolling(WINDOW)
               .apply(uniq_count, raw=True))
               .fillna(0.0)
    )


    # 결과 구성 및 정규화

    out = pd.DataFrame({
        "user_id": df.index.get_level_values(0).to_numpy(),
        "event_time": df.index.get_level_values(1).tz_convert("UTC").to_numpy(),
        "recency_days": recency_days.to_numpy(),
        "freq_30d": freq_30d.to_numpy(),
        "avg_price_30d": avg_price_30d.to_numpy(),
        "diversity_30d": diversity_30d.to_numpy(),
        "price": df["price"].to_numpy(),
        "order_id": df["order_id"].astype("string").to_numpy(),
        "product_id": df["product_id"].astype("string").to_numpy(),
        "category_id": df["category_id"].to_numpy(),
        "category_code": df["category_code"].astype("string").to_numpy(),
        "brand": df["brand"].astype("string").to_numpy(),
    })

    out["recency_norm"] = np.clip(out["recency_days"] / 30.0, 0, 1.5)
    out["frequency_norm"] = np.clip(out["freq_30d"] / 10.0, 0, 1.0)
    out["avg_price_norm"] = np.clip(out["avg_price_30d"] / p_ref, 0, 1.5)
    out["diversity_norm"] = np.clip(out["diversity_30d"] / 5.0, 0, 1.0)

    # 저장

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[OK] saved -> {OUT} (rows={len(out):,}) median_price={p_ref:.2f}")


if __name__ == "__main__":
    main()
    print("저장 :", os.path.exists(OUT), OUT)


# %%

out_path = r"C:\Users\admin\Desktop\강화학습_과제\Data\states_30d.csv" 
out_df = pd.read_csv(out_path)

out_df.head()

# %%
out_df.info()

out_df.isnull().sum()

# %%



