##############################
# 題名:ランダムフォレストモデルの理解
# 作者:Yasutaka Yoshihara
# 日付:2022/12/04
##############################

# ライブラリの読み出し
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# CSVファイル読みだし
iowa_file_path = './data/input/train.csv'
home_data = pd.read_csv(iowa_file_path)


# 目的変数yと説明変数Xの定義
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]


# 訓練データとテストデータの分割
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# ランダムフォレストモデルの生成とモデルフィッティング
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)


# 予測精度検証と平均絶対値誤差(MAE)の計算
rf_val_pred = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_pred)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


