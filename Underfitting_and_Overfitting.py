##############################
# 題名:過剰学習と過少学習の理解
# 作者:Yasutaka Yoshihara
# 日付:2022/12/03
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

# # 学習モデル(決定木)の定義とフィッティング
# iowa_model = DecisionTreeRegressor(random_state=1)
# iowa_model.fit(train_X, train_y)

# # 予測精度検証と平均絶対値誤差(MAE)の計算
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(val_predictions, val_y)
# print("Validation MAE: {:,.0f}".format(val_mae))


# learntoolsのセットアップ確認
# from learntools.core import binder
# binder.bind(globals())
# from learntools.machine_learning.ex5 import *
# print("\nSetup complete")


# 関数の定義(決定木モデルの定義とフィッティング)
def get_mae(candidate_max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = candidate_max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# 最適なツリーのノード数の決定/モデルパラメータ最適化
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size : get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print("\nノード数の候補は:" + str(candidate_max_leaf_nodes))
print("最適なツリーのノード数は:" + str(best_tree_size))

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)
print(">>> モデルパラメータ最適化完了")
