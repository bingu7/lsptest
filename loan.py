# 安装依赖库（已注释，实际运行时可能需要取消注释）
# !pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple/
# !pip install shap -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 导入基础库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 导入统计和机器学习库
from scipy.stats import chi2_contingency,ks_2samp,spearmanr,f_oneway
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_curve, auc

# 导入深度学习库
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# 导入模型解释库
import shap

# ======================
# 数据加载与基本信息查看
# ======================
train_data = pd.read_csv('./data/train.csv')  # 加载训练数据
test_data = pd.read_csv('./data/test.csv')    # 加载测试数据

# 打印数据集基本信息
print('训练集信息：')
print(train_data.info())
print('测试集信息：')
print(test_data.info())

# 检查重复值
print(f'训练集中存在的重复值：{train_data.duplicated().sum()}')
print(f'测试集中存在的重复值：{test_data.duplicated().sum()}')

# ======================
# 数据探索与可视化
# ======================
# 查看分类特征的唯一值
characteristic = train_data.select_dtypes(include=['object']).columns
print('训练集中分类变量的唯一值情况：')
for i in characteristic:
    print(f'{i}:')
    print(f'共有:{len(train_data[i].unique())}条唯一值')
    print(train_data[i].unique())
    print('-'*50)

# 定义数值特征及其中文名称映射
feature_map = {
    'person_age': '借款人年龄',
    'person_income': '借款人年收入',
    'person_emp_length': '借款人工作年限',
    'loan_amnt': '贷款金额',
    'loan_int_rate': '贷款利率（百分比）',
    'loan_percent_income': '贷款金额占收入的比例',
    'cb_person_cred_hist_length': '信用历史长度（年）'
}

# 绘制训练集和测试集的箱线图（检测异常值）
plt.figure(figsize=(20, 15))
for i, (col, col_name) in enumerate(feature_map.items(), 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=train_data[col])
    plt.title(f'训练集中{col_name}的箱线图', fontsize=14)
plt.tight_layout()
plt.show()

# 同样绘制测试集的箱线图
plt.figure(figsize=(20, 15))
for i, (col, col_name) in enumerate(feature_map.items(), 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=test_data[col])
    plt.title(f'测试集中{col_name}的箱线图', fontsize=14)
plt.tight_layout()
plt.show()

# ======================
# 数据清洗与特征工程
# ======================
# 处理异常值：删除年龄>120或工作年限>120的记录
abnormal_data = train_data[(train_data['person_age'] > 120) | (train_data['person_emp_length'] > 120)]
train_data = train_data[(train_data['person_age'] <= 120) & (train_data['person_emp_length'] <= 120)]

# 创建新特征：年龄-工作年限差
train_age_emp = train_data.copy()
train_age_emp['age_minus_emp_length'] = train_age_emp['person_age'] - train_age_emp['person_emp_length']

test_age_emp = test_data.copy()
test_age_emp['age_minus_emp_length'] = test_age_emp['person_age'] - test_age_emp['person_emp_length']

# 分析年龄-工龄差的分布
for i in range(8,19):
    # 统计差异过小的记录
    ...

for i in range(30, 60, 10):
    # 统计差异过大的记录
    ...

# 调整年龄和工作年限的逻辑函数
def adjust_age_and_emp_length(df):
    """修正年龄和工作年限的不合理差值"""
    df['age_minus_emp_length'] = df['person_age'] - df['person_emp_length']
    
    # 差值小于16：增加年龄
    df.loc[df['age_minus_emp_length'] < 16, 'person_age'] += 16 - df['age_minus_emp_length']
    
    # 差值大于40：增加工龄
    df.loc[df['age_minus_emp_length'] > 40, 'person_emp_length'] += df['age_minus_emp_length'] - 40
    
    # 重新计算差值
    df['age_minus_emp_length'] = df['person_age'] - df['person_emp_length']
    return df

# 应用调整函数
train_age_emp = adjust_age_and_emp_length(train_age_emp)
test_age_emp = adjust_age_and_emp_length(test_age_emp)

# 更新原始数据
train_data['person_age'] = train_age_emp['person_age']
train_data['person_emp_length'] = train_age_emp['person_emp_length']
test_data['person_age'] = test_age_emp['person_age']
test_data['person_emp_length'] = test_age_emp['person_emp_length']

# 创建新特征：年龄-信用历史长度差
train_age_emp['age_minus_hist_length'] = train_age_emp['person_age'] - train_age_emp['cb_person_cred_hist_length']
test_age_emp['age_minus_hist_length'] = test_age_emp['person_age'] - test_age_emp['cb_person_cred_hist_length']

# 删除信用历史异常的记录
train_data = train_data[train_data['id'] != 21827]

# 重新计算贷款收入比
train_data['loan_percent_income'] = round(train_data['loan_amnt'] / train_data['person_income'],2)
test_data['loan_percent_income'] = round(test_data['loan_amnt'] / test_data['person_income'],2)

# ======================
# 数据集分布比较
# ======================
# 合并训练集和测试集进行比较分析
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'
combined_df = pd.concat([train_data, test_data], ignore_index=True)

# 绘制多变量分布比较图（数值变量用核密度，分类变量用堆叠条形图）
plt.figure(figsize=(20,15))
# 多个子图展示不同特征的分布比较...
plt.tight_layout()
plt.show()

# 移除临时添加的dataset列
train_data.drop(columns=['dataset'], inplace=True)
test_data.drop(columns=['dataset'], inplace=True)

# ======================
# 统计检验
# ======================
# KS检验：比较训练集和测试集数值特征的分布差异
numerical_features = test_data.select_dtypes(include=['int64','float64']).columns[1:]
results = []
for feature in numerical_features:
    statistic, p_value = ks_2samp(train_data[feature], test_data[feature])
    results.append({'Feature': feature,'Statistic': statistic, 'p-value': p_value})
results_df = pd.DataFrame(results)

# 卡方检验：比较训练集和测试集分类特征的分布差异
categorical_features = train_data.select_dtypes(include=['object']).columns
results = []
for feature in categorical_features:
    table = pd.crosstab(train_data[feature], test_data[feature])
    chi2, p, dof, expected = chi2_contingency(table)
    results.append({'Feature': feature, 'Statistic': chi2, 'p-value': p})
results_df = pd.DataFrame(results)

# ======================
# 目标变量分析
# ======================
# 单变量与目标变量关系可视化
plt.figure(figsize=(20,12))
# 多个子图展示不同特征与贷款批准状态的关系...
plt.tight_layout()
plt.show()

# 置信区间分析函数
def calculate_confidence_interval(successes, total, confidence=0.95):
    """计算二项分布的置信区间"""
    ...

def analyze_feature(df, feature_name):
    """分析特定特征的贷款批准率及其置信区间"""
    ...

# 对关键分类特征进行置信区间分析
print("房屋拥有情况的置信区间分析:")
analyze_feature(train_data, 'person_home_ownership')
# 其他特征类似...

# ======================
# 特征编码与预处理
# ======================
# 有序类别编码
grade_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
defaul_mapping = {'N': 0, 'Y':1}
train_data['loan_grade'] = train_data['loan_grade'].map(grade_order)
test_data['loan_grade'] = test_data['loan_grade'].map(grade_order)
train_data['cb_person_default_on_file'] = train_data['cb_person_default_on_file'].map(defaul_mapping)
test_data['cb_person_default_on_file'] = test_data['cb_person_default_on_file'].map(defaul_mapping)

# 相关性分析热力图
def plot_spearmanr(data,features,title,wide,height):
    """绘制斯皮尔曼相关系数热力图"""
    ...

features = train_data.drop(['id','person_home_ownership','loan_intent'],axis=1).columns.tolist()
plot_spearmanr(train_data,features,'各变量之间的斯皮尔曼相关系数热力图',12,15)

# 卡方检验：分类特征与目标变量的关系
def chi_square_test(var1, var2):
    """执行卡方检验"""
    ...
    
loan_status_chi_square_results = {feature: chi_square_test(feature, 'loan_status') 
                                  for feature in ['person_home_ownership', 'loan_intent']}

# 特征选择与预处理
new_train_data = train_data.drop(columns=['id','person_age','cb_person_cred_hist_length'])
new_test_data = test_data.drop(columns=['id','person_age','cb_person_cred_hist_length'])

# 独热编码
one_hot_features = ['person_home_ownership', 'loan_intent']
new_train_data = pd.get_dummies(new_train_data, columns=one_hot_features, drop_first=True)
new_test_data = pd.get_dummies(new_test_data, columns=one_hot_features, drop_first=True)

# 数据类型转换
new_train_data = new_train_data.astype(int)
new_test_data = new_test_data.astype(int)

# 数值特征标准化
numerical_features = ['person_income', 'person_emp_length','loan_amnt','loan_int_rate','loan_percent_income']
scaler = StandardScaler()
new_train_data[numerical_features] = scaler.fit_transform(new_train_data[numerical_features])
new_test_data[numerical_features] = scaler.transform(new_test_data[numerical_features])

# ======================
# 模型训练与评估
# ======================
# 数据集划分
x = new_train_data.drop(['loan_status'],axis=1)
y = new_train_data['loan_status']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=15)

# 1. 逻辑回归模型
log_reg = LogisticRegression(random_state=15)
log_reg.fit(x_train, y_train)
y_pred_log = log_reg.predict(x_test)
# 输出评估报告、绘制混淆矩阵和ROC曲线...

# 2. 随机森林模型
rf_clf = RandomForestClassifier(random_state=15)
rf_clf.fit(x_train, y_train)
y_pred_rf = rf_clf.predict(x_test)
# 输出评估报告、绘制混淆矩阵和ROC曲线...

# 3. 多层感知机（浅层神经网络）
mlp_model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=0)
y_pred_mlp = (mlp_model.predict(x_test) >= 0.6).astype(int)
# 输出评估报告、绘制混淆矩阵和ROC曲线...

# 4. 深度神经网络
deep_model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
deep_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
deep_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=0)
y_pred_deep = (deep_model.predict(x_test) >= 0.6).astype(int)
# 输出评估报告、绘制混淆矩阵和ROC曲线...

# ======================
# 模型解释（SHAP）
# ======================
# 使用SHAP解释深度神经网络
explainer = shap.DeepExplainer(deep_model, x_test.values[:1000])
shap_values = explainer.shap_values(x_test.values)
shap_values = shap_values.squeeze(-1)

# 绘制SHAP摘要图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
plt.title("深度神经网络的 SHAP 特征重要性")
plt.tight_layout()
plt.show()

# 计算特征重要性
feature_importance = np.abs(shap_values).mean(0)
importance_df = pd.DataFrame(list(zip(x_test.columns, feature_importance)), 
                             columns=['特征', '重要性'])
importance_df = importance_df.sort_values('重要性', ascending=False)

# 绘制SHAP值分布图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, x_test, show=False)
plt.title("SHAP 值分布")
plt.tight_layout()
plt.show()

# ======================
# 测试集预测
# ======================
new_y_pred_log = log_reg.predict(new_test_data)
new_y_pred_rf = rf_clf.predict(new_test_data)
new_y_pred_mlp = (mlp_model.predict(new_test_data) >= 0.6).astype(int).ravel()
new_y_pred_deep = (deep_model.predict(new_test_data) >= 0.6).astype(int).ravel()

# 保存预测结果
test_data['log_pred'] = new_y_pred_log
test_data['rf_pred'] = new_y_pred_rf
test_data['mlp_pred'] = new_y_pred_mlp
test_data['nn_pred'] = new_y_pred_deep
test_data.head()