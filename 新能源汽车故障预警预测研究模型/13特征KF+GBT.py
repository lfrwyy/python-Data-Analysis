import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import joblib
from imblearn.under_sampling import RandomUnderSampler

# 在preprocess_data函数中修改特征列表
def preprocess_data(filepath):
    raw_data = pd.read_csv(filepath, encoding='GB2312')

    features = [
        '累计里程', '总电压', '总电流', 'SOC', '绝缘电阻',
        '驱动电机控制器温度', '驱动电机转矩', '驱动电机温度',
        '电机控制器输入电压', '电池单体电压最高值',
        '电池单体电压最低值', '最高温度值', '最低温度值'
    ]
    target = '最高报警等级'

    X = raw_data[features].copy()
    y_binary = raw_data[target].apply(lambda x: 1 if x >= 1 else 0)
    y_level = raw_data[target].clip(upper=3)
    timestamps = raw_data.iloc[:, 0]
    alarm_level = raw_data[target]

    # 对等级0和等级2进行欠采样
    rus = RandomUnderSampler(sampling_strategy={0: 40000, 2: 20000}, random_state=42)
    X_res, y_level_res = rus.fit_resample(X, y_level)
    timestamps = timestamps.iloc[rus.sample_indices_]
    y_binary = y_binary.iloc[rus.sample_indices_]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    processed_data = pd.DataFrame(X_scaled, columns=features, index=rus.sample_indices_)
    processed_data.insert(0, '数据采集时间', timestamps)
    return processed_data, y_binary, y_level_res, scaler, alarm_level


def kalman_predict(X):
    n_dim = X.shape[1]
    kf = KalmanFilter(
        transition_matrices=np.eye(n_dim),
        observation_matrices=np.eye(n_dim),
        initial_state_mean=X.iloc[0].values.astype(float),
        initial_state_covariance=np.eye(n_dim),
        observation_covariance=0.5 * np.eye(n_dim),
        transition_covariance=0.1 * np.eye(n_dim)
    )

    # 初始化状态估计数组
    state_means = np.zeros((len(X), n_dim))
    current_state = kf.initial_state_mean
    current_covariance = kf.initial_state_covariance

    # 流式处理数据
    for t in range(len(X)):
        if t > 0:
            current_state, current_covariance = kf.filter_update(
                current_state,
                current_covariance,
                observation=X.iloc[t].values.astype(float)
            )
        state_means[t] = current_state

    return pd.DataFrame(state_means,
                        columns=[f'kalman_{col}' for col in X.columns],
                        index=X.index)


def train_level_classifier(X, y_level):
    fault_mask = y_level > 0
    X_fault = X[fault_mask]
    y_fault = y_level[fault_mask]

    # 计算类别权重
    classes = np.unique(y_fault)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_fault)
    class_weights = dict(zip(classes, weights))

    # 调整过采样策略，删除等级2
    smote = SMOTE(sampling_strategy={
        1: 10000,  # 从6过采样到10000
        3: 10000   # 从61过采样到10000
    }, random_state=42, k_neighbors=2)

    X_res, y_res = smote.fit_resample(X_fault, y_fault)

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res)

    # 优化分类器参数
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10
    )

    # 使用样本权重
    sample_weights = np.array([class_weights[y] for y in y_train])

    clf.fit(X_train, y_train, sample_weight=sample_weights)
    # 验证性能
    print("\n等级分类器性能:")
    print(classification_report(y_test, clf.predict(X_test)))
    return clf


# 在preprocess_new_data函数中修改特征列表
def preprocess_new_data(filepath, scaler):
    raw_data = pd.read_csv(filepath, encoding='GB2312')

    features = [
        '累计里程', '总电压', '总电流', 'SOC', '绝缘电阻',
        '驱动电机控制器温度', '驱动电机转矩', '驱动电机温度',
        '电机控制器输入电压', '电池单体电压最高值',
        '电池单体电压最低值', '最高温度值', '最低温度值'
    ]
    target = '最高报警等级'

    X = raw_data[features].copy()
    timestamps = raw_data.iloc[:, 0]

    X_scaled = scaler.transform(X)

    processed_data = pd.DataFrame(X_scaled, columns=features, index=raw_data.index)
    processed_data.insert(0, '数据采集时间', timestamps)
    return processed_data, timestamps


def add_time_series_features(data):
    data['移动平均_电压差'] = data['电池单体电压最高值'] - data['电池单体电压最低值']
    data['滑动平均_电流'] = data['总电流'].rolling(window=5, min_periods=1).mean()
    return data


def detect_faults(data, predictions, threshold_factor=3):
    residuals = data - predictions.values
    threshold = threshold_factor * np.nanstd(residuals.values)
    residual_faults = (np.abs(residuals) > threshold).any(axis=1)
    return residual_faults


def predict_fault_levels(X_combined, level_clf):
    final_pred = np.zeros(len(X_combined), dtype=int)
    fault_mask = X_combined['有无故障'].astype(bool)
    # 移除 '有无故障' 列
    X_combined_no_fault = X_combined.drop(columns=['有无故障'])
    final_pred[fault_mask] = level_clf.predict(X_combined_no_fault[fault_mask])
    return final_pred


def save_results(result, output_path):
    result.to_csv(output_path, index=False, encoding='GB2312')


def predict_new_data(new_data_path, scaler, level_clf, output_path):
    # 预处理新数据
    new_data, timestamps = preprocess_new_data(new_data_path, scaler)
    features = new_data.columns[2:]

    # 增加时序特征
    new_data = add_time_series_features(new_data)
    features = new_data.columns[2:]

    # 分块卡尔曼滤波预测
    chunk_size = 100000
    predictions = pd.DataFrame()

    for i in range(0, len(new_data), chunk_size):
        chunk = new_data[features].iloc[i:i + chunk_size]
        pred_chunk = kalman_predict(chunk)
        predictions = pd.concat([predictions, pred_chunk])

    # 残差分析与故障检测
    residuals = new_data[features] - predictions.values
    residual_faults = detect_faults(new_data[features], predictions)
    final_faults = residual_faults

    # 优化后的等级预测
    X_combined = pd.concat([
        new_data[features],
        predictions.add_prefix('预测_'),
        predictions.diff().add_prefix('变化率_')
    ], axis=1)
    X_combined['有无故障'] = final_faults.astype(int)

    final_pred = predict_fault_levels(X_combined, level_clf)

    # 生成预测结果
    result = pd.DataFrame({
        '数据采集时间': timestamps,
        '有无故障': final_faults.astype(int),
        '预测等级': final_pred,
        '残差最大值': residuals.max(axis=1)
    })

    # 保存结果
    save_results(result, output_path)
    print(f"预测结果已保存到 {output_path}")


def main():
    # 训练模型
    data, y_binary, y_level, scaler, alarm_level = preprocess_data('finaldata.csv')
    features = data.columns[2:]

    # 增加时序特征
    data['移动平均_电压差'] = data['电池单体电压最高值'] - data['电池单体电压最低值']
    data['滑动平均_电流'] = data['总电流'].rolling(window=5, min_periods=1).mean()
    features = data.columns[2:]  # 获取更新后的特征列表

    # 分块卡尔曼滤波预测（使用更新后的特征）
    chunk_size = 100000
    predictions = pd.DataFrame()

    for i in range(0, len(data), chunk_size):
        chunk = data[features].iloc[i:i + chunk_size]  # 使用更新后的特征
        pred_chunk = kalman_predict(chunk)
        predictions = pd.concat([predictions, pred_chunk])

    # 增加时序特征
    data['移动平均_电压差'] = data['电池单体电压最高值'] - data['电池单体电压最低值']
    data['滑动平均_电流'] = data['总电流'].rolling(window=5, min_periods=1).mean()
    features = data.columns[2:]
    # 残差分析与故障检测
    residuals = data[features] - predictions.values
    threshold = 3 * np.nanstd(residuals.values)
    residual_faults = (np.abs(residuals) > threshold).any(axis=1)
    final_faults = y_binary.astype(bool) | residual_faults

    # 优化后的等级预测
    X_combined = pd.concat([
        data[features],
        predictions.add_prefix('预测_'),
        predictions.diff().add_prefix('变化率_')  # 新增变化率特征
    ], axis=1)

    level_clf = train_level_classifier(X_combined, y_level)

    # 保存模型和预处理器
    joblib.dump(level_clf, 'level_clf.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # 生成预测结果
    final_pred = np.zeros(len(data), dtype=int)
    fault_mask = final_faults.astype(bool)
    final_pred[fault_mask] = level_clf.predict(X_combined[fault_mask])

    # 保存优化后的结果
    result = pd.DataFrame({
        '数据采集时间': data['数据采集时间'],
        '有无故障': final_faults.astype(int),
        '预测等级': final_pred,
        '实际等级': y_level.values,
        '残差最大值': residuals.max(axis=1)  # 新增残差特征
    })
    result.to_csv('13特征卡尔曼+梯度提升树预测结果.csv', index=False, encoding='GB2312')

    # 优化评估指标
    valid_mask = (y_level[fault_mask] > 0) & (final_pred[fault_mask] > 0)
    print("\n最终评估结果 (仅故障样本):")
    print(classification_report(y_level[fault_mask][valid_mask], final_pred[fault_mask][valid_mask]))

    # 使用模型预测新数据
    predict_new_data('10finaldata.csv', scaler, level_clf, '10KF+GBT_new_predictions.csv')


if __name__ == '__main__':
    main()
