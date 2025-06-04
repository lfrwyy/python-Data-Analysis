import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from imblearn.over_sampling import SMOTE


def preprocess_data(df, seq_length=10):
    """优化后的数据预处理函数"""
    selected_features = ['累计里程', '总电压', '总电流', 'SOC', '绝缘电阻',
                         '驱动电机控制器温度', '驱动电机转矩', '驱动电机温度',
                         '电机控制器输入电压', '电池单体电压最高值',
                         '电池单体电压最低值', '最高温度值', '最低温度值']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[selected_features]).astype(np.float32)

    # 序列数据生成
    X_sequence, valid_indices = [], []
    for i in range(len(scaled_data) - seq_length):
        X_sequence.append(scaled_data[i:i + seq_length])
        valid_indices.append(i + seq_length - 1)

    X_xgb = scaled_data[valid_indices]
    y = df['最高报警等级'].values[valid_indices]

    # 平衡采样
    sample_sizes = {0: 40000, 1: 10000, 2: 20000, 3: 10000}  # 修改采样数量
    balanced_indices = []
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        balanced_indices.extend(np.random.choice(cls_indices, sample_sizes[cls], replace=True))

    return np.array(X_sequence)[balanced_indices], X_xgb[balanced_indices], y[balanced_indices], scaler


# 数据准备
full_data = pd.read_csv('finaldata.csv', encoding='gb2312')
X_seq, X_xgb, y, scaler = preprocess_data(full_data)

# 数据集划分
X_seq_train, X_seq_test, X_xgb_train, X_xgb_test, y_train, y_test = train_test_split(
    X_seq, X_xgb, y, test_size=0.2, stratify=y, random_state=42)

# 类别权重计算
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# 数据增强
# 删除SMOTE过采样部分
# sampler = SMOTE(sampling_strategy={1: 500, 3: 500}, k_neighbors=2, random_state=42)
# X_xgb_train, y_train = sampler.fit_resample(X_xgb_train, y_train)

# 混合模型架构
def build_model(input_shape, xgb_feature_num):
    # LSTM分支
    lstm_input = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True, dropout=0.3)(lstm_input)
    x = LSTM(32, dropout=0.2)(x)
    lstm_output = Dense(16, activation='relu')(x)

    # XGBoost分支
    xgb_input = Input(shape=(xgb_feature_num,))

    # 特征融合
    combined = concatenate([lstm_output, xgb_input])
    x = Dense(32, activation='relu', kernel_regularizer='l2')(combined)
    x = Dense(16, activation='relu')(x)
    final_output = Dense(4, activation='softmax')(x)

    model = Model(inputs=[lstm_input, xgb_input], outputs=final_output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 初始化模型
model = build_model((X_seq.shape[1], X_seq.shape[2]), X_xgb.shape[1])

# XGBoost模型训练（修复未定义错误）
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    tree_method='hist'
)
xgb_model.fit(X_xgb_train, y_train)

# 训练配置
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# 模型训练
history = model.fit(
    [X_seq_train, X_xgb_train], y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# 模型评估
test_loss, test_acc = model.evaluate([X_seq_test, X_xgb_test], y_test, verbose=0)
print(f'测试集准确率: {test_acc:.4f}')
y_pred = np.argmax(model.predict([X_seq_test, X_xgb_test]), axis=1)
print(classification_report(y_test, y_pred, zero_division=0))


# 预测函数（带异常处理）
# 修改预测函数以保留时间戳
def predict_new_data(model, xgb_model, scaler, file_path, seq_length=10):
    try:
        test_df = pd.read_csv(file_path, encoding='gb2312')
        selected_features = ['累计里程', '总电压', '总电流', 'SOC', '绝缘电阻',
                             '驱动电机控制器温度', '驱动电机转矩', '驱动电机温度',
                             '电机控制器输入电压', '电池单体电压最高值',
                             '电池单体电压最低值', '最高温度值', '最低温度值']

        scaled_test = scaler.transform(test_df[selected_features]).astype(np.float32)

        # 初始化预测结果存储
        all_predictions = []
        valid_indices = []

        # 生成序列数据并获取时间戳
        for i in range(len(scaled_test) - seq_length + 1):
            X_seq = scaled_test[i:i + seq_length].reshape(1, seq_length, -1)
            X_xgb = scaled_test[i + seq_length - 1].reshape(1, -1)

            # 进行预测
            final_pred = np.argmax(model.predict([X_seq, X_xgb], verbose=0), axis=1)
            all_predictions.append(final_pred[0])
            valid_indices.append(i + seq_length - 1)

        # 获取有效时间戳
        timestamps = test_df['数据采集时间'].iloc[valid_indices].values

        return np.array(all_predictions), timestamps
    except Exception as e:
        print(f"预测错误: {str(e)}")
        return np.array([-1]), np.array([-1])

# 修改结果保存部分
predictions, timestamps = predict_new_data(model, xgb_model, scaler, '10finaldata.csv')
result_df = pd.DataFrame({
    '数据采集时间': timestamps,
    'Predicted_Level': predictions
})
result_df.to_csv('10XGBoost+LSTM_new_predictions.csv', index=False)