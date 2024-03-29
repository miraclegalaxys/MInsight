import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
from imblearn.over_sampling import SMOTE 
import pandas as pd

def load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column): # สร้างฟังก์ชัน load_and_preprocess_data ที่รับพารามิเตอร์ 4 ตัวคือ filepath, numeric_columns, categorical_features, target_column
    df = pd.read_csv(filepath) # อ่านไฟล์ CSV จาก filepath และเก็บไว้ในตัวแปร df
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median()) # ใส่ค่าเฉลี่ยของคอลัมน์ที่เป็นตัวเลขใน numeric_columns ในที่นี้คือ ['Frequency', 'Duration', 'Targets'] ในคอลัมน์ที่มีค่าว่าง
    df = pd.get_dummies(df, columns=categorical_features) # แปลงข้อมูลที่เป็นข้อความใน categorical_features ให้เป็นข้อมูลที่เป็นตัวเลข
    features = df.drop(target_column, axis=1)
    labels = pd.get_dummies(df[target_column])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels

def create_model(num_features, num_classes):
    model = models.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(features, labels, patience=10, num_repeats=5):
    best_accuracy = 0
    best_model = None

    for repeat in range(num_repeats):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cvscores = []

        for fold, (train, test) in enumerate(kfold.split(features, np.argmax(labels.values, axis=1))):
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(features[train], np.argmax(labels.values[train], axis=1))
            y_train = pd.get_dummies(y_train)

            model = create_model(num_features=features.shape[1], num_classes=labels.shape[1])
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping], verbose=0)
            scores = model.evaluate(features[test], labels.values[test], verbose=0)
            cvscores.append(scores[1] * 100)


        mean_accuracy = np.mean(cvscores)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_model = model

        # Print the overall progress for each iteration
        percent_complete = ((repeat + 1) / num_repeats) * 100
        print(f'Iteration {repeat + 1} Completed: {percent_complete:.2f}% Done. Mean accuracy: {mean_accuracy:.2f}%')

    print(f'Best accuracy: {best_accuracy:.2f}%')
    return best_model


def predict_next_attack(model, new_data, labels):
    prediction = model.predict(new_data.reshape(1, -1))
    predicted_attack = labels.columns[np.argmax(prediction)]
    print(f"Predicted attack type for the next data point using the best model: {predicted_attack}")

if __name__ == '__main__':
    numeric_columns = ['Frequency', 'Duration', 'Targets']
    categorical_features = ['Severity']
    filepath = '/Users/miracle/CPE/cpe_270/MInsight/cyberattacks.csv'
    target_column = 'Attack_Type'

    features, labels = load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column)
    best_model = train_and_evaluate_model(features, labels, patience=10)
    new_data = features[0]  # Replace with your new data point
    predict_next_attack(best_model, new_data, labels)
    
    