import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
from imblearn.over_sampling import SMOTE
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests



ascii_art = fr'''

/$$      /$$ /$$$$$$                     /$$           /$$         /$$
| $$$    /$$$|_  $$_/                    |__/          | $$        | $$
| $$$$  /$$$$  | $$   /$$$$$$$   /$$$$$$$ /$$  /$$$$$$ | $$$$$$$  /$$$$$$
| $$ $$/$$ $$  | $$  | $$__  $$ /$$_____/| $$ /$$__  $$| $$__  $$|_  $$_/
| $$  $$$| $$  | $$  | $$  \ $$|  $$$$$$ | $$| $$  \ $$| $$  \ $$  | $$
| $$\  $ | $$  | $$  | $$  | $$ \____  $$| $$| $$  | $$| $$  | $$  | $$ /$$
| $$ \/  | $$ /$$$$$$| $$  | $$ /$$$$$$$/| $$|  $$$$$$$| $$  | $$  |  $$$$/
|__/     |__/|______/|__/  |__/|_______/ |__/ \____  $$|__/  |__/   \___/
                                              /$$  \ $$
                                             |  $$$$$$/
                                              \______/

'''
print(ascii_art)

credit = 'Created by: Ryota Kimura Aekkapop Teamkratok and Kittisak Chaukum'

print(f'{credit}\n\n')

def load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column):
    df = pd.read_csv(filepath)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    df = pd.get_dummies(df, columns=categorical_features)
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
    return model # ส่งค่า model กลับ

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

        
        percent_complete = ((repeat + 1) / num_repeats) * 100 
        print(f'รอบที่ {repeat + 1} สำเร็จแล้ว: {percent_complete:.2f}% Done. ความแม่นยำเฉลี่ยอยู่ที่: {mean_accuracy:.2f}%')

    print(f'ความแม่นยำสูงสุดอยู่ที่: {best_accuracy:.2f}%') 
    return best_model, best_accuracy 


def predict_next_attacks(model, data_points, labels, best_accuracy, additional_columns):
    predictions = [] 
    for new_data in data_points: 
        prediction = model.predict(new_data.reshape(1, -1))
        predicted_attack = labels.columns[np.argmax(prediction)]
        print(f"การโจมตีครั้งต่อไปมีโอกาศที่จะเป็น: {predicted_attack} สูงถึง {best_accuracy:.2f}%\n") 
        data_dict = {col: val for col, val in zip(additional_columns, new_data)}
        data_dict['Predicted_Attack_Type'] = predicted_attack 
        data_dict['Accuracy'] = best_accuracy 
        predictions.append(data_dict) 
    return pd.DataFrame(predictions) 


def save_predict(prediction_df): 
    while True:  
        save_pre = input("คุณต้องการบันทึกการคาดการณ์การโจมตีครั้งล่าสุดหรือไม่? (y/n): ")
        if save_pre == 'y':
            filepath = 'Prediction.csv'
            if os.path.exists(filepath):
                prediction_df.to_csv(filepath, mode='a', index=False, header=False)
                print(f"การคาดการณ์การโจมตีครั้งล่าสุดถูกเพิ่มลงในไฟล์ {filepath} แล้ว")
            else:
                prediction_df.to_csv(filepath, index=False)
                print(f"การคาดการณ์การโจมตีครั้งล่าสุดถูกบันทึกลงในไฟล์ {filepath} แล้ว")
            break 
        elif save_pre == 'n':
            break 
        else:
            print("กรุณาป้อน y/n เท่านั้น โปรดลองอีกครั้ง") 



def plot_graph(accuracy_df, predicted_attack): 
    while True:  
        plot_gra = input("คุณต้องการพล็อตกราฟและบันทึกกราฟหรือไม่? (y/n): ") 
        if plot_gra == 'y': 
            plt.figure(figsize=(10, 6))
            palette = ["red" if attack_type == predicted_attack else "blue" for attack_type in accuracy_df['Predicted']]
            sns.barplot(data=accuracy_df, x='Predicted', y='Accuracy', hue='Predicted', palette=palette, dodge=False)
            plt.xlabel('Attack Type')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=0)
            plt.title('Prediction Accuracy for Each Attack Type')
            plt.legend(title='Predicted', loc='upper right', labels=['Predicted'])
            plt.tight_layout()
            plt.savefig('prediction_accuracy_graph.png')
            print("กราฟถูกบันทึกเป็นไฟล์รูปภาพ 'Plot_Prediction.png'")
            plt.close()  
            break  
        elif plot_gra == 'n':
            break  
        else:
            print("กรุณาป้อน y/n เท่านั้น โปรดลองอีกครั้ง") 



def aggregate_predictions(model, features, labels):
    predictions = model.predict(features)
    predicted_labels = np.argmax(predictions, axis=1) 
    true_labels = np.argmax(labels.values, axis=1) 

    
    attack_types = labels.columns
    predicted_attack_types = attack_types[predicted_labels]

    results_df = pd.DataFrame({'True': true_labels, 'Predicted': predicted_attack_types})

    
    results_df.replace({'True': dict(enumerate(attack_types)),
                        'Predicted': dict(enumerate(attack_types))}, inplace=True)

    
    accuracy_per_attack_type = results_df[results_df['True'] == results_df['Predicted']]\
                               .groupby('Predicted').size()\
                               .div(results_df.groupby('Predicted').size()) * 100

    
    accuracy_df = accuracy_per_attack_type.reset_index(name='Accuracy')
    return accuracy_df


def download_file_from_url(url): 
    response = requests.get(url, allow_redirects=True) 
    if response.status_code == 200: 
        print(f"Status Code: 200 (OK)")
        file_name = url.split('/')[-1] 
        save_pre = input(f"คุณต้องการดาวน์โหลดไฟล์คู่มือการป้องกันหรือไม่ (y/n): ")
        if save_pre.lower() == 'y': 
            with open(file_name, 'wb') as file: 
                file.write(response.content) 
            print(f"ไฟล์คู่มือการป้องกันถูกดาวน์โหลดเรียบร้อยแล้ว successfully.") 
        else:
            pass
    else:
        print(f"Status Code: {response.status_code} (Failed)") 

url = 'https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf' 


def main():
    input("Press Enter To Start:\n") 
    print("Starting the program...\n") 
    numeric_columns = ['Frequency', 'Duration', 'Targets'] 
    categorical_features = ['Severity'] 
    filepath = '/content/drive/MyDrive/Colab Notebooks/cyberattacks.csv' 

    if filepath is None: 
        print("โปรดระบุที่อยู่ของไฟล์ CSV โดยใช้ argument -fp หรือ --filepath\n") 
    else:
        target_column = 'Attack_Type' 
        features, labels = load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column) 
        best_model, best_accuracy = train_and_evaluate_model(features, labels, patience=10) 
        new_data = features[0] 
        all_features = features 
        all_labels = labels 
        accuracy_df = aggregate_predictions(best_model, all_features, all_labels) 

        additional_columns = ['Severity', 'Frequency', 'Duration', 'Targets'] 
        predicted_attack_df = predict_next_attacks(best_model, [new_data], labels, best_accuracy, additional_columns)  
        plot_graph(accuracy_df, predicted_attack_df['Predicted_Attack_Type'].values[0]) 
        save_predict(predicted_attack_df) 
        download_file_from_url(url) 

    input("\nPress Enter To Exit:") 


if __name__ == '__main__': 
    main() 