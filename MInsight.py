#!/usr/bin/env python3
#MInsight
#Project: CPE 270 AI
#Machine Learning Model for Predicting Next Cyber Attack Type

import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks 
from imblearn.over_sampling import SMOTE 
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, send_file, jsonify
import requests


parser = argparse.ArgumentParser()
parser.add_argument('-nr', '--num_repeats', type=int, default=5, help='จำนวนครั้งที่ต้องการ Train Model') # สร้าง argument ชื่อ num_repeats โดยกำหนดค่า default เป็น 5 และเก็บค่าที่รับเข้ามาในตัวแปร args.num_repeats 
parser.add_argument('-fp', '--filepath', type=str, help='ที่อยู่ของไฟล์ CSV') # สร้าง argument ชื่อ filepath และเก็บค่าที่รับเข้ามาในตัวแปร args.filepath
args = parser.parse_args() # นำ argument ที่รับเข้ามาเก็บไว้ในตัวแปร args 


ascii_art = r'''

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
print(ascii_art)  # Display the ASCII art

credit = 'Created by: '

print(f'{credit}\n\n') # Display the credit/license information

def load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column): # สร้างฟังก์ชัน load_and_preprocess_data ที่รับพารามิเตอร์ 4 ตัวคือ filepath, numeric_columns, categorical_features, target_column
    df = pd.read_csv(filepath) # อ่านไฟล์ CSV จาก filepath และเก็บไว้ในตัวแปร df
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median()) # ใส่ค่าเฉลี่ยของคอลัมน์ที่เป็นตัวเลขใน numeric_columns ในที่นี้คือ ['Frequency', 'Duration', 'Targets'] ในคอลัมน์ที่มีค่าว่าง
    df = pd.get_dummies(df, columns=categorical_features) # แปลงข้อมูลที่เป็นข้อความใน categorical_features ให้เป็นข้อมูลที่เป็นตัวเลข และเก็บไว้ใน df โดยใช้ pd.get_dummies() ในการแปลงข้อมูล 
    features = df.drop(target_column, axis=1) # ลบคอลัมน์ target_column ออกจาก features และเก็บไว้ในตัวแปร features โดยใช้ drop() ในการลบคอลัมน์ target_column ออก 
    labels = pd.get_dummies(df[target_column]) # แปลงข้อมูลใน target_column ให้เป็นข้อมูลที่เป็นตัวเลข และเก็บไว้ใน labels โดยใช้ pd.get_dummies() ในการแปลงข้อมูล
    scaler = StandardScaler() # สร้างออบเจกต์ StandardScaler และเก็บไว้ในตัวแปร scaler 
    features = scaler.fit_transform(features) # ปรับข้อมูลใน features ให้มี mean = 0 และ standard deviation = 1 และเก็บไว้ในตัวแปร features โดยใช้ fit_transform() ในการปรับข้อมูล 
    return features, labels # ส่งค่า features และ labels กลับ 

def create_model(num_features, num_classes): # สร้างฟังก์ชัน create_model ที่รับพารามิเตอร์ 2 ตัวคือ num_features, num_classes
    model = models.Sequential([ # สร้างโมเดล Sequential และเก็บไว้ในตัวแปร model 
        layers.Input(shape=(num_features,)), # สร้างชั้น Input โดยกำหนด shape ของข้อมูลเป็น (num_features,) และเพิ่มชั้น Input นี้ในโมเดล model 
        layers.Dense(128, activation='relu'), # สร้างชั้น Dense ที่มี 128 units และใช้ activation function เป็น 'relu' และเพิ่มชั้น Dense นี้ในโมเดล model 
        layers.Dropout(0.3), # สร้างชั้น Dropout ที่มีค่า dropout rate เท่ากับ 0.3 และเพิ่มชั้น Dropout นี้ในโมเดล model 
        layers.Dense(64, activation='relu'), # สร้างชั้น Dense ที่มี 64 units และใช้ activation function เป็น 'relu' และเพิ่มชั้น Dense นี้ในโมเดล model 
        layers.Dropout(0.3), # สร้างชั้น Dropout ที่มีค่า dropout rate เท่ากับ 0.3 และเพิ่มชั้น Dropout นี้ในโมเดล model 
        layers.Dense(num_classes, activation='softmax') # สร้างชั้น Dense ที่มี units เท่ากับ num_classes และใช้ activation function เป็น 'softmax' และเพิ่มชั้น Dense นี้ในโมเดล model 
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # คอมไพล์โมเดล model โดยใช้ optimizer เป็น 'adam', loss function เป็น 'categorical_crossentropy', และ metrics เป็น ['accuracy']
    return model # ส่งค่า model กลับ

def train_and_evaluate_model(features, labels, patience=10, num_repeats=args.num_repeats): # สร้างฟังก์ชัน train_and_evaluate_model ที่รับพารามิเตอร์ 3 ตัวคือ features, labels, patience=10, num_repeats=5
    best_accuracy = 0 # กำหนดค่า best_accuracy เป็น 0
    best_model = None # กำหนดค่า best_model เป็น None

    for repeat in range(num_repeats): # วนลูปเพื่อทำการฝึกโมเดล num_repeats ครั้ง
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # สร้างออบเจกต์ StratifiedKFold โดยกำหนด n_splits เป็น 5, shuffle เป็น True, และ random_state เป็น 42 และเก็บไว้ในตัวแปร kfold 
        cvscores = [] # สร้าง list ว่างเพื่อเก็บค่าความแม่นยำของโมเดลที่ฝึกในแต่ละ fold 

        for fold, (train, test) in enumerate(kfold.split(features, np.argmax(labels.values, axis=1))): # วนลูปเพื่อทำการฝึกและประเมินโมเดลในแต่ละ fold โดยใช้ enumerate เพื่อเก็บค่า fold และ (train, test) และเก็บค่าในตัวแปร train, test 
            smote = SMOTE(random_state=42) # สร้างออบเจกต์ SMOTE โดยกำหนด random_state เป็น 42 และเก็บไว้ในตัวแปร smote 
            X_train, y_train = smote.fit_resample(features[train], np.argmax(labels.values[train], axis=1)) # ใช้ smote.fit_resample() เพื่อสร้างข้อมูลใหม่จากข้อมูลใน features[train] และ labels.values[train] และเก็บไว้ในตัวแปร X_train, y_train
            y_train = pd.get_dummies(y_train) # แปลงข้อมูลใน y_train ให้เป็นข้อมูลที่เป็นตัวเลข และเก็บไว้ใน y_train โดยใช้ pd.get_dummies() ในการแปลงข้อมูล 

            model = create_model(num_features=features.shape[1], num_classes=labels.shape[1]) # สร้างโมเดลโดยใช้ฟังก์ชัน create_model และเก็บไว้ในตัวแปร model 
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True) # สร้างออบเจกต์ EarlyStopping โดยกำหนด monitor เป็น 'val_loss', patience เป็น patience, และ restore_best_weights เป็น True และเก็บไว้ในตัวแปร early_stopping
            model.fit(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping], verbose=0) # ฝึกโมเดล model โดยใช้ข้อมูล X_train, y_train ใน 20 epochs และ validation_split เท่ากับ 0.2 และใช้ callbacks=[early_stopping] ในการหยุดการฝึกโมเดล และ verbose เท่ากับ 0
            scores = model.evaluate(features[test], labels.values[test], verbose=0) # ประเมินโมเดลโดยใช้ข้อมูล features[test] และ labels.values[test] และเก็บค่าความแม่นยำไว้ใน scores 
            cvscores.append(scores[1] * 100) # เพิ่มค่าความแม่นยำลงใน cvscores โดยคูณด้วย 100


        mean_accuracy = np.mean(cvscores) # คำนวณค่าความแม่นยำเฉลี่ยจาก cvscores และเก็บไว้ในตัวแปร mean_accuracy
        if mean_accuracy > best_accuracy: # ถ้าความแม่นยำเฉลี่ยมากกว่า best_accuracy 
            best_accuracy = mean_accuracy # กำหนดค่า best_accuracy เป็น mean_accuracy
            best_model = model # กำหนดค่า best_model เป็น model
 
        # Print the overall progress for each iteration
        percent_complete = ((repeat + 1) / num_repeats) * 100 # คำนวณเปอร์เซ็นต์ของการทำงานทั้งหมดในแต่ละรอบ 
        print(f'รอบที่ {repeat + 1} สำเร็จแล้ว: {percent_complete:.2f}% Done. ความแม่นยำเฉลี่ยอยู่ที่: {mean_accuracy:.2f}%') # แสดงข้อความเพื่อแสดงความคืบหน้าของการทำงานในแต่ละรอบ

    print(f'ความแม่นยำสูงสุดอยู่ที่: {best_accuracy:.2f}%') # แสดงค่าความแม่นยำที่ดีที่สุด
    return best_model, best_accuracy # ส่งค่า best_model และ best_accuracy กลับ


def predict_next_attacks(model, data_points, labels, best_accuracy, additional_columns):
    predictions = []
    for new_data in data_points:
        prediction = model.predict(new_data.reshape(1, -1))
        predicted_attack = labels.columns[np.argmax(prediction)]
        print(f"การโจมตีครั้งต่อไปมีโอกาศที่จะเป็น: {predicted_attack} สูงถึง {best_accuracy:.2f}%")
        data_dict = {col: val for col, val in zip(additional_columns, new_data)}
        data_dict['Predicted_Attack_Type'] = predicted_attack
        data_dict['Accuracy'] = best_accuracy
        predictions.append(data_dict)
    return pd.DataFrame(predictions)


def save_predict(prediction_df):
    while True:  # Start an infinite loop
        save_pre = input("คุณต้องการบันทึกการคาดการณ์การโจมตีครั้งล่าสุดหรือไม่? (y/n): ")
        if save_pre == 'y':
            # Check if the file exists and append the new prediction, otherwise create a new file
            filepath = 'Prediction.csv'
            if os.path.exists(filepath):
                prediction_df.to_csv(filepath, mode='a', index=False, header=False)
                print(f"\nการคาดการณ์การโจมตีครั้งล่าสุดถูกเพิ่มลงในไฟล์ {filepath} แล้ว")
            else:
                prediction_df.to_csv(filepath, index=False)
                print(f"\nการคาดการณ์การโจมตีครั้งล่าสุดถูกบันทึกลงในไฟล์ {filepath} แล้ว")
            break  # Exit the loop
        elif save_pre == 'n':
            break  # Exit the loop
        else:
            print("คุณได้ป้อนค่าที่ไม่ถูกต้อง โปรดลองอีกครั้ง")

    
        
def plot_graph(accuracy_df, predicted_attack):
    while True:  # Start an infinite loop
        plot_gra = input("\nคุณต้องการพล็อตกราฟหรือไม่? (y/n):")
        if plot_gra == 'y':
            plt.figure(figsize=(10, 6))
        # Use a different color for new attack types
            palette = ["red" if attack_type == predicted_attack else "blue" for attack_type in accuracy_df['Predicted']]
            sns.barplot(data=accuracy_df, x='Predicted', y='Accuracy', hue='Predicted', palette=palette, dodge=False)
            plt.xlabel('Attack Type')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=0)
            plt.title('Prediction Accuracy for Each Attack Type')
            plt.legend(title='Predicted', loc='upper right', labels=['Predicted'])
            plt.tight_layout()
            plt.show()
            break  # Exit the loop
        elif plot_gra == 'n':
            break  # Exit the loop
        else:
            print("คุณได้ป้อนค่าที่ไม่ถูกต้อง โปรดลองอีกครั้ง")
            
        
        
def aggregate_predictions(model, features, labels):
    predictions = model.predict(features)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels.values, axis=1)

    # Map predicted labels back to the original attack type names
    attack_types = labels.columns
    predicted_attack_types = attack_types[predicted_labels]

    # Create a DataFrame with true and predicted labels
    results_df = pd.DataFrame({'True': true_labels, 'Predicted': predicted_attack_types})
    
    # Map numerical labels back to categorical labels
    results_df.replace({'True': dict(enumerate(attack_types)),
                        'Predicted': dict(enumerate(attack_types))}, inplace=True)
    
    # Calculate accuracy for each attack type
    accuracy_per_attack_type = results_df[results_df['True'] == results_df['Predicted']]\
                               .groupby('Predicted').size()\
                               .div(results_df.groupby('Predicted').size()) * 100
    
    # Create a DataFrame with the calculated accuracy
    accuracy_df = accuracy_per_attack_type.reset_index(name='Accuracy')
    
    # Add a column to indicate whether the attack type is new
    
    return accuracy_df


def download_file_from_url(url):
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        file_name = url.split('/')[-1]
        save_pre = input(f"คุณต้องการดาวน์โหลดไฟล์คู่มือการป้องกันหรือไม่ (y/n): ")
        if save_pre.lower() == 'y':
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f"ไฟล์คู่มือการป้องกันถูกดาวน์โหลดเรียบร้อยแล้ว successfully.")
        else:
            pass
    else:
        print(f"ไม่สามารถดาวน์โหลดไฟล์คู่มือการป้องกันได้ โปรดลองอีกครั้ง {response.status_code}")

url = 'https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf'





if __name__ == '__main__': # Check if the script is being run directly
    input("Press Enter To Start:\n") # Prompt to start the program
    print("Starting the program...\n") # Display message indicating the start of the program
    numeric_columns = ['Frequency', 'Duration', 'Targets'] # Define numeric columns
    categorical_features = ['Severity'] # Define categorical features
    filepath = args.filepath # Get the filepath from the command-line arguments

    if filepath is None:
        print("โปรดระบุที่อยู่ของไฟล์ CSV โดยใช้ argument -fp หรือ --filepath\n")
    else:
        target_column = 'Attack_Type' # Define the target column
        features, labels = load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column) # Load and preprocess the data
        best_model, best_accuracy = train_and_evaluate_model(features, labels, patience=10) # Train and evaluate the model
        new_data = features[0] # Define new data for prediction
        all_features = features # Define all features for prediction
        all_labels = labels # Define all labels for prediction
        accuracy_df = aggregate_predictions(best_model, all_features, all_labels) # Aggregate predictions

        additional_columns = ['Severity', 'Frequency', 'Duration', 'Targets']  # Define additional columns
        predicted_attack_df = predict_next_attacks(best_model, [new_data], labels, best_accuracy, additional_columns)  # Update this line
        plot_graph(accuracy_df, predicted_attack_df['Predicted_Attack_Type'].values[0]) # Plot the graph
        save_predict(predicted_attack_df) # Save the prediction
        download_file_from_url(url)

    input("\nPress Enter To Exit:") # Prompt to exit the program


        
    

