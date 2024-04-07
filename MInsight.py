#!/usr/bin/env python3
#MInsight
#Project: CPE 270 AI
#Machine Learning Model for Predicting Next Cyber Attack Type

import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks 
from colorama import Fore
from imblearn.over_sampling import SMOTE 
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import colorama


colorama.init(autoreset=True)

parser = argparse.ArgumentParser()
parser.add_argument('-nr', '--num_repeats', type=int, default=5, help='Number of repeats for model training') # สร้าง argument ชื่อ num_repeats โดยกำหนดค่า default เป็น 5 และเก็บค่าที่รับเข้ามาในตัวแปร args.num_repeats 
parser.add_argument('-fp', '--filepath', type=str, help='Location of CSV to need model training') # สร้าง argument ชื่อ filepath และเก็บค่าที่รับเข้ามาในตัวแปร args.filepath
args = parser.parse_args() # นำ argument ที่รับเข้ามาเก็บไว้ในตัวแปร args 


ascii_art = fr'''{Fore.CYAN}

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

credit = 'Created by: '

print(f'{credit}\n\n') 

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
        print(f'รอบที่ {repeat + 1} สำเร็จแล้ว: {Fore.GREEN}{percent_complete:.2f}% Done. {Fore.WHITE}ความแม่นยำเฉลี่ยอยู่ที่: {mean_accuracy:.2f}%') # แสดงข้อความเพื่อแสดงความคืบหน้าของการทำงานในแต่ละรอบ

    print(f'ความแม่นยำสูงสุดอยู่ที่: {Fore.GREEN}{best_accuracy:.2f}%') # แสดงค่าความแม่นยำที่ดีที่สุด
    return best_model, best_accuracy # ส่งค่า best_model และ best_accuracy กลับ


def predict_next_attacks(model, data_points, labels, best_accuracy, additional_columns): # สร้างฟังก์ชัน predict_next_attacks ที่รับพารามิเตอร์ 5 ตัวคือ model, data_points, labels, best_accuracy, additional_columns
    predictions = [] # สร้าง list ว่างเพื่อเก็บค่าการคาดการณ์การโจมตีในแต่ละครั้ง
    for new_data in data_points: # วนลูปเพื่อทำการคาดการณ์การโจมตีในแต่ละครั้ง
        prediction = model.predict(new_data.reshape(1, -1)) # คาดการณ์การโจมตีโดยใช้ model.predict() โดยใส่ข้อมูลใหม่ใน new_data และเก็บไว้ในตัวแปร prediction
        predicted_attack = labels.columns[np.argmax(prediction)] # คำนวณค่าการโจมตีที่คาดการณ์ได้จาก prediction และเก็บไว้ในตัวแปร predicted_attack
        print(f"การโจมตีครั้งต่อไปมีโอกาศที่จะเป็น: {Fore.GREEN}{predicted_attack} สูงถึง {Fore.GREEN}{best_accuracy:.2f}%\n") # แสดงข้อความเพื่อแสดงการคาดการณ์การโจมตี
        data_dict = {col: val for col, val in zip(additional_columns, new_data)} # สร้าง dict ที่มี key เป็น additional_columns และ value เป็นข้อมูลใน new_data และเก็บไว้ในตัวแปร data_dict
        data_dict['Predicted_Attack_Type'] = predicted_attack # เพิ่ม key 'Predicted_Attack_Type' ใน data_dict และใส่ค่า predicted_attack
        data_dict['Accuracy'] = best_accuracy # เพิ่ม key 'Accuracy' ใน data_dict และใส่ค่า best_accuracy
        predictions.append(data_dict) # เพิ่ม data_dict ลงใน predictions 
    return pd.DataFrame(predictions) # ส่งค่า predictions ในรูปของ DataFrame กลับ 


def save_predict(prediction_df): # สร้างฟังก์ชัน save_predict ที่รับพารามิเตอร์ 1 ตัวคือ prediction_df
    while True:  #เริ่มลูปอย่างต่อเนื่อง
        save_pre = input(f"คุณต้องการบันทึกการคาดการณ์การโจมตีครั้งล่าสุดหรือไม่? (y/n): ")
        if save_pre == 'y':
            #เช็คว่าไฟล์ Prediction.csv มีอยู่หรือไม่ ถ้ามีให้เพิ่มข้อมูลลงไป ถ้าไม่มีให้สร้างไฟล์ใหม่
            filepath = 'Prediction.csv'
            if os.path.exists(filepath): 
                prediction_df.to_csv(filepath, mode='a', index=False, header=False)
                print(f"\nการคาดการณ์การโจมตีครั้งล่าสุดถูกเพิ่มลงในไฟล์ {filepath} แล้ว")
            else:
                prediction_df.to_csv(filepath, index=False)
                print(f"\nการคาดการณ์การโจมตีครั้งล่าสุดถูกบันทึกลงในไฟล์ {filepath} แล้ว")
            break # ออกจากลูป
        elif save_pre == 'n':
            break # ออกจากลูป
        else:
            print("กรุณาป้อน y/n เท่านั้น โปรดลองอีกครั้ง") # แสดงข้อความเพื่อแจ้งให้ผู้ใช้ป้อน y/n เท่านั้น

    
        
def plot_graph(accuracy_df, predicted_attack): # สร้างฟังก์ชัน plot_graph ที่รับพารามิเตอร์ 2 ตัวคือ accuracy_df, predicted_attack
    while True:  #เริ่มลูปอย่างต่อเนื่อง
        plot_gra = input("คุณต้องการพล็อตกราฟหรือไม่? (y/n): ") # รับค่าจากผู้ใช้ว่าต้องการพล็อตกราฟหรือไม่
        if plot_gra == 'y': # ถ้าผู้ใช้ต้องการพล็อตกราฟ 
            plt.figure(figsize=(10, 6)) # สร้างกราฟขนาด 10x6 และเก็บไว้ในตัวแปร plt 
            # สร้างกราฟแท่งโดยใช้ sns.barplot() โดยให้ x เป็น 'Predicted', y เป็น 'Accuracy', hue เป็น 'Predicted', palette เป็นสีแดงถ้า attack_type เท่ากับ predicted_attack และสีน้ำเงินถ้าไม่เท่ากัน
            palette = ["red" if attack_type == predicted_attack else "blue" for attack_type in accuracy_df['Predicted']] 
            sns.barplot(data=accuracy_df, x='Predicted', y='Accuracy', hue='Predicted', palette=palette, dodge=False)
            plt.xlabel('Attack Type')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=0)
            plt.title('Prediction Accuracy for Each Attack Type')
            plt.legend(title='Predicted', loc='upper right', labels=['Predicted'])
            plt.tight_layout()
            plt.show()
            break  # ออกจากลูป
        elif plot_gra == 'n':
            break  # ออกจากลูป
        else:
            print("กรุณาป้อน y/n เท่านั้น โปรดลองอีกครั้ง") # แสดงข้อความเพื่อแจ้งให้ผู้ใช้ป้อน y/n เท่านั้น
            
        
        
def aggregate_predictions(model, features, labels): # สร้างฟังก์ชัน aggregate_predictions ที่รับพารามิเตอร์ 3 ตัวคือ model, features, labels
    predictions = model.predict(features) # คำนวณค่าการคาดการณ์การโจมตีโดยใช้ model.predict() โดยใส่ข้อมูลใน features และเก็บไว้ในตัวแปร predictions
    predicted_labels = np.argmax(predictions, axis=1) # คำนวณค่าการโจมตีที่คาดการณ์ได้จาก predictions และเก็บไว้ในตัวแปร predicted_labels
    true_labels = np.argmax(labels.values, axis=1) # คำนวณค่าการโจมตีจริงจาก labels และเก็บไว้ในตัวแปร true_labels

    #นำข้อมูลที่ได้มาแปลงเป็นชื่อของการโจมตี และเก็บไว้ในตัวแปร predicted_attack_types และ attack_types 
    attack_types = labels.columns
    predicted_attack_types = attack_types[predicted_labels]

    #สร้าง DataFrame ที่มีคอลัมน์ 'True' และ 'Predicted' โดยให้ค่าในคอลัมน์ 'True' เป็น true_labels และ 'Predicted' เป็น predicted_attack_types 
    results_df = pd.DataFrame({'True': true_labels, 'Predicted': predicted_attack_types})
    
    #แปลงค่าในคอลัมน์ 'True' และ 'Predicted' ให้เป็นชื่อของการโจมตี และเก็บไว้ใน results_df 
    results_df.replace({'True': dict(enumerate(attack_types)),
                        'Predicted': dict(enumerate(attack_types))}, inplace=True)
    
    #คำนวณความแม่นยำของการคาดการณ์ของแต่ละการโจมตี และเก็บไว้ใน accuracy_per_attack_type 
    accuracy_per_attack_type = results_df[results_df['True'] == results_df['Predicted']]\
                               .groupby('Predicted').size()\
                               .div(results_df.groupby('Predicted').size()) * 100
    
    #แปลง accuracy_per_attack_type ให้เป็น DataFrame และเก็บไว้ใน accuracy_df และเพิ่มคอลัมน์ 'Accuracy' 
    accuracy_df = accuracy_per_attack_type.reset_index(name='Accuracy')
    
    #แสดงค่าความแม่นยำของการคาดการณ์ของแต่ละการโจมตี 
    
    return accuracy_df


def download_file_from_url(url): # สร้างฟังก์ชัน download_file_from_url ที่รับพารามิเตอร์ 1 ตัวคือ url
    response = requests.get(url, allow_redirects=True) # ดาวน์โหลดไฟล์จาก url โดยใช้ requests.get() และเก็บไว้ในตัวแปร response 
    if response.status_code == 200: # ถ้าสถานะของ response เป็น 200
        print(f"Status Code: 200 {Fore.GREEN}(OK)") 
        file_name = url.split('/')[-1] # แยกชื่อไฟล์จาก url และเก็บไว้ในตัวแปร file_name 
        save_pre = input(f"คุณต้องการดาวน์โหลดไฟล์คู่มือการป้องกันหรือไม่ (y/n): ") # รับค่าจากผู้ใช้ว่าต้องการดาวน์โหลดไฟล์หรือไม่
        if save_pre.lower() == 'y': # ถ้าผู้ใช้ต้องการดาวน์โหลดไฟล์
            with open(file_name, 'wb') as file: # สร้างไฟล์ใหม่เพื่อเก็บไฟล์ที่ดาวน์โหลด และเก็บไว้ในตัวแปร file 
                file.write(response.content) # เขียนข้อมูลที่ดาวน์โหลดลงในไฟล์ที่สร้างขึ้น
            print(f"ไฟล์คู่มือการป้องกันถูกดาวน์โหลดเรียบร้อยแล้ว {Fore.GREEN}successfully.") # แสดงข้อความเพื่อแจ้งให้ผู้ใช้ทราบว่าไฟล์ถูกดาวน์โหลดเรียบร้อยแล้ว
        else:
            pass 
    else:
        print(f"Status Code: {response.status_code} {Fore.RED}(Failed)") # แสดงข้อความเพื่อแจ้งให้ผู้ใช้ทราบว่าไม่สามารถดาวน์โหลดไฟล์ได้

url = 'https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf' # กำหนด url ของไฟล์คู่มือการป้องกัน



if __name__ == '__main__': # สร้างบล็อกเพื่อเริ่มโปรแกรม
    input("Press Enter To Start:\n") #กด Enter เพื่อเริ่มโปรแกรม
    print("Starting the program...\n") #แสดงข้อความเพื่อแจ้งเตือนว่าโปรแกรมกำลังเริ่มทำงาน
    numeric_columns = ['Frequency', 'Duration', 'Targets'] #กำหนดคอลัมน์ที่เป็นตัวเลข 
    categorical_features = ['Severity'] #กำหนดคอลัมน์ที่เป็นข้อความ
    filepath = args.filepath # กำหนดค่า filepath จาก args.filepath 

    if filepath is None: # ถ้าไม่มี filepath ที่ระบุ
        print("โปรดระบุที่อยู่ของไฟล์ CSV โดยใช้ argument -fp หรือ --filepath\n") # แสดงข้อความเพื่อแจ้งให้ผู้ใช้ระบุที่อยู่ของไฟล์ CSV
    else:
        target_column = 'Attack_Type' # กำหนดคอลัมน์ที่เป็น target ในการคาดการณ์ 
        features, labels = load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column) # โหลดข้อมูลและทำการประมวลผลข้อมูลโดยใช้ฟังก์ชัน load_and_preprocess_data และเก็บไว้ใน features และ labels
        best_model, best_accuracy = train_and_evaluate_model(features, labels, patience=10) # ฝึกและประเมินโมเดลโดยใช้ฟังก์ชัน train_and_evaluate_model และเก็บค่า best_model และ best_accuracy 
        new_data = features[0] # กำหนดข้อมูลใหม่ที่จะใช้ในการคาดการณ์โจมตี โดยใช้ข้อมูลใน features ที่ index เป็น 0 และเก็บไว้ในตัวแปร new_data 
        all_features = features # กำหนด all_features ให้เป็น features ทั้งหมด 
        all_labels = labels # กำหนด all_labels ให้เป็น labels ทั้งหมด
        accuracy_df = aggregate_predictions(best_model, all_features, all_labels) # คำนวณความแม่นยำของการคาดการณ์ของแต่ละการโจมตี โดยใช้ฟังก์ชัน aggregate_predictions และเก็บไว้ใน accuracy_df 

        additional_columns = ['Severity', 'Frequency', 'Duration', 'Targets']  # กำหนดคอลัมน์ที่เพิ่มเติมที่จะใช้ในการคาดการณ์โจมตี
        predicted_attack_df = predict_next_attacks(best_model, [new_data], labels, best_accuracy, additional_columns)  # คาดการณ์การโจมตีครั้งต่อไป โดยใช้ฟังก์ชัน predict_next_attacks และเก็บไว้ใน predicted_attack_df 
        plot_graph(accuracy_df, predicted_attack_df['Predicted_Attack_Type'].values[0]) # พล็อตกราฟของความแม่นยำของการคาดการณ์ของแต่ละการโจมตี โดยใช้ฟังก์ชัน plot_graph
        save_predict(predicted_attack_df) # บันทึกการคาดการณ์การโจมตีครั้งล่าสุด โดยใช้ฟังก์ชัน save_predict 
        download_file_from_url(url) # ดาวน์โหลดไฟล์คู่มือการป้องกัน โดยใช้ฟังก์ชัน download_file_from_url

    input("\nPress Enter To Exit:") #กด Enter เพื่อออกจากโปรแกรม 



