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

parser = argparse.ArgumentParser()
parser.add_argument('-nr', '--num_repeats', type=int, default=5, help='Number of repeats for model training') # สร้าง argument ชื่อ num_repeats โดยกำหนดค่า default เป็น 5 และเก็บค่าที่รับเข้ามาในตัวแปร args.num_repeats 
parser.add_argument('-fp', '--filepath', type=str, help='Path to the CSV file')
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

    print(f'ความแม่นยำที่ดีที่สุดอยู่ที่: {best_accuracy:.2f}%') # แสดงค่าความแม่นยำที่ดีที่สุด
    return best_model, best_accuracy # ส่งค่า best_model และ best_accuracy กลับ


def predict_next_attack(model, new_data, labels, best_accurary): # สร้างฟังก์ชัน predict_next_attack ที่รับพารามิเตอร์ 3 ตัวคือ model, new_data, labels
    prediction = model.predict(new_data.reshape(1, -1)) # ทำนายคลาสของ new_data โดยใช้ model.predict() และเก็บไว้ในตัวแปร prediction
    predicted_attack = labels.columns[np.argmax(prediction)] # หาชื่อคลาสที่ทำนายได้จาก prediction และเก็บไว้ในตัวแปร predicted_attack
    print(f"การโจมตีครั้งต่อไปมีโอกาศที่จะเป็น: {predicted_attack} ถึง {best_accurary:.2f}%") # แสดงข้อความที่บอกคลาสที่ทำนายได้

if __name__ == '__main__': # ตรวจสอบว่าโปรแกรมถูกเรียกใช้โดยตรงหรือไม่

    numeric_columns = ['Frequency', 'Duration', 'Targets'] # กำหนดคอลัมน์ที่เป็นตัวเลขใน numeric_columns ในที่นี้คือ ['Frequency', 'Duration', 'Targets'] 
    categorical_features = ['Severity'] # กำหนดคอลัมน์ที่เป็นข้อความใน categorical_features ในที่นี้คือ ['Severity']
    filepath = args.filepath # กำหนด path ของไฟล์ CSV ใน filepath โดยใช้ args.filepath
    target_column = 'Attack_Type' # กำหนดคอลัมน์ที่เป็น target ใน target_column ในที่นี้คือ 'Attack_Type' 

    features, labels = load_and_preprocess_data(filepath, numeric_columns, categorical_features, target_column) # โหลดข้อมูลและทำการประมวลผลข้อมูลโดยใช้ฟังก์ชัน load_and_preprocess_data และเก็บไว้ใน features, labels
    best_model, best_accurary = train_and_evaluate_model(features, labels, patience=10) # ฝึกและประเมินโมเดลโดยใช้ฟังก์ชัน train_and_evaluate_model และเก็บโมเดลที่ดีที่สุดไว้ใน best_model 
    new_data = features[0] # กำหนดข้อมูลใหม่ที่จะทำนายใน new_data โดยใช้ข้อมูลใน features ที่ index เท่ากับ 0
    predict_next_attack(best_model, new_data, labels, best_accurary) # ทำนายคลาสของข้อมูลใหม่โดยใช้ฟังก์ชัน predict_next_attack
    
