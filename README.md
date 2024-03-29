
# MInsight 

## Project for CPE 270 AI 

### ขั้นตอนการใช้งานโปรแกรม MInsight

#### 1.) Install PIP ดังนี้
        1. pip install numpy หรือใช้ pip3 install numpy
        2. pip install pandas
        3. pip install imblearn
        4. pip install scikit-learn
        5. pip install smote
        6. pip install tensorflow

#### 2.) นำเข้าไฟล์ CSV ผ่าน args (parser) ใน Terminal
        python3 MInsight.py -fp [ที่อยู่ File CSV ที่ต้องการ Train Model] 
        python3 MInsight.py -fp xxx/folder/xxxx.csv
        python3 MInsight.py --filepath xxx/folder/xxxx.csv 

        หรือใช้ python MInsight.py -fp [ที่อยู่ File CSV ที่ต้องการ Train Model] 

#### 3.) กำหนดจำนวนรอบที่ต้องการ Train Model ผ่าน args (parser) ใน Terminal โดยที่จำนวนรอบเริ่มต้นคือ 5
        python3 MInsight.py -fp xxx/folder/xxxx.csv -np [จำนวนรอบที่ต้องการ]
        python3 MInsight.py -fp xxx/folder/xxxx.csv -np 10
        python3 MInsight.py --filepath xxx/folder/xxxx.csv --num_repeats 10

        หรือใ่ช้ python MInsight.py -fp xxx/folder/xxxx.csv -np [จำนวนรอบที่ต้องการ]





