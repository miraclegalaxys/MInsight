# MInsight

## Project for CPE 270 AI

## Deep Learning Model

### ขั้นตอนการใช้งานโปรแกรม MInsight

#### 1.) Install PIP ดังนี้

        1. pip install numpy หรือใช้ pip3 install numpy
        2. pip install pandas
        3. pip install imblearn
        4. pip install scikit-learn
        5. pip install smote
        6. pip install tensorflow
        7. pip install imbalanced-learn
        8. pip install matplotlib
        9. pip install seaborn
       10. pip install requests
       11. pip install colorama

        หรือ pip install numpy scikit-learn tensorflow imbalanced-learn pandas matplotlib seaborn requests smote colorama

#### 2.) นำเข้าไฟล์ CSV ผ่าน args (parser) ใน Terminal

        python3 MInsight.py -fp [ที่อยู่ File CSV ที่ต้องการ Train Model]
        python3 MInsight.py -fp xxx/folder/xxxx.csv
        python3 MInsight.py --filepath xxx/folder/xxxx.csv

        หรือใช้ python MInsight.py -fp [ที่อยู่ File CSV ที่ต้องการ Train Model]

#### 3.) กำหนดจำนวนรอบที่ต้องการ Train Model ผ่าน args (parser) ใน Terminal โดยที่จำนวนรอบเริ่มต้นคือ 5

        python3 MInsight.py -fp xxx/folder/xxxx.csv -nr [จำนวนรอบที่ต้องการ]
        python3 MInsight.py -fp xxx/folder/xxxx.csv -nr 10
        python3 MInsight.py --filepath xxx/folder/xxxx.csv --num_repeats 10

        หรือใ่ช้ python MInsight.py -fp xxx/folder/xxxx.csv -nr [จำนวนรอบที่ต้องการ]

### การใช้งานโปรแกรมใน Google Colab

        1. เข้าสู่ระบบ Google Colab โดยเข้าไปที่เว็บไซต์ Google Colab
        2. สร้าง Notebook ใหม่หรือเปิด Notebook ที่มีอยู่แล้วได้ เลือก "File" > "New notebook" เพื่อสร้าง Notebook ใหม่ หรือ "File" > "Open notebook" เพื่อเปิด Notebook ที่มีอยู่แล้วจาก Google Drive หรือ GitHub หรืออื่น ๆ
        3. ลง library !pip install numpy scikit-learn tensorflow imbalanced-learn pandas matplotlib seaborn requests smote ในเซลล์โค้ดอันแรกของ Notebook
           สามารถรันโปรแกรมโดยคลิกที่เครื่องหมาย "Play" ที่ด้านซ้ายของเซลล์โค้ด
           หรือใช้ปุ่มลัด Ctrl+Enter (Cmd+Enter สำหรับ macOS) เพื่อรันโค้ด !pip install
        4. ในเซลล์โค้ดใหม่ของ Notebook คัดลอกโค้ดในไฟล์ชื่อ "MInsight_Colab.py" แล้ววางโค้ดทั้งหมดลงในเซลล์รันโค้ดนั้น **กรุณาอ่าน # ในโค้ด**
        5. เมื่อโค้ดถูกวางลงในเซลล์โค้ดแล้ว สามารถรันโปรแกรมโดยคลิกที่เครื่องหมาย "Play" ที่ด้านซ้ายของเซลล์โค้ด
           หรือใช้ปุ่มลัด Ctrl+Enter (Cmd+Enter สำหรับ macOS) เพื่อรันโค้ด
        6. ผลลัพธ์จะปรากฏในส่วนของเอาต์พุตด้านล่างของเซลล์โค้ด หรือถ้ามีการพิมพ์ข้อความผ่านคำสั่ง print ก็จะแสดงผลเช่นกัน
        7. สามารถทำซ้ำขั้นตอนการเขียนโค้ดและรันโปรแกรมเพิ่มเติมได้ตามต้องการ เพียงเพิ่มเซลล์โค้ดใหม่หรือใช้เซลล์โค้ดที่มีอยู่แล้ว
        8. เมื่อคุณเสร็จสิ้นการทำงานและต้องการบันทึก Notebook ให้เลือก "File" > "Save" หรือ "File" > "Save a copy in Drive" เพื่อบันทึกไฟล์ Notebook ไว้ใน Google Drive
        9. เมื่อคุณเสร็จสิ้นการใช้งาน ไม่ลืมปิด Notebook เพื่อประหยัดทรัพยากรคอมพิวเตอร์ของคุณครับ ให้เลือก "File" > "Close notebook"
