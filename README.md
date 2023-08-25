# YOLO Model-Evaluation

## Steps:

    1.Data Gathering
    2.Data Pre Processing
    3. Model Training
    4.Model Evaluation
    5.Infrence

## 3. Model Training

    > Run Command: yolo task=detect  mode=train model=yolov8l.pt epochs=50 imgsz=1280 data=yolo8.yaml --batch=8

## 4. Model Evaluation:

    1> First get 9 diffrent json file of detected bounding boxes you can get this files by changing confidence threshold in this and running this command.

    Run Command: yolo detect val model=./runs/detect/train13/weights/best.pt save_json=True conf=0.8 batch=10

    2> Now place you groud truth (real labels) in ground-truth folder.
    
    3> Now in Model run json2txt.py file for converting those json file in to text files. This script convert those json file in to text file and store them into predicted folder according to their confidence threshold.

    Run commnad: python json2txt.py 

     4> Now for getting evaluation as excel sheet run this command. by executing thi command you'll get excel file name data which contains all the information about models performace at diffrence confidence thresholds.

      Run Command: python calculate_states.py


## Note: You can change the batch size as per you gpu memory.
