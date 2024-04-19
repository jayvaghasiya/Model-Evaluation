from utils import ConfusionMatrix
import torch
import pandas as pd
import numpy as np
import os
classes = ["Drink bottle","Milk bottle","Other hard plastic","Soft plastic","Paper","Cardboard","Aluminum Cans","Food Cans", "Left Hand","Tetrapacks","Food tray and cups","Crisps bags","Aluminium trays","Foil","Coffe Cups","Aerosols","Other Metal","Cloth","Wood","Right Hand","Drink bottle other","Cosmetic Tubes","Black Plastic","Ambigous","Residual"]


def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H
    voc = []
    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))
    return [int(v) for v in voc]


def read_yolo_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        values = line.strip().split()
        
        if len(values) == 5:  # Assuming YOLO format with class_id, x_center, y_center, width, height
            k = [float(val) for val in values]
            k[1:] = yoloFormattocv(k[1],k[2],k[3],k[4],1200,1920)
            labels.append(k)

    return labels


def read_pred_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        values = line.strip().split()
        if len(values) == 6:  # Assuming YOLO format with class_id, x_center, y_center, width, height
            k = [float(values[1]),float(values[2]),float(values[3]),float(values[4]),float(values[5]),float(values[0])]
            k[:4] = yoloFormattocv(k[0],k[1],k[2],k[3],1200,1920)
            labels.append(k)

    return labels


def generate_sheet(gt_path,classes):
    temp = [[f"TP_{i/10}",f"FP_{i/10}",f"Precision_{i/10}",f"Recall_{i/10}",f"F1_{i/10}"] for i in range(1,10)]
    cf_heads = []
    for i in temp:
        cf_heads.extend(i)
    performance = {'Class':classes}
    for i in cf_heads:
        performance[i]= []
    for conf in range(1,10):
        gt_labels= [i for i in os.listdir(gt_path) if i.endswith(".txt")]

        confusion_matrix = ConfusionMatrix(nc=len(classes), conf=conf/10, iou_thres=0.45)

        for gt_file in gt_labels:
            gt_data = read_yolo_labels(f"{gt_path}/{gt_file}")
            # print(f"{gt_path}/{gt_file}")
            try:
                pred_data = read_pred_labels(f"./predicted/{gt_file}")
                pred_data = np.concatenate([pred_data])
                pred_data =torch.tensor(pred_data)
            except Exception as e:
                # print("------------>",str(e))
                pred_data = None
            gt_data = np.concatenate([gt_data])
            gt_data = torch.tensor(gt_data)
            confusion_matrix.process_batch(pred_data, gt_data)
        
                

        tp,fp = confusion_matrix.tp_fp()

        # print("-------------->",tp)
        # print("-------------->",fp)

        cm = confusion_matrix.matrix
        # print(cm)
        for c in range(len(classes)):
            tp = cm[c,c]
            total_preds = sum(cm[c,:])
            fp = total_preds - tp
            total_gt = sum(cm[:,c])
            # print("FP",cm[:,c])
            # print("FN",cm[c,:])

            recall = tp/(total_gt)
            precision = tp/(total_preds)

            f1_score = 2*((precision*recall)/(precision+recall))
            performance[f"TP_{conf/10}"].append(tp)
            performance[f"FP_{conf/10}"].append(fp)
            performance[f'Precision_{conf/10}'].append(100 * round(precision,4))
            performance[f'Recall_{conf/10}'].append(100 * round(recall,4))
            performance[f'F1_{conf/10}'].append(100 * round(f1_score,4))
            
    # print(performance)
    df = pd.DataFrame(performance)
    df.fillna(0, inplace=True)
    print(df)
    df.to_csv("./Performance-sheet.csv")


if __name__=="__main__":
    generate_sheet("/home/bm/SSD/sorted_tech/modelKPIs/ModelKPIs/test-set-cleaned-Evaluation/test/labels")