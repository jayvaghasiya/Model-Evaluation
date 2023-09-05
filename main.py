import subprocess
import os
import argparse
import json
import yaml
import shutil
import re
from tqdm import tqdm
import glob
import json
import os
import shutil
import pandas as pd
import argparse
import matplotlib
import warnings
from statistics import mean
import sys
import cv2
import operator
import matplotlib.pyplot as plt

from calculate_states import * 

def run_yolo_inference(model_path, data_yaml_path, batch, conf_threshold):
    yolo_command = f'yolo detect val model={model_path} data={data_yaml_path}'
    yolo_command += f' conf={conf_threshold}'
    yolo_command += f' batch={batch} save_json=True'
    result = subprocess.run(yolo_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines = result.stdout.splitlines()
    if result.returncode != 0:
        print("ERROR : ")
        print(result)
    return output_lines
        

def convert_to_yolo_format(bbox, image_width, image_height):
    x, y, width, height = bbox
    x_center = x + width / 2
    y_center = y + height / 2
    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height
    return round(x_center_normalized,6), round(y_center_normalized,6), round(width_normalized,6), round(height_normalized,6)


def json2txt(path,conf):
    os.makedirs(f"./predicted/{conf}/",exist_ok=True)
    with open(f'{path}/predictions.json', 'r') as json_file:
        data = json.load(json_file)

    for item in data:
        image_id = item['image_id']
        category_id = item['category_id']
        score = item['score']
        bbox = item['bbox']
        image_width = 1920  
        image_height = 1200  
        yolo_bbox = convert_to_yolo_format(bbox, image_width, image_height)
        yolo_content = f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]} {score}\n"
        with open(f"predicted/{conf}/{image_id}.txt", 'a') as txt_file:
            txt_file.write(yolo_content)
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference with different confidence thresholds.")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--data", required=True, help="Path to YOLO data configuration file")
    parser.add_argument("--batch", default=-1, help="Batch size as per you gpu memory")
    args = parser.parse_args()

    try:
        with open(args.data, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        ground_truth = (yaml_data.get("val")).replace("images","labels")
       # print(ground_truth)

        conf_thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        paths = []
        print("Inference Started for different confidence thresholds...")
        print("*************************************************************************")
        for conf_threshold in tqdm(conf_thresholds):
            res =run_yolo_inference(args.model, args.data, args.batch, conf_threshold)
            path = res[-1].replace("Results saved to ","")
            path = re.sub(r'\x1b\[\d+m', '', path)
            paths.append(path)
            # print(str(path))
    
        print("Inference completed for different confidence thresholds...")
        print("*************************************************************************")

        if os.path.exists("./predicted"):
            shutil.rmtree("./predicted")
        
        
        for path,conf in zip(paths,conf_thresholds):
            json2txt(path,conf)
        # print(paths)
        print("Json to txt Conversion Completed...")
        print("*************************************************************************")

        calculate_performance(ground_truth)
        print("Performance Sheet Generated Succesfully...")
        print("*************************************************************************")
    
    except Exception as e:
        print("ERROR: ", str(e))
