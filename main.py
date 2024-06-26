import subprocess
import os
import argparse
import json
import yaml
import shutil
import re
from tqdm import tqdm
import shutil
from calculate_states import generate_sheet


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


def json2txt(path):
    os.makedirs(f"./predicted/",exist_ok=True)
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
        # yolo_content = f"{int(category_id)+17} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"
        with open(f"predicted/{image_id}.txt", 'a') as txt_file:
            txt_file.write(yolo_content)
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference with different confidence thresholds.")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--data", required=True, help="Path to YOLO data configuration file")
    parser.add_argument("--batch", default=12, help="Batch size as per you gpu memory")
    args = parser.parse_args()

    with open(args.data, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    ground_truth = yaml_data.get("val")
    classes = yaml_data.get("names")
    print("----------------------------------------------------------------------->")
    print(classes)
    print("----------------------------------------------------------------------->")
    # ground_truth = (yaml_data.get("val")).replace("images","labels")
   
    print("Inference Started for 0.1 confidence thresholds...")
    print("*************************************************************************")
    
    conf_threshold = 0.1
    res =run_yolo_inference(args.model, args.data, args.batch, conf_threshold)
    path = res[-2].replace("Results saved to ","")
    op_path = re.sub(r'\x1b\[\d+m', '', path)
    # print(str(path))

    print("Inference completed for different confidence thresholds...")
    print("*************************************************************************")

    if os.path.exists("./predicted"):
        shutil.rmtree("./predicted")
    
    
    json2txt(op_path)
    # print(paths)
    print("Json to txt Conversion Completed...")
    print("*************************************************************************")

    generate_sheet(ground_truth,classes)
    print("Performance Sheet Generated Succesfully...")
    print("*************************************************************************")
    
