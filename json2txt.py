import json

def convert_to_yolo_format(bbox, image_width, image_height):
    x, y, width, height = bbox
    x_center = x + width / 2
    y_center = y + height / 2
    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height
    return round(x_center_normalized,6), round(y_center_normalized,6), round(width_normalized,6), round(height_normalized,6)

# Load the JSON data from the file
with open('predictions.json', 'r') as json_file:
    data = json.load(json_file)

# Process the data and create text files
for item in data:
    image_id = item['image_id']
    category_id = item['category_id']
    score = item['score']
    bbox = item['bbox']

    # Convert bbox to YOLO format
    image_width = 1920  # Provide the actual image width here
    image_height = 1200  # Provide the actual image height here
    yolo_bbox = convert_to_yolo_format(bbox, image_width, image_height)

    # Create the content string in YOLO format
    yolo_content = f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]} {score}\n"

    # Write YOLO content to the text file named after image_id
    with open(f"predicted/09/{image_id}.txt", 'a') as txt_file:
        txt_file.write(yolo_content)
