# Model-Evaluation-Script

* This script will generate and Excel sheet of your model's performance across diffrent confidence thresholds.


# Steps:

  1. Installing Dependencies:

      Run this command : ```bash
     pip install -r requirements.txt```

  3. Running Evaluation script:

     Run this command : ```bash
     python main.py --model /path/to/model --data dataset.yaml --batch 25```

     * In this command you need to pass the path to your custom yolo model using "--model" argument.
     * And you need to pass the yaml file of your dataset using  "--data" argument.
     * for refrence you can view dataset.yaml file and you need to change or replace that file as per your requirement.
     * You can also change the batch size as per your gpu memory you can pass batch size using "--batch" argument (default is 10).

  5. Execution of this command will take some time and after that you will find "Performance-sheet.csv" file which contains Evaluation record of your model's performance.
     
     
