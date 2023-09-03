import glob
import json
import os
import shutil
import pandas as pd
import argparse
import matplotlib
import warnings
from statistics import mean

from util import *

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

MINOVERLAP = 0.3  # default value (defined in the PASCAL VOC2012 challenge)


class MAP:
    def __init__(self, args):
        self.gt_path = "./ground-truth"
        self.pred_path = "./predicted"
        self.tmp_files_path = "./tmp_files"
        self.results_files_path = "./results"

        self.predicted_files_list = glob.glob(f'{self.pred_path}/*.txt')
        self.ground_truth_files_list = glob.glob(f'{self.gt_path}/*.txt')

        # self.predicted_files_list.sort()
        self.draw_plot = False
        if args.ignore is None:
            args.ignore = []

        self.confidence_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        output_list = []
        self.args = args
        for conf in self.confidence_list:
            self.gt_counter_per_class = {}
            self.create_directories()
            self.gt_classes = self.generate_gt_json()
            self.n_classes = len(self.gt_classes)
            self.generate_pred_json(self.gt_classes, conf)
            output_list.append(self.calculate_map())
            output_list[-1]["confidence"] = conf

        final_data = {}
        # Loop through the data and populate the DataFrame
        for d in output_list:
            confidence = d['confidence']
            for key, values in d.items():
                if key not in final_data.keys():
                    final_data[key] = {}
                if key == 'confidence':
                    continue
                class_data = values
                final_data[key].update(({
                    f'Conf{confidence:.1f}_{metric}': class_data[metric]
                        for metric in ['tp', 'fp', 'prec', 'rec', 'f1', 'ap']
                }))

        df = pd.DataFrame(final_data).T
        df.to_excel('data.xlsx', index=True)

    def create_directories(self):
        if not os.path.exists(self.tmp_files_path):  # if it doesn't exist already
            os.makedirs(self.tmp_files_path)
        if os.path.exists(self.results_files_path):  # if it exist already
            # reset the results directory
            shutil.rmtree(self.results_files_path)
        os.makedirs(self.results_files_path)
        if self.draw_plot:
            os.makedirs(self.results_files_path + "/classes")

    def generate_gt_json(self):
        # get a list with the ground-truth files

        if len(self.ground_truth_files_list) == 0:
            error("Error: No ground-truth files found!")
        self.ground_truth_files_list.sort()
        # dictionary with counter per class
        for txt_file in self.ground_truth_files_list:
            # print(txt_file)
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # check if there is a correspondent predicted objects file
            if not os.path.exists(f"{self.pred_path}/0.1/" + file_id + ".txt"):
                error_msg = "Error. File not found: predicted/" + file_id + ".txt\n"
                error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                error(error_msg)
            lines_list = file_lines_to_list(txt_file)
            # create ground-truth dictionary
            bounding_boxes = []
            is_difficult = False
            for line in lines_list:
                try:
                    if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                    else:
                        class_name, left, top, right, bottom = line.split()
                        left, top, right, bottom = float(left) * 1920, float(top) * 1200, float(right) * 1920, float(
                            bottom) * 1200
                        left, top, right, bottom = str(int(left - right // 2)), str(int(top - bottom // 2)), str(
                            int(left + right // 2)), str(int(top + bottom // 2))
                        # left, top, right, bottom = str(float(left)*1920), str(float(top)*1200), str(float(right)*1920), str(float(bottom)*1200)
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                    error_msg += " Received: " + line
                    error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                    error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                    error(error_msg)
                # check if class is in the ignore list, if yes skip
                if class_name in args.ignore:
                    continue
                bbox = left + " " + top + " " + right + " " + bottom
                if is_difficult:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                    is_difficult = False
                else:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                    # count that object
                    if class_name in self.gt_counter_per_class:
                        self.gt_counter_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        self.gt_counter_per_class[class_name] = 1
            # dump bounding_boxes into a ".json" file
            with open(self.tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

            gt_classes = list(self.gt_counter_per_class.keys())
            # let's sort the classes alphabetically
            gt_classes = sorted(gt_classes, key=int)
        return gt_classes

    def generate_pred_json(self, gt_classes, conf):
        self.predicted_files_list = glob.glob(f'{self.pred_path}/{conf}/*.txt')
        # get a list with the predicted files
        for class_index, class_name in enumerate(gt_classes):
            bounding_boxes = []
            for txt_file in self.predicted_files_list:
                # print(txt_file)
                # the first time it checks if all the corresponding ground-truth files exist
                file_id = txt_file.split(".txt", 1)[0]
                file_id = os.path.basename(os.path.normpath(file_id))
                # if class_index == 0:
                    # if not os.path.exists('ground-truth/' + file_id + ".txt"):
                    #     error_msg = "Error. File not found: ground-truth/" + file_id + ".txt\n"
                    #     error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                    #     error(error_msg)
                lines = file_lines_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, left, top, right, bottom, confidence = line.split()
                        if float(confidence) < conf:
                            continue
                        left, top, right, bottom = float(left) * 1920, float(top) * 1200, float(right) * 1920, float(
                            bottom) * 1200
                        left, top, right, bottom = str(left - right // 2), str(top - bottom // 2), str(
                            left + right // 2), str(
                            top + bottom // 2)
                    except ValueError:
                        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                        error_msg += " Expected: <class_name> <left> <top> <right> <bottom> <confidence>\n"
                        error_msg += " Received: " + line
                        error(error_msg)
                    if tmp_class_name == class_name:
                        # print("match")
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
                        # print(bounding_boxes)
            # sort predictions by decreasing confidence
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(self.tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

    def calculate_map(self):
        sum_AP = 0.0
        ap_dictionary = {}

        output_data = {}

        count_true_positives = {}
        for class_index, class_name in enumerate(self.gt_classes):
            output_data[class_name] = {
                "pred": 0,
                "tp": 0,
                "fp": 0,
                "prec": 0,
                "rec": 0,
                "f1": 0,
                "ap": 0
            }
            count_true_positives[class_name] = 0
            predictions_file = self.tmp_files_path + "/" + class_name + "_predictions.json"
            predictions_data = json.load(open(predictions_file))
            nd = len(predictions_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]

                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = self.tmp_files_path + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [float(x) for x in prediction["bbox"].split()]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (
                                         bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # set minimum overlap
                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]

            apre = mean(prec) * 100
            arec = mean(rec) * 100

            # if not args.quiet:
            #     print("Precision: {}\nRecall: {}\nf1 score: {}\nAP: {}".format(round(apre, 2), round(arec, 2),
            #                                                                    round(2 * apre * arec / (apre + arec),
            #                                                                        2), round(ap, 2)).replace("\n", " "))
            output_data[class_name]["prec"] = round(apre, 2)
            output_data[class_name]["rec"] = round(arec, 2)
            output_data[class_name]["f1"] = round(2 * apre * arec / (apre + arec), 2)
            output_data[class_name]["ap"] = round(ap, 2)

            ap_dictionary[class_name] = ap

            """
             Draw plot
            """
            if self.draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                fig.savefig(self.results_files_path + "/classes/" + class_name + ".png")
                plt.cla()  # clear axes for next plot

            mAP = sum_AP / self.n_classes

        # remove the tmp_files directory
        shutil.rmtree(self.tmp_files_path)

        """
         Count total of Predictions
        """
        # iterate through all the files
        pred_counter_per_class = {}
        # all_classes_predicted_files = set([])
        for txt_file in self.predicted_files_list:
            # get lines to list
            lines_list = file_lines_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                # check if class is in the ignore list, if yes skip
                if class_name in args.ignore:
                    continue
                # count that object
                if class_name in pred_counter_per_class:
                    pred_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    pred_counter_per_class[class_name] = 1
        pred_classes = list(pred_counter_per_class.keys())

        """
         Plot the total number of occurences of each class in the ground-truth
        """
        if self.draw_plot:
            window_title = "Ground-Truth Info"
            plot_title = "Ground-Truth\n"
            plot_title += "(" + str(len(self.ground_truth_files_list)) + " files and " + str(
                self.n_classes) + " classes)"
            x_label = "Number of objects per class"
            output_path = self.results_files_path + "/Ground-Truth Info.png"
            to_show = False
            plot_color = 'forestgreen'
            draw_plot_func(
                self.gt_counter_per_class,
                self.n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                '',
            )

        """
         Finish counting true positives
        """
        for class_name in pred_classes:
            # if class exists in predictions but not in ground-truth then there are no true positives in that class
            if class_name not in self.gt_classes:
                count_true_positives[class_name] = 0
        print("==")
        print(self.gt_counter_per_class)
        print(pred_counter_per_class)
        print("==")
        # print("\n\n # Number of ground-truth objects per class\n")
        for class_name in sorted(self.gt_counter_per_class, key=int):
            if class_name not in pred_counter_per_class:
                n_pred = 0
            else:
                n_pred = pred_counter_per_class[class_name]
            text = class_name + ": " + " GT " + str(self.gt_counter_per_class[class_name]) + " pred " + str(n_pred)
            text += " ( tp: " + str(count_true_positives[class_name]) + " "
            text += ", fp: " + str(n_pred - count_true_positives[class_name]) + " )"
            # print(text)
            output_data[class_name]["gt"] = self.gt_counter_per_class[class_name]
            output_data[class_name]["pred"] = int(n_pred)
            output_data[class_name]["tp"] = int(count_true_positives[class_name])
            output_data[class_name]["fp"] = int(n_pred) - int(count_true_positives[class_name])

        """
         Plot the total number of occurences of each class in the "predicted" folder
        """
        if self.draw_plot:
            window_title = "Predicted Objects Info"
            # Plot title
            plot_title = "Predicted Objects\n"
            plot_title += "(" + str(len(self.predicted_files_list)) + " files and "
            count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
            plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
            # end Plot title
            x_label = "Number of objects per class"
            output_path = self.results_files_path + "/Predicted Objects Info.png"
            to_show = False
            plot_color = 'forestgreen'
            true_p_bar = count_true_positives
            draw_plot_func(
                pred_counter_per_class,
                len(pred_counter_per_class),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                true_p_bar
            )

        """
         Write number of predicted objects per class to results.txt
        """
        # for class_name in sorted(pred_classes):
        #     n_pred = pred_counter_per_class[class_name]
        #     text = class_name + ": " + str(n_pred)
        #     text += " (tp:" + str(count_true_positives[class_name]) + ""
        #     text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"

        """
         Draw mAP plot (Show AP's of all classes in decreasing order)
        """
        if self.draw_plot:
            window_title = "mAP"
            plot_title = "mAP = {0:.2f}%".format(mAP * 100)
            x_label = "Average Precision"
            output_path = self.results_files_path + "/mAP.png"
            to_show = True
            plot_color = 'royalblue'
            draw_plot_func(
                ap_dictionary,
                self.n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                ""
            )

        return output_data


def main(args):
    MAP(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # argparse receiving list of classes to be ignored
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    args = parser.parse_args()
    main(args)
