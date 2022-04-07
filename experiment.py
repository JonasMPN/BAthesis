import os
from vortex_detection import helper_functions as hf, data_prep as prep, data_post as post
from vortex_detection import structure_test_rig as test, structure_train_rig as train, structure_evaluation_rig as eval
from data_visualise import Visualise
from helper_functions import Helper
import datetime
helper = Helper()

workload = {
    "creation": True,
    "train": True,
    "test_after_train": True,
    "test": False,
    "visualisation": True,
    "evaluation": False,
    "post_visualisation": False,
    "compare": False,
    "turn_off_when_done": False
}

root = "D:/Jonas/Studium/TU/Semester/Bachelor/pythonProject/vortex_detection"
data_dir = "pre_large/vec_info_diff"
file_ending_additional_info = "imgsInformation.dat"

categories = {"worst": [0, 10], "best": [90, 100]}  # 0,10 means the biggest 10% of the data
col_criteria = "mean_ap"
colours = ["red", "green"]
normalise_except = []
cols_to_ignore = []
single_param = "nLocalNoiseAreas"
cols_non_param = ["too_few", "too_many"]
amount = None  # means all

test_single = False
dir_model = "data_vortex_2111/(200, 200)/n_img_300/vortices_[1, 2, 3, 4]"
dir_to_test = "data_vortex_2111/(200, 200)/n_img_300/vortices_[1, 2, 3, 4]/test"
file_bbox_test = "data_vortex_2111/(200, 200)/n_img_300/vortices_[1, 2, 3, 4]"

prep_experimental_data = False
plot_as = "col"
mean = True
normalise = False
percentage_information_x_axis = 20
percentage_information_y_axis = 20
exp_data_files = ["UV_VN014.mat", "UV_VN026.mat", "UV_VN035.mat"]

add_train_bbox = False
add_test_bbox = False

order_algorithm = "full_fac"  # can either be 'full_fac' or 'specific'
parameters = {
    "data": {
        "arrowhead": ["yes"],
        "plotType": ["vec"],  # "vec" (vector) or "col" (colour)
        "imgSize": [[600,600]],  # (width, height)
        "nImg": [261],
        "bboxSizeFac": [0.2],
        # "nInfoPerAxis": [[20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70,70], [80,80], [90,90], [100, 100]],
        "nInfoPerAxis": [[40, 40]],
        "nVortices": [[1,2]],
        "noiseFac": [0],
        "localNoiseFac": [0],
        "nLocalNoiseAreas": [[10]],  # do not set this to 0. If no local noise is desired, set noise_fac to 0
        "presetDistance": [1], # either a float or None
        "randRotDir": "no", # if 'no': rotation direction alternates everytime a vortex is created
        "seed": [42],
        "gamma": [[3,3]],
        "time": [[80,90]],
        "ignore": [["plotType", ["col", "arrowheadWanted", "size"]]]
    },
    "train": {
        "nImgTrain": [100],
        "validationInfo": [[3, 20]],  # how many subsets of how many pictures
        "maxEpochs": [30],
        "earlyStopperCriteria": ["distance"],  # or ap,
        "predictionCriteria": ["score"],
        "predictionThreshold": [0.75], # which predictions to use for the calculation of the distances and labels,
        "predictionBetter": ["above"],
        "assignCriteria": ["distance"],
        "assignTpThreshold": [0.02],  # which predictions are regarded true positives (using assignCriteria)
        "assignBetter": ["below"],
        "secondaryParameters": ["epochs", "stoppedBy"]
    },
    "test": {
        "nImgTest": [100]   # can be at maximum n_images-n_img_train-(validation_info[0][0]*validation_info[0][1])
    }
}

handle_additional_information = {
    "all": "mean",
    "n_vortices": "sum"
}

compare_param = "nInfoPerAxis"
criteria = {"mean_ap": "bigger"}
result_cols = ["mean_distance_normalised", "not_detected_normalised", "epochs"]
mean_cols = ["mean_distance_normalised", "not_detected_normalised", "epochs"]
ignore_cols = ["stoppedBy", "arrowheadWanted", "size", "plotType"]
plot_as_compare = ["bar"]

working_dir = root+"/"+data_dir
helper.create_dir(working_dir)
file_protocol = working_dir+"/protocol.dat"
file_single_params = working_dir+"/single_params.dat"
files_parameters, current_experiment_idx = helper.save_parameters(working_dir=working_dir, parameters=parameters)
if any(list(workload.values())):
    print(f"Starting creation of orders, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    orders = prep.Orders(["data", "train", "test"], working_dir)
    orders.set_params_from_files(files=files_parameters, index_experiment=current_experiment_idx)
    file_databases = orders.get_database_files()
    orders.create_orders(order_algorithm)
    orders.conduct_orders()
    print(f"Finished order creation, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

if workload["creation"]:
    print(f"Starting creation of data, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    creator = prep.HamelOseenVortexCreator(working_dir=working_dir, database_file=file_databases["data"])
    creator.create_pngs()
    print(f"Finished creation of data, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if prep_experimental_data:
    dir_experimental_data = "D:/Jonas/Studium/TU/Semester/Bachelor/pythonProject/exercise_object_detection/data_experimental"
    real_life_data = prep.NonAnalyticalData()
    for file in exp_data_files:
        print(f"Preparing {file}.")
        real_life_data.plot_mat(file=dir_experimental_data+f"/{file}", plot_as=plot_as, normalise=normalise,
                                mean=mean, percentage_information_x_axis=percentage_information_x_axis,
                                percentage_information_y_axis=percentage_information_y_axis)

if workload["train"]:
    train = train.TrainRig(orders, working_dir)
    train.all(test_after_train=workload["test_after_train"])

if workload["test"]:
    test = test.TestRig(orders, working_dir)
    test.all()

if workload["visualisation"]:
    vis = Visualise(orders, working_dir)
    vis.all()

if workload["evaluation"]:
    evaluate = eval.EvaluationRig(orders, working_dir, file_protocol)
    evaluate.predictions(handle_additional_information, ["noiseFac", "localNoiseFac", "nLocalNoiseAreas"," valInfo",
                                                         "maxEpochs"])

if workload["post_visualisation"]:
    vis = post.PostVisualisation(dir_results=working_dir+"/results_vis", protocol_file=file_protocol,
                                 criteria_column=col_criteria, single_params_file=file_single_params,
                                 columns_to_ignore=cols_to_ignore)
    vis.parallel_coordinates(categories=categories, normalise_except=normalise_except, colours=colours)
    # vis.single_parameter(parameter=single_param, criteria_col=col_criteria, non_parameter_cols=cols_non_param,
    #                      amount=amount)

if workload["compare"]:
    evaluate = eval.EvaluationRig(orders, working_dir, file_protocol)
    for plot_as in plot_as_compare:
        evaluate.compare(save_directory=working_dir+"/results_vis", compare_col=compare_param, criteria=criteria,
                         result_cols=result_cols, mean_cols=mean_cols, ignore_cols=ignore_cols, plot_as=plot_as)

if workload["turn_off_when_done"]:
    os.system("shutdown -s")