import os
from vortex_detection import helper_functions as hf, data_prep as prep, data_post as post
from vortex_detection import structure_test_rig as test, structure_train_rig as train, structure_evaluation_rig as eval
from data_visualise import Visualise
import datetime


creation_wanted = False
train_wanted = False
test_after_train_wanted = False
test_wanted = False
visualisation_wanted = False
evaluation_wanted = True
post_visualisation_wanted = True
turn_off_when_done = False


root = "D:/Jonas/Studium/TU/Semester/Bachelor/pythonProject/vortex_detection"
data_dir = "col_size_diff"
file_ending_additional_info = "imgsInformation.dat"

categories = {"worst": [0, 10], "best": [90, 100]}  # 0,10 means the biggest 10% of the data
col_criteria = "mean_ap"
colours = ["red", "green"]
normalise_except = []
cols_to_ignore = []
single_param = "nLocalNoiseAreas"
cols_non_param = ["too_few", "too_many"]
amount = None  # means all

test_wanted_single = False
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

arrowhead_wanted = [False]
add_train_bbox = False
add_test_bbox = False
plot_type = ["col"]  # "vec" (vector) or "col" (colour)

order_algorithm = "full_fac"  # can either be 'full_fac' or 'specific'
img_size = [[600, 600]]  # (width, height)
n_images = [261]
bbox_size_fac = [0.2]
n_information_per_axis = [[25, 25], [200, 200], [400, 400]]
n_vortices_per_img = [[2]]
noise_fac = [0, 3]
local_noise_fac = [0, 3]
n_local_noise_areas = [[10]]  # do not set this to 0. If no local noise is desired, set noise_fac to 0
preset_distance = [0.5, 1.5]  # either a float or None
IoU_file = root+data_dir
seed = [42]
n_img_train = [100]
validation_info = [[4, 15]]  # how many subsets of how many pictures
n_img_test = [100]   # can be at maximum n_images-n_img_train-(validation_info[0][0]*validation_info[0][1])
batch_size = 8
max_epochs = [30]
n_classes = 2
gamma = [[3,3], [1, 4]]
time = [[80,90]]
true_positive_criteria = "IoU"
true_positive_threshold = [0.75]
true_negative_better = "above"
early_stopper_criteria = "distance"  # or ap

working_dir = root+"/"+data_dir
file_protocol = working_dir+"/protocol.dat"
file_single_params = working_dir+"/single_params.dat"
if creation_wanted or train_wanted or test_after_train_wanted or test_wanted or visualisation_wanted or \
        evaluation_wanted:
    print(f"Starting creation of orders, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    orders = prep.Orders(["data", "train", "test"], working_dir)
    orders.set_params("data",
                      plotType=plot_type,
                      seed=seed,
                      size=img_size,
                      nInfoPerAxis=n_information_per_axis,
                      arrowheadWanted=arrowhead_wanted,
                      bboxSize=bbox_size_fac,
                      noiseFac=noise_fac,
                      localNoiseFac=local_noise_fac,
                      nLocalNoiseAreas=n_local_noise_areas,
                      gamma=gamma,
                      time=time,
                      nImgs=n_images,
                      vorticesImage=n_vortices_per_img,
                      vortex_distance=preset_distance)
    orders.set_params("train",
                      nTrainImgs=n_img_train,
                      valInfo=validation_info,
                      maxEpochs=max_epochs,
                      secondary_parameters=["epochs", "stoppedBy"])
    orders.set_params("test",
                      nTestImgs=n_img_test,
                      tpThreshold=true_positive_threshold)
    file_databases = orders.get_database_files()
    orders.set_ignore_params_for_order("data", {"plotType": ["col", "arrowheadWanted", "size"]})
    orders.create_orders(order_algorithm)
    orders.conduct_orders()
    print(f"Finished order creation, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

if creation_wanted:
    print(f"Starting creation of data, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    creator = prep.HamelOseenVortexCreator(working_dir=working_dir, database_file=file_databases["data"])
    creator.create_pngs()
    print(f"Finished creation of data, {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if prep_experimental_data:
    dir_experimental_data = "D:/Jonas/Studium/TU/Semester/Bachelor/pythonProject/exercise_object_detection/data_experimental"
    for file in exp_data_files:
        print(f"Preparing {file}.")
        prep.experimental_mat_data(file=dir_experimental_data+f"/{file}", plot_as=plot_as, normalise=normalise,
                                   mean=mean, percentage_information_x_axis=percentage_information_x_axis,
                                   percentage_information_y_axis=percentage_information_y_axis)


if add_train_bbox or add_test_bbox:
    train_or_test = {
        "train": True if add_train_bbox else False,
        "test": True if add_test_bbox else False
    }
    for width, height in zip(img_height, img_height):
        for n_vortices in n_vortices_per_img:
            cwd_path_variables = [{(width, height): None}, {"n_img": n_images}, {"vortices": n_vortices}]
            dir_imgs = hf.create_directory_from_value(data_dir, cwd_path_variables)
            for work_type in [[*train_or_test][i] for i in range(2) if train_or_test[[*train_or_test][i]]]:
                print(f"Adding bounding boxes to the {work_type} data in {dir_imgs}/{work_type}_bbox")
                prep.add_bboxes(working_dir=root+dir_imgs+"/", train_or_test=work_type,
                                colour="black" if plot_type == "col" else "green")

if train_wanted:
    train_rig = train.TrainRig(orders, working_dir)
    if len(early_stopper_criteria) != 0:
        train_rig.set_threshold(criteria=true_positive_criteria,
                                above_or_below=true_negative_better,
                                value=true_positive_threshold[0])
    train_rig.train_all(test_after_train=test_after_train_wanted, early_stopper_criteria=early_stopper_criteria)

if test_wanted:
    test_rig = test.TestRig(orders, working_dir)
    test_rig.test_all()

if visualisation_wanted:
    vis = Visualise(orders, working_dir)
    vis.set_threshold(criteria=true_positive_criteria,
                      above_or_below=true_negative_better,
                      value=true_positive_threshold[0])
    vis.all()

if evaluation_wanted:
    eval_rig = eval.EvaluationRig(orders, working_dir, file_protocol)
    eval_rig.set_threshold(criteria=true_positive_criteria,
                           above_or_below=true_negative_better,
                           value=true_positive_threshold[0])
    eval_rig.evaluate_predictions(["noiseFac", "localNoiseFac", "nLocalNoiseAreas", "valInfo", "maxEpochs"])

if post_visualisation_wanted:
    vis = post.PostVisualisation(dir_results=working_dir+"/results_vis", protocol_file=file_protocol,
                                 criteria_column=col_criteria, single_params_file=file_single_params,
                                 columns_to_ignore=cols_to_ignore)
    vis.parallel_coordinates(categories=categories, normalise_except=normalise_except, colours=colours)
    # vis.single_parameter(parameter=single_param, criteria_col=col_criteria, non_parameter_cols=cols_non_param,
    #                      amount=amount)

if turn_off_when_done:
    os.system("shutdown -s")