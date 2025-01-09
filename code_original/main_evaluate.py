from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
#from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = 4
model_mode = 'training'

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'C:\\Users\\F293483\\data_mesures\\TU\\Models\\ssd300_XS_pascal_07+12_epoch-28_loss-5.2015_val_loss-4.8362.h5'
# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

dataset = DataGenerator()

# TODO: Set the paths to your dataset here.

# Images
working_dir = "C:\\Users\\F293483\\data_mesures\\TU\\"
'''
images_dir = working_dir + 'BaseTU/small_TU'

# Ground truth
annotations_dir = working_dir + 'BaseTU/small_TU'
image_set_filename_train = working_dir + 'BaseTU/image_names_train.txt'
image_set_filename_val = working_dir + 'BaseTU/image_names_val.txt'
'''
# The XML parser needs to know what object class names to look for and in which order to map them to integers.
#todo: give the classes you chose for your training
classes = ['background', 'temoin:25', 'temoin:50', 'temoin:75', 'temoin:100']

train_dataset_path = working_dir + 'TU300_Xsmall_train.h5'
val_dataset_path = working_dir + 'TU300_Xsmall_val.h5'


# reload existing dataset
dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=val_dataset_path)

'''dataset.parse_xml(images_dirs=[images_dir],
                  image_set_filenames=[image_set_filename_val],
                  annotations_dirs=[annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)'''

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=False,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results


for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes: break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0,1,11))
        cells[i, j].set_yticks(np.linspace(0,1,11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)

'''
evaluator.write_predictions_to_txt(classes=classes,
                                 out_file_prefix=working_dir + 'BaseTU/eval300Xsmall',
                                 verbose=True)
'''
plt.show()
#step by step
'''evaluator.get_num_gt_per_class(ignore_neutral_boxes=True,
                               verbose=False,
                               ret=False)

evaluator.match_predictions(ignore_neutral_boxes=True,
                            matching_iou_threshold=0.5,
                            border_pixels='include',
                            sorting_algorithm='quicksort',
                            verbose=True,
                            ret=False)

precisions, recalls = evaluator.compute_precision_recall(verbose=True, ret=True)

average_precisions = evaluator.compute_average_precisions(mode='integrate',
                                                          num_recall_points=11,
                                                          verbose=True,
                                                          ret=True)

mean_average_precision = evaluator.compute_mean_average_precision(ret=True)'''