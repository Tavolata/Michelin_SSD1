import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import os

from models.keras_ssd import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

"""
    Model Parameters / can be adujsted
"""

img_height = 300  # Height of the input images
img_width = 480  # Width of the input images
img_channels = 3  # Number of color channels of the input images
intensity_mean = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 8  # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96]  # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size


if __name__ == "__main__":
    working_dir = "C:\\Users\\F293483\\data_mesures\\TU"

    # 1: Build the Keras model

    K.clear_session()  # Clear previous models from memory.

    model = build_model(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_global=aspect_ratios,
                        aspect_ratios_per_layer=None,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=intensity_mean,
                        divide_by_stddev=intensity_range)

    # 2: Instantiate an Adam optimizer and the SSD loss function and compile the model

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.
    """
    train_dataset = DataGenerator(load_images_into_memory=False,
                                  hdf5_dataset_path='D:\\Object Detection\\TU\\TU_train.h5')
    val_dataset = DataGenerator(load_images_into_memory=False,
                                hdf5_dataset_path='D:\\Object Detection\\TU\\TU_val.h5')

    """
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    # 2: Parse the image and label lists for the training and validation datasets.

    # Images
    images_dir = working_dir + '\\BaseTU\\TU'
    annotations_dir = working_dir + '\\BaseTU\\TU'
    image_set_filename_train = working_dir + '\\BaseTU\\TU\\image_names_train.txt'
    image_set_filename_val = working_dir + '\\BaseTU\\TU\\image_names_val.txt'

    # The XML parser needs to know what object class names to look for and in which order to map them to integers.
    # classes = ['background', 'temoin:0', 'temoin:25', 'temoin:50', 'temoin:75',
    #            'temoin:80', 'temoin:90', 'temoin:95', 'temoin:100']
    classes = ['background', 'temoin:25', 'temoin:50', 'temoin:75', 'temoin:100']

    train_dataset.parse_xml(images_dirs=[images_dir],
                            image_set_filenames=[image_set_filename_train],
                            annotations_dirs=[annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[images_dir],
                          image_set_filenames=[image_set_filename_val],
                          annotations_dirs=[annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)

    # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # option in the constructor, because in that case the images are in memory already anyway. If you don't
    # want to create HDF5 datasets, comment out the subsequent two function calls.

    train_dataset.create_hdf5_dataset(file_path=working_dir + '\\TU_train.h5',
                                      resize=(img_height, img_width),
                                      variable_image_size=True,
                                      verbose=True)

    val_dataset.create_hdf5_dataset(file_path=working_dir + '\\TU_val.h5',
                                    resize=(img_height, img_width),
                                    variable_image_size=True,
                                    verbose=True)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size = val_dataset.get_dataset_size()

    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

    # 3: Set the batch size.

    batch_size = 8

    # 4: Define the image processing chain.

    data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                                random_contrast=(0.5, 1.8, 0.5),
                                                                random_saturation=(0.5, 1.8, 0.5),
                                                                random_hue=(18, 0.5),
                                                                random_flip=0.5,
                                                                random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
                                                                random_scale=(0.5, 2.0, 0.5),
                                                                n_trials_max=3,
                                                                clip_boxes=True,
                                                                overlap_criterion='area',
                                                                bounds_box_filter=(0.3, 1.0),
                                                                bounds_validator=(0.5, 1.0),
                                                                n_boxes_min=1,
                                                                background=(0, 0, 0))

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                       model.get_layer('classes5').output_shape[1:3],
                       model.get_layer('classes6').output_shape[1:3],
                       model.get_layer('classes7').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_global=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.3,
                                        normalize_coords=normalize_coords)

    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             transformations=[data_augmentation_chain],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)
    
    # Define model callbacks.

    models_dir = working_dir + "\\Models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_checkpoint_filename = 'ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5'
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(models_dir, model_checkpoint_filename),
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)

    csv_logger_filename = 'ssd7_training_log.csv'
    csv_logger = CSVLogger(filename=os.path.join(models_dir, csv_logger_filename),
                           separator=',',
                           append=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0,
                                   patience=20,
                                   verbose=1)

    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.2,
                                             patience=8,
                                             verbose=1,
                                             epsilon=0.001,
                                             cooldown=0,
                                             min_lr=0.00001)

    callbacks = [model_checkpoint,
                 csv_logger,
                 early_stopping,
                 reduce_learning_rate]

    # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
    initial_epoch = 0
    final_epoch = 35
    steps_per_epoch = 1000

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=final_epoch,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=ceil(val_dataset_size / batch_size),
                                  initial_epoch=initial_epoch)

    # 1: Set the generator for the predictions.

    predict_generator = val_dataset.generate(batch_size=1,
                                             shuffle=True,
                                             transformations=[],
                                             label_encoder=None,
                                             returns={'processed_images',
                                                      'processed_labels',
                                                      'filenames'},
                                             keep_images_without_gt=False)

    # 2: Generate samples

    batch_images, batch_labels, batch_filenames = next(predict_generator)

    i = 0  # Which batch item to look at

    # print("Image:", batch_filenames[i])
    print()
    print("Ground truth boxes:\n")
    print(batch_labels[i])

    # 3: Make a prediction

    y_pred = model.predict(batch_images)

    # 4: Decode the raw prediction `y_pred`

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded[i])

    plt.figure(figsize=(20, 12))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right', prop={'size': 24})

    """
    # 5: Draw the predicted boxes onto the image

    plt.figure(figsize=(20, 12))
    plt.imshow(batch_images[i])

    current_axis = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()  # Set the colors for the bounding boxes
    classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist',
               'light']  # Just so we can print class names onto the image instead of IDs

    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in batch_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
        # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
    
    
    # Draw the predicted boxes in blue
    for box in y_pred_decoded[i]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
    """