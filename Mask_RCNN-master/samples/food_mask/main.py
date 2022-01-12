import random
import os
import sys


ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library


import food
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
import matplotlib.pyplot as plt


MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = food.FoodConfig()
FOOD_DIR = os.path.join(ROOT_DIR, "datasets/food")

dataset = food.FoodDataset()
dataset.load_food(FOOD_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


def train():
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    init_with = "coco"
    if init_with == "coco":
        # skipping the layers different due to the class numbers
        model.load_weights(os.path.join(ROOT_DIR, "logs/mask_rcnn_coco.h5"), by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(os.path.join(ROOT_DIR, "logs/mask_rcnn_food_0044.h5"), by_name=True)

    dataset_train = food.FoodDataset()
    dataset_train.load_food(FOOD_DIR, "train")
    dataset_train.prepare()

    dataset_val = food.FoodDataset()
    dataset_val.load_food(FOOD_DIR, "val")
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                layers='all')


# train()


def test():
    class InferenceConfig(food.FoodConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # create the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    model_path = os.path.join(ROOT_DIR, "logs/mask_rcnn_food_0044.h5")
    print(model_path)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    dataset_val = food.FoodDataset()
    dataset_val.load_food(FOOD_DIR, "val")
    dataset_val.prepare()

    image_id = random.choice(dataset_val
                             .image_ids)
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset_val.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset_val.class_names)
    original_image = image

    plt.axis("off")
    plt.imshow(image)

    results = model.detect([original_image], verbose=0)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'])

    masked_plate_pixels = 1130972
    real_plate_size = 12
    real_plate_area = 113.04
    pixels_per_inch_sq = masked_plate_pixels / real_plate_area
    calories = []
    items = []
    for index in range(r['masks'].shape[-1]):
        #   print(i)
        masked_food_pixels = r['masks'][:, :, index].sum()
        class_name = dataset_val.class_names[r['class_ids'][index]]
        real_food_area = masked_food_pixels / pixels_per_inch_sq
        calorie = food.get_calorie(class_name, real_food_area)
        calories.append(calorie)
        items.append(class_name)
        print("{1} with {0} calories".format(int(calorie), class_name))

    # Create a pieplot
    plt.pie(calories, labels=items)
    # plt.show()

    # add a circle at the center
    my_circle = plt.Circle((0, 0), 0.6, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)

    plt.show()


test()
