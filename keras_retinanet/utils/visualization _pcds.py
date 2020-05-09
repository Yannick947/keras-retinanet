
import cv2
import numpy as np
from matplotlib.pyplot import plt


from .colors import label_color
from .visualization import draw_box, draw_caption, draw_boxes, draw_detections, draw_annotations
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.bin import evaluate

THRES_SCORE = 0.9

def predict(image, model):
  image = preprocess_image(image.copy())
  image, scale = resize_image(image)

  boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
  )

  boxes /= scale

  return boxes, scores, labels


def draw_detections(image, boxes, scores, labels, labels_to_names):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(image, b, caption)

def show_detected_objects(img_path, model, labels_to_names):
    # img_path = '/content/drive/My Drive/person_detection/WiderPerson/bus_showcase.jpg'
    image = read_image_bgr(img_path)

    boxes, scores, labels = predict(image, model)

    for i, score in enumerate(scores[0]): 
        if score > THRES_SCORE:
            print('row ', img_path, 'score ', scores[0, i])
  
    if all(i < THRES_SCORE for i in scores[0]):
        print('no detections')
        return

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    draw_detections(draw, boxes, scores, labels, labels_to_names)

    plt.axis('off')
    plt.imshow(draw)
    plt.show()
