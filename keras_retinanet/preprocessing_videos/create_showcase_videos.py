import cv2
import time
import numpy as np

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

THRESH = 0.6
TOP_PATH = '/content/drive/My Drive/person_detection/bus_video_showcases/'  
FPS = 20

def run_detection_video(video_path, vcapture, vwriter, num_frames, THRESH=THRESH):
    frame_index = 0
    success = True
    start = time.time()

    while success:
        if frame_index % 100 == 0:
            print("frame: ", frame_index)
        frame_index += 1
        # Read next image
        success, image = vcapture.read()

        if success:

            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            image = preprocess_image(image)
            image, scale = resize_image(image, min_side=800, max_side=1333)
            boxes, scores, labels = model.predict_on_batch(
                np.expand_dims(image, axis=0))
            boxes /= scale

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < THRESH:
                    break

                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format('Person', score)
                draw_caption(draw, b, caption)
            detected_frame = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
            vwriter.write(detected_frame)  # overwrites video slice

    vcapture.release()
    vwriter.release()
    end = time.time()
    print("Total Time: ", end - start)


def create_showcase_videos(video_showcases, video_path_parent=TOP_PATH, THRESH=0.55, fps=FPS):
    ''' Create a video showcase and save to given folder
    Arguments: 
        video_showcases: List of showcases to consider
        video_path_parent: Path where videos are stored
        THRESH: Threshold for the predictor to filter detections
        fps: Frames per second for velocity of video (Original video has 25fps)
    '''

    for showcase in video_showcases:
        video_path = video_path_parent + 'bus_showcase_' + showcase + '.avi'
        output_path = video_path_parent + 'bus_showcase_' + showcase + '_detected.mp4'
        vcapture = cv2.VideoCapture(video_path)

        # uses given video width and height
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('width: ', width, 'height: ', height)

        vwriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))        
        num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
        run_detection_video(video_path, vcapture, vwriter, num_frames)
