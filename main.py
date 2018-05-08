import pickle
from scipy.ndimage.measurements import label
import numpy as np
from moviepy.editor import VideoFileClip
from sliding_window import find_cars, apply_threshold, draw_labeled_bboxes, add_heat
from itertools import chain


def image_pipelane(img):
    # List of the boxes generated for each frame
    boxes = []
    # Find cars that are far away
    ystart = 400
    ystop = 464
    scale = 1
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    ystart = 416
    ystop = 480
    scale = 1
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    # Find cars that are medium away
    ystart = 400
    ystop = 496
    scale = 1.5
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    ystart = 432
    ystop = 528
    scale = 1.5
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    # Find cars that are close
    ystart = 400
    ystop = 528
    scale = 2
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    ystart = 464
    ystop = 592
    scale = 2
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    # Find cars very close
    ystart = 400
    ystop = 596
    scale = 3.5
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    ystart = 464
    ystop = 660
    scale = 3.5
    boxes += (find_cars(img, ystart, ystop, scale, svc, scaler, spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis_boxes=False))

    # Save boxes over several frames
    global boxes_memory
    if len(boxes_memory) < 15:
        boxes_memory.append(boxes)
    else:
        boxes_memory.pop(0)
        boxes_memory.append(boxes)
    boxes = list(chain.from_iterable(boxes_memory))
    # Make heatmap of the boxes
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, boxes)
    heatmap = apply_threshold(heatmap, 48)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img


with open('classifier.pkl', 'rb') as input:
    svc = pickle.load(input)

with open('scaler.pkl', 'rb') as input:
    scaler = pickle.load(input)

boxes_memory = []

clips = 'project_video.mp4'

clip1 = VideoFileClip(clips)
write_output = 'output_video/' + clips
write_clip = clip1.fl_image(image_pipelane)  # NOTE: this function expects color images!!
write_clip.write_videofile(write_output, audio=False)
