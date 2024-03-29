from csv import DictWriter
import numpy as np
import cv2
import logging as log

FONT = cv2.FONT_HERSHEY_SIMPLEX

def write_file(all_data):
    with open('./spreadsheet.csv','w') as outfile:
        writer = DictWriter(outfile,('time','current_count','num_tracked',
                'num_persons_in','previous_count',
                'total_count','stay_time',
                'mean_stay_time','infer_time',
                'process_time','result'))
        writer.writeheader()
        writer.writerows(all_data)

def count_persons(detections, image):
    num_detections = 0
    frame_with_bb = image
    if len(detections) > 0:
        frame_with_bb, num_detections = get_draw_boxes(detections, image)
    return num_detections, frame_with_bb

 
def get_draw_boxes(boxes, image):
    '''
        Function that returns the boundinng boxes detected for class "person" 
        with a confidence greater than 0, paint the bounding boxes on image
        and counts them
    '''
    # image_h, image_w, _ = image.shape
    num_detections = 0
    for box in boxes:
        # logger.debug("box: {}".format(box))
        if box['class_id'] == 0:
            if box['confidence'] > 0:
                cv2.rectangle(image,(box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0,255,0), 1)
                num_detections +=1

       
    return image, num_detections


def get_draw_boxes_on_image(boxes, image,prob_threshold=0.5,draw_label=False):
    '''
        Function that returns the boundinng boxes detected for class "person" 
        with a confidence greater than 0, paint the bounding boxes on image
        and counts them
    '''
    image_h, image_w, _ = image.shape
    num_detections = 0
    for box in boxes[0][0]:
        label   = box[1]
        conf    = box[2]
        x_min   = int(box[3]* image_w)
        y_min   = int(box[4]*image_h)
        x_max   = int(box[5]*image_w)
        y_max   = int(box[6]*image_h)

        if(label==1):
            if(conf>prob_threshold):
                dist=(y_max-y_min)/(y_min+y_max);
                color = (0,dist*255,255-dist*255)
                cv2.rectangle(image,(x_min,y_min), (x_max, y_max),
                 color, int(dist*2))
                num_detections +=1
        else:
            label_box_pos=None

    return image, num_detections