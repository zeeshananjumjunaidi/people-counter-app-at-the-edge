from csv import DictWriter
import numpy as np
import cv2

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

def get_draw_boxes_on_image(boxes, image,prob_threshold=0.5):
    '''
        Function that returns the boundinng boxes detected for class "person" 
        with a confidence greater than 0, paint the bounding boxes on image
        and counts them
    '''
    image_h, image_w, _ = image.shape
    num_detections = 0
    for box in boxes:
        # logger.debug("box: {}".format(box))
        # print("box: {}".format(box))
        image_id = box[0][0][0]
        label = box[0][0][1]
        conf = box[0][0][2]
        x_min =box[0][0][3]* image_w
        y_min =box[0][0][4]*image_h
        x_max =box[0][0][5]*image_w
        y_max =box[0][0][6]*image_h
        x_min=int(x_min)
        y_min=int(y_min)
        x_max=int(x_max)
        y_max=int(y_max)
        print('image id:{}, label:{}, confidence:{}, Xmin:{}, Ymin:{}, Xmax:{}, Ymax:{}'.
        format(image_id,label,conf,x_min,y_min,x_max,y_max))
        if(label==1):
            if(conf>prob_threshold):
                print('person found')                
                cv2.rectangle(image,(x_min,y_min), (x_max, y_max), (0,255,0), 4)
                num_detections +=1
        # if box['class_id'] == 0:
        #     if box['confidence'] > 0:
        #         cv2.rectangle(image,(box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0,255,0), 1)
        #         num_detections +=1

       
    return image, num_detections