'''
    Function:
        draw skeleton on image/video
        support 18 skeleton for now
    Example:
        can see main function
'''

import os
import cv2
import numpy as np
import math
import json
import time
from skeleton_reranking import re_rank_skeleton_data


p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
           # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
           (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
l_pair = [
    (0, 14), (0, 15), (14, 16), (15, 17),(0, 1 ),  # Head
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6),(6,7),
    (1, 8), (1, 11),  # Body
    (8, 9), (9, 10), (11, 12), (12, 13)
]

'''
    Method: draw skeleton lines and points on video
    video_path: the path of video
    skeleton_data: the skeleton data. Type: List_Frames[List_Person[List_Joints[x,y,score]]]
'''
def draw_skeleton_video(video_path, skeleton_data):
    if not os.path.exists(video_path):
        print(video_path+' does not exist')
    cap = cv2.VideoCapture(video_path)
    out_name = os.path.basename(video_path).split('.')[0] + '_skeleton.avi'
    dir_path = os.path.dirname(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames = []
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('frame list len:' + str(len(skeleton_data)))
    print('video len:' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    max_frames = min(len(skeleton_data),cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("drawing joints and skeletons")
    # draw joints and lines
    i = 0
    while (cap.isOpened() and i < max_frames):
        ret, frame = cap.read()
        if not ret:
            print('bad frame')
            break
        for p in skeleton_data[i]:
            frame = draw_joints_per_frame(frame, p)
        i = i + 1
        frames.append(frame)
        print('\rprocessing : %d/%d \t' %(i,total_frame),end="")
    print(' ')
    cap.release()
    print("draw finish")

    print("saving")
    out = cv2.VideoWriter(os.path.join(dir_path, out_name), fourcc, float(fps), (int(width), int(height)))
    for f in frames:
        out.write(f)
    out.release()
    print("save finish")


'''
    Method: draw skeleton lines and points on black backgrond
    output_dir: output video dir
    skeleton_data: the skeleton data. Type: List_Frames[List_Person[List_Joints[x,y,score]]]
'''
def draw_skeleton_black_backgrond(output_path,skeleton_data,resolution=(720,360)):
    print('generating: ' + output_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames = []
    for skeleton_frame in skeleton_data:
        pic_f = np.zeros([resolution[1],resolution[0],3],np.uint8)
        for skeleton_person in skeleton_frame:
            pic_f = draw_joints_per_frame(pic_f,skeleton_person)
        frames.append(pic_f)
    video_out = cv2.VideoWriter(output_path,fourcc,20.0,resolution)
    for f in frames:
        video_out.write(f)
    video_out.release()
    print('generate finish')

'''
    convert json_file to the skeleton_list we need for draw
'''
def convert_json_joints(path):
    '''
    convert json file to joints_list
    :param path:
    :return: [frames[persons[joints[x,y,score]]]]
    '''
    if not os.path.exists(path):
        print("can't find file : " + path)
    result = []
    framd_id = []
    name = []
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for name_f, data_f in json_data.items():
        data_persons = data_f['people']
        f_l = []
        for data_p in data_persons:
            j_l = data_p['pose_keypoints_2d']
            if len(j_l) == 18 * 3:
                p_l = [[j_l[3 * i], j_l[3 * i + 1], j_l[3 * i + 2]]
                       for i in range(0, 18)]
                f_l.append(p_l)
        result.append(f_l)
        framd_id.append(int(name_f.split('.')[0][-4:]))
        name.append(name_f)
    # sort
    data = [(id, res) for id, res in zip(framd_id, result)]
    data.sort()
    result = [res for id, res in data]

    data = [(id, n) for id, n in zip(framd_id, name)]
    data.sort()
    name = [n for id, n in data]
    return result

'''
    vis_thres: only the joints score >= vis_thres will display
'''
def draw_joints_per_frame(img, joints):
    vis_thres = 0.4
    part_line = {}
    for n, j in enumerate(joints):
        if j[2] <= vis_thres:
            continue
        cor_x, cor_y = int(j[0]), int(j[1])
        part_line[n] = (int(cor_x), int(cor_y))
        if n < len(p_color):
            cv2.circle(img, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
        else:
            cv2.circle(img, (int(cor_x), int(cor_y)), 1, (255, 255, 255), 2)
    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            X = (start_xy[0], end_xy[0])
            Y = (start_xy[1], end_xy[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = (joints[start_p][2] + joints[end_p][2]) + 1
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360,
                                       1)
            if i < len(line_color):
                cv2.fillConvexPoly(img, polygon, line_color[i])
            else:
                cv2.line(img, start_xy, end_xy, (255, 255, 255), 1)
    return img

'''
if __name__ == '__main__':
    json_path = './yejian/waveknife_84.json'
    video_path = './yejian/waveknife_84.mp4'
    skeleton_data = convert_json_joints(json_path)
    draw_skeleton_video(video_path, skeleton_data)
'''

def draw_skeleton_batch(intput_dir,output_dir,resolution=(720,360),sub=''):
    intput_files = os.listdir(intput_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i,intput_file in enumerate(intput_files):
        print('processing :[{0}/{1}]'.format(i+1,len(intput_files)))
        time_b = time.time()
        if not '.json' in intput_file:
            continue
        output_path = os.path.join(output_dir,intput_file.split('.json')[0]+sub+'.avi')
        skeleton_data = convert_json_joints(os.path.join(intput_dir,intput_file))
        #skeleton_data = re_rank_skeleton_data(skeleton_data)
        draw_skeleton_black_backgrond(output_path,skeleton_data,resolution)
        time_e = time.time()
        print('cost time: {0}s'.format(time_e-time_b))



if __name__ == '__main__':
    input_dir = './test'
    output_dir = './output_test'
    draw_skeleton_batch(input_dir,output_dir,resolution=(2560,1440),sub='_no')
