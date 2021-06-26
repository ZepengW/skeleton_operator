import numpy as np
import copy
import math
import json
import os


def re_rank_skeleton_data(skeleton_data,max_num=5):
    '''
    this method is used to re-rank the person position in one frame to make sure the identity consistence
    :param skeleton_data: skeleton data extract from json file
    :param max_num: max person number
    :return: skeleton data re-ranked
    '''
    skeleton_data_person_pre = []
    skeleton_data_new = []
    f_id = 0
    for f in skeleton_data:
        f_d = {}
        f_id += 1
        pids_map = get_person_ids(f,skeleton_data_person_pre)
        j = 0
        for i,p in enumerate(f):
            if i in pids_map:
                pid = pids_map[i]
                skeleton_data_person_pre[pid] = p
            else:
                pid = len(skeleton_data_person_pre)+j
                j += 1
                skeleton_data_person_pre.append(p)
            if pid >= max_num or pid < 0:
                continue
            f_d[pid] = copy.deepcopy(p)
        pids = list(f_d.keys())
        pids.sort()

        f_n = []
        for pid in pids:
            while(pid > len(f_n)): # append zero
                f_n.append(list(np.zeros([len(f_d[pid]),3])))
            f_n.append(f_d[pid])
        skeleton_data_new.append(f_n)
    return skeleton_data_new


def get_person_id(skeleton_data_person,skeleton_data_person_pre,thd=0.4,dis_thre=1500):
    index = -1
    min_value = -1
    for idx,data in enumerate(skeleton_data_person_pre):
        value = 0.0
        for j_id,j in enumerate(skeleton_data_person):
            if j[2] >= thd and data[j_id][2] >= thd:
                value += math.dist(j[0:2],data[j_id][0:2])
        if value < min_value or min_value == -1:
            index = idx
            min_value = value
    if index == -1 or min_value > dis_thre:
        index = len(skeleton_data_person_pre)
    return index

def get_person_ids(skeleton_data_persons, skeleton_data_person_pre, thre = 0.4, dis_thre = 30):
    pid_map = {}
    dis_map = np.zeros([len(skeleton_data_persons),len(skeleton_data_person_pre)])
    dis_map.fill(dis_thre + 1)
    for idx_n,data_n in enumerate(skeleton_data_persons):
        for idx_p,data_p in enumerate(skeleton_data_person_pre):
            # calculate distance
            value = 0.0
            num_j_avail = 0
            believe = 0.0
            for j_id in range(len(data_n)):
                believe += data_n[j_id][2]
                if data_n[j_id][2] >= thre:
                    num_j_avail += 1
                    value += math.dist(data_n[j_id][0:2],data_p[j_id][0:2])
            if num_j_avail < 5:
                value = -1
            else:
                value = value / num_j_avail
            dis_map[idx_n][idx_p] = value
    while not (dis_map > dis_thre).all():
        value_min = np.min(dis_map)
        min_pos = np.where(dis_map == value_min)
        idx_row = min_pos[0][0]
        idx_col = min_pos[1][0]
        if value_min < 0:
            pid_map[idx_row] = -1
            dis_map[idx_row, :] = dis_thre + 1
        else:
            pid_map[idx_row] = idx_col
            dis_map[idx_row,:] = dis_thre+1
            dis_map[:,idx_col] = dis_thre+1
    return pid_map



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

if __name__ == '__main__':
    skeleton_path = './test/destroy_198.json'
    skeleton_data = convert_json_joints(skeleton_path)
    re_rank_skeleton_data(skeleton_data)