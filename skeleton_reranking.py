import numpy as np
import copy
import math


def re_rank_skeleton_data(skeleton_data,max_num=5):
    '''
    this method is used to re-rank the person position in one frame to make sure the identity consistence
    :param skeleton_data: skeleton data extract from json file
    :param max_num: max person number
    :return: skeleton data re-ranked
    '''
    skeleton_data_person_pre = []
    skeleton_data_new = []
    for f in skeleton_data:
        f_d = {}
        for p in f:
            pid = get_person_id(p,skeleton_data_person_pre)
            if pid >= max_num:
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


def get_person_id(skeleton_data_person,skeleton_data_person_pre,thd=0.4):
    index = 0
    min_value = -1
    for idx,data in enumerate(skeleton_data_person_pre):
        value = 0.0
        for j_id,j in enumerate(skeleton_data_person):
            if j[2] >= thd and data[j_id][2] >= thd:
                value += math.dist(j[0:2],data[j_id][0:2])
        if value < min_value or min_value == -1:
            index = idx
            min_value = value
    return index