#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 16:50:55 2021

@author: yingma
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env,create_train_env_atari
from src.model_num_seq import ActorCritic, ActorCritic_seq
import torch.nn.functional as F
import time
from torch.distributions import Categorical
import numpy as np
from statistics import stdev
import math
#import fovea.foa_image as foai
#import fovea.foa_convolution as foac
#import fovea.foa_saliency as foas
import pickle
import copy
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
import cv2
from PIL import Image

from fista_para import *

import scipy.misc
from skimage.draw import line_aa
#from templete import truth_matching as detector_truth_matching


def str2bool(value):

    if value.lower()=='true':
        return True
    return False
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="training_result")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--num_sequence", type=int, default=5)
    parser.add_argument("--final_step", type=int, default=5000)
    parser.add_argument("--start_initial", type=str, default="random",help="inital method, can be random, noop or reset")
    parser.add_argument("--start_interval", type=int, default=20)
    parser.add_argument("--game", type=str, default="Supermario", help= "game select, can be Supermario, MsPacman-v0")
    parser.add_argument("--trials", type=int, default=1, help= "num test trials")
    parser.add_argument("--max_distance", type=int, default=1800, help= "win condition before trial stops")
    parser.add_argument("--save_detector", type=str, default="detector_models/dual_detector")
    parser.add_argument("--action_max", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    args = parser.parse_args()
    return args


def test(opt):
    dual_memory_decision_time = 0
    NN_decision_time = 0
    # read external memory content, these two pickle files are generated from dual_Detector_Creat.py
    with open('figure_patch.pickle', 'rb') as handle:
        figure_patch = pickle.load(handle)
    with open('Dual_Memory.pickle', 'rb') as handle:
        Dual_Memory = pickle.load(handle)
    
    fig, axsfigure = plt.subplots(4,3,figsize=(20, 20))  
    plt.rcParams['font.size'] = '6' 
    plt.setp(axsfigure, xlim=(-220,130), ylim=(-130,130))
    plt.gca().invert_yaxis()

     


    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 3
    video_filename = 'inference_phase.avi'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (2500, 2000),True)
    
    external_memory_fig = cv2.imread('image_template/external_memory_final.png')
    height_memory = external_memory_fig.shape[0]
    width_memory = external_memory_fig.shape[1]      
    
    NN_fig = cv2.imread('image_template/NN.png')
    height_NN = NN_fig.shape[0]
    width_NN = NN_fig.shape[1]    
    image_canvas1 = np.ones((240,256,3),dtype=np.uint8)*255
    image_canvas2 = np.ones((240,256,3),dtype=np.uint8)*255


#    plt.subplot(3,1)
    gate_max=True
    if opt.action_max == 1:
        action_max=True
    else:
        action_max =False
    torch.manual_seed(123)

    print("Action max: ", action_max)

    action_counts = []
    distance_counts = []

    

    # max amount of each object that can be in frame
    num_mario_range = 1
    num_goomba_range = 5
    num_pipe_range= 4
    num_koopa_range = 1


    goomba_template = np.load('templates/goomba_template.npy')
    tube_template = np.load('templates/pipe_template.npy')
    koopa_template = np.load("templates/template_turtle.npy")

    # mario templates assumed to be known
    mario_template = cv2.imread('templates/mario_templete.png', 0)
    mario_template = mario_template.astype(np.uint8)

    mario_template_1 = np.load('templates/mario_1.npy')
    mario_template_2 = np.load('templates/mario_2.npy')
    
    
    perfect_distance = opt.max_distance
    num_trials = opt.trials
    trial_count = 0

    print("trials: ", num_trials)
    print("perfect distance: ", perfect_distance)

    if opt.game == "Supermario":
        env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        env, num_states, num_actions = create_train_env_atari(opt.game,opt.saved_path,output_path=None)
#    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,
#                                                    "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))
    model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if torch.cuda.is_available() and opt.use_gpu:
        model.load_state_dict(torch.load(opt.saved_path+"/trained_model"))
        model.cuda()
    else:
        model.load_state_dict(torch.load(opt.saved_path+"/trained_model",
                                         map_location=lambda storage, loc: storage))        

    done=True
    while True:
        if done:
            
            curr_step_test = 0
            cum_r=0    
            with torch.no_grad():
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
                g_0_ini = torch.ones((1))
                
                g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
               
                env.reset()
                if opt.start_initial =='random':
                    for i in range(opt.start_interval):
                        if opt.game =='Supermario':
                            state, reward, _,done, info, unprocessed_state = env.step(env.action_space.sample())
                        else:
                            state, reward, _,done, info = env.step(env.action_space.sample(),0,video_save=False)
                        if done:
                            env.reset()       
                    state=torch.from_numpy(state)
                else:
                    state = torch.from_numpy(env.reset())
            if opt.use_gpu:
                state = state.cuda()        
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()
                g_0_ini = g_0_ini.cuda() 
                g_0 = g_0.cuda()         
                
            num_interaction=1
            score=0           
            
            
        curr_step_test += 1
#        with torch.no_grad():
#            h_0 = h_0.detach()
#            c_0 = c_0.detach()
#    
#        if gate_max:
#            logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_,update_state = model(state, h_0, c_0,g_0,g_0_ini,certain=True)
#        else:
#            logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_,update_state = model(state, h_0, c_0,g_0,g_0_ini)
#        g_0_ini = torch.zeros((1))
#        if opt.use_gpu:
#            g_0_ini = g_0_ini.cuda()            
#        policy = F.softmax(logits, dim=1)
#        if action_max:
#            action = torch.argmax(policy).item()
#        else:
#            m = Categorical(policy)
#            action = m.sample().item()  



        img_gray = rgb2gray(unprocessed_state[3])
        img_gray = img_gray.astype(np.uint8)  
        image_with_obj = unprocessed_state[3]
#        image_canvas = np.ones_like(image_with_obj)*255
        image_canvas = np.ones((240,256,3),dtype=np.uint8)*255
        width = 15
        # returns coords of all occurences of mario in frame
        mario_list =find_mario(img_gray, mario_template, mario_template_1, mario_template_2)
        # returns coords of all occurences of template
        # [x1, x2, x3] [y1, y2, y3]
        goomba_location = truth_matching(img_gray, goomba_template)
        tube_location = truth_matching(img_gray, tube_template)
        koopa_location = truth_matching(img_gray, koopa_template)

        # convert [x1, x2, x3] [y1, y2, y3] into [[x1, y1], [x2,y2], [x3,y3]]
        goomba_list = []
        tube_list = []
        koopa_list = []
        no_ankor = True
        if len(mario_list) == 1:
            image_canvas[mario_list[0][0]:mario_list[0][0]+width,mario_list[0][1]:mario_list[0][1]+width,:]=image_with_obj[mario_list[0][0]:mario_list[0][0]+width,mario_list[0][1]:mario_list[0][1]+width,:]
            for i in range(len(goomba_location[0])):
                if no_ankor:
                    ankor_coords = [goomba_location[0][i], goomba_location[1][i]]
                    coords = ankor_coords
                    goomba_list.append([0,0])
                    no_ankor = False
                else:
                    coords = [goomba_location[0][i]-ankor_coords[0], goomba_location[1][i]-ankor_coords[1]]
                    goomba_list.append(coords)
                real_coords = [goomba_location[0][i], goomba_location[1][i]]
                image_canvas[real_coords[0]:real_coords[0]+width,real_coords[1]:real_coords[1]+width,:]=image_with_obj[real_coords[0]:real_coords[0]+width,real_coords[1]:real_coords[1]+width,:]
    #            image_with_obj[coords[0]:coords[0]+width,coords[1]:coords[1]+width,0]=255
            
            for i in range(len(tube_location[0])):
                if i%3==0:
                    if no_ankor:
                        ankor_coords =[tube_location[0][i], tube_location[1][i]]
                        coords = ankor_coords
                        tube_list.append([0,0])
                        no_ankor = False
                    else:
                        coords = [tube_location[0][i]-ankor_coords[0], tube_location[1][i]-ankor_coords[1]]
                        tube_list.append(coords)
                    real_coords = [tube_location[0][i], tube_location[1][i]]
                    image_canvas[real_coords[0]:real_coords[0]+width,real_coords[1]:real_coords[1]+width,:]=image_with_obj[real_coords[0]:real_coords[0]+width,real_coords[1]:real_coords[1]+width,:]

                
                
            for i in range(len(koopa_location[0])):
                if no_ankor:
                    ankor_coords = [koopa_location[0][i], koopa_location[1][i]]
                    coords = ankor_coords
                    koopa_list.append([0,0])
                    no_ankor = False
                else:                   
                    coords = [koopa_location[0][i]-ankor_coords[0], koopa_location[1][i]-ankor_coords[1]]
                    koopa_list.append(coords)
                real_coords = [koopa_location[0][i], koopa_location[1][i]]
                image_canvas[real_coords[0]:real_coords[0]+width,real_coords[1]:real_coords[1]+width,:]=image_with_obj[real_coords[0]:real_coords[0]+width,real_coords[1]:real_coords[1]+width,:]
#            image_with_obj[coords[0]:coords[0]+width,coords[1]:coords[1]+width,:]=0
#            image_with_obj[coords[0]:coords[0]+width,coords[1]:coords[1]+width,2]=255
        # object count, used as key when appending to list
        current_num_each = [len(mario_list), len(goomba_list), len(tube_list), len(koopa_list)]  
#        print('num_each',current_num_each)
        if len(mario_list)!=1:
            print('number of mario detected', len(mario_list))
        # if object count is in in range,
        # we now have obj coords, and policy from above, add as tuple to list
        if (not no_ankor) and current_num_each[1]<=num_goomba_range and current_num_each[2]<=num_pipe_range:
#            item = [mario_list, goomba_list, tube_list, koopa_list,  policy]
            in_range = True
            mario_list[0][0] = mario_list[0][0] -ankor_coords[0]
            mario_list[0][1] = mario_list[0][1]-ankor_coords[1]
            coords_set = [mario_list, goomba_list, tube_list, koopa_list]
            image_canvas1=copy.deepcopy(image_canvas)
            # print(coords_set)
        else:
            coords_set = []
            in_range = False      

    ###########################################################################    
        key_list = []
        for item in coords_set:
            for item1 in item:
                key_list.append(tuple(item1))
        key_tuple= tuple(key_list)
        if tuple(current_num_each) in figure_patch :
            min_dis, min_dis_posi = distance(key_tuple,figure_patch[tuple(current_num_each)])
        if tuple(current_num_each) in figure_patch and min_dis<20:
            print('action from dual memory','min_dis, min_dis_posi',min_dis, min_dis_posi)
            key_tuple = tuple(figure_patch[tuple(current_num_each)][min_dis_posi])
            current_policy = Dual_Memory[tuple(current_num_each)][key_tuple][0]
            action = torch.argmax(current_policy).item()
            state, reward, raw_reward,done, info, unprocessed_state = env.step(action)
            dual_memory_decision_time += 1
            env.render()
            while Dual_Memory[tuple(current_num_each)][key_tuple][1]!=None:
                key_list = Dual_Memory[tuple(current_num_each)][key_tuple][1]
                key_tuple = tuple(key_list)
                current_policy = Dual_Memory[tuple(current_num_each)][key_tuple][0]
                action = torch.argmax(current_policy).item()
                state, reward, raw_reward,done, info, unprocessed_state = env.step(action)
                dual_memory_decision_time += 1
                env.render()
        else:
            print('action from Trajectory RL')
            
            with torch.no_grad():
                h_0 = h_0.detach()
                c_0 = c_0.detach()
        
            if gate_max:
                logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_,update_state = model(state, h_0, c_0,g_0,g_0_ini,certain=True)
            else:
                logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_,update_state = model(state, h_0, c_0,g_0,g_0_ini)
            g_0_ini = torch.zeros((1))
            if opt.use_gpu:
                g_0_ini = g_0_ini.cuda()            
            policy = F.softmax(logits, dim=1)
            if action_max:
                action = torch.argmax(policy).item()
            else:
                m = Categorical(policy)
                action = m.sample().item()              
            
            
            state, reward, raw_reward,done, info, unprocessed_state = env.step(action)
            env.render()          
            NN_decision_time += 1

        im=np.zeros((2000,2500,3),dtype=np.uint8)      
        height_raw_image= image_with_obj.shape[0]
        width_rawimage = image_with_obj.shape[1]
        im[600:600+height_raw_image,10:10+width_rawimage,:]=image_with_obj
        if in_range:
            red_rec = np.zeros((1600,2000,3),dtype=np.uint8)
            red_rec[:,:,0] = 255
            im[400:400+1600,410:410+2000,:] = red_rec
        else:
            red_rec = np.zeros((400,1000,3),dtype=np.uint8)
            red_rec[:,:,0] = 255
            im[10:10+400,410:410+1000,:] = red_rec        
        if in_range:
            im[1000:1000+height_raw_image,410:410+width_rawimage,:]=image_canvas1
            im[100:100+height_raw_image,410:410+width_rawimage,:]=image_canvas2
        else:
            image_canvas2 = np.ones((240,256,3),dtype=np.uint8)*255
            canvas_gray=cv2.cvtColor(image_with_obj, cv2.COLOR_RGB2GRAY)
            image_canvas2[:,:,0]=canvas_gray
            image_canvas2[:,:,1]=canvas_gray
            image_canvas2[:,:,2]=canvas_gray           
            im[100:100+height_raw_image,410:410+width_rawimage,:]=image_canvas2               
            im[1000:1000+height_raw_image,410:410+width_rawimage,:]=image_canvas1
            
        im[500:500+height_memory,900:900+width_memory,:]=external_memory_fig
        im[50:50+height_NN,900:900+width_NN,:]=NN_fig                            
        RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out.write(RGB_img) 
    
        print(info['x_pos'])
        score += raw_reward
        state = torch.from_numpy(state)
        if opt.use_gpu:
            state = state.cuda()
        cum_r = cum_r+reward     
        x=info['x_pos']
        if x>=perfect_distance:
            done = True
        time.sleep(.2)
        if g_0_cnt==0:
            # time.sleep(1)
            num_interaction+=1
        if done:
            x=info['x_pos']
            print(trial_count, x,num_interaction)
            distance_counts.append(x)
            action_counts.append(num_interaction)
            trial_count+=1
            print('decision with memory/NN',dual_memory_decision_time/NN_decision_time)
    
        if trial_count== num_trials:
            out.release()
            break



    num_perfect = sum(i > perfect_distance for i in distance_counts) 
    avg_distance = round(sum(distance_counts) / len(distance_counts), 1)
    avg_actions = round(sum(action_counts) / len(action_counts), 1)
    sample_dev = round(stdev(distance_counts), 1)
    interval_95 = sample_dev / math.sqrt(num_trials)
    low_95 =  round(avg_distance - 1.98 * interval_95, 1)
    high_95 = round(avg_distance + 1.98 * interval_95, 1)

    print()
    print("Trials: ", num_trials)
    print("Perfect Runs: ", num_perfect)
    print("Perfect pct: ", round((num_perfect * 100/ num_trials),1))
    print("Avg Distance: ", avg_distance)
    #print("Standard Dev: ", round(sample_dev, 1))
    print("95 percent range {} - {}".format(low_95, high_95))
    # print("Avg Action Cnt: ", avg_actions)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# returns coords of all occurences of mario in frame
def find_mario(img_gray, mario_template, mario_template_1, mario_template_2):
    # convert [x1, x2, x3] [y1, y2, y3] into [[x1, y1], [x2,y2], [x3,y3]]
    mario_location = truth_matching_mario(img_gray, mario_template)
    # lets convert the x y into coordinate list
    mario_list = []
    for i in range(len(mario_location[0])):
        coords = [mario_location[0][i], mario_location[1][i]]
        mario_list.append(coords)

        #    mario_list.append(coords)
    # if no mario was detected try a different template,
    # we assume the actor (mario) has aknown location
    if len(mario_list)!=1:
        mario_location = truth_matching_mario(img_gray, mario_template_1)

        mario_list = []
        for i in range(len(mario_location[0])):
            coords = [mario_location[0][i], mario_location[1][i]]
            mario_list.append(coords)
    
    if len(mario_list)!=1:
        mario_location = truth_matching_mario(img_gray, mario_template_2)

        mario_list = []
        for i in range(len(mario_location[0])):
            coords = [mario_location[0][i], mario_location[1][i]]
            mario_list.append(coords)
    
    return mario_list        
        
def distance(co1, co2):
    min_dis = 10000
    min_dis_posi = None
    for j in range (len(co2)):
        dis = 0
        for i in range(len(co1)):
            dis = dis + pow(abs(co1[i][0] - co2[j][i][0]), 2)+pow(abs(co1[i][1] - co2[j][i][1]), 2)
        if dis < min_dis:
            min_dis = dis
            min_dis_posi = j
    return  min_dis/len(co1),min_dis_posi

def place_image(im, loc=[3,4], ax=None, zoom=1, **kw):
    if ax==None: ax=plt.gca()
    imagebox = OffsetImage(im, zoom=zoom*0.72)
#    ab = AnchoredOffsetbox(loc=loc, child=imagebox, frameon=False, **kw)
    ab = AnnotationBbox(imagebox, loc)
    ax.add_artist(ab)
      
def draw_action(loc, point_clr,axsfigure,fig_index):
    circ = plt.Circle((loc[1], -loc[0]), 3, color=point_clr)
    axsfigure[fig_index//3][fig_index%3].add_patch(circ)

def draw_line(start_loc, end_loc,line_color,axsfigure,fig_index):
    axsfigure[fig_index//3][fig_index%3].plot([start_loc[1],end_loc[1]], [-start_loc[0],-end_loc[0]],color='k')

def draw_ankor(num_each, cor,fig_index,img_combo,fig, axsfigure):
    for i in range(3):
        num = num_each[i]
        im = img_combo[i]
        for j in range(num):
            im = im
            place_image(im, loc=[cor[i][j][1],-cor[i][j][0]], ax=axsfigure[fig_index//3][fig_index%3], pad=0, zoom=1)
if __name__ == "__main__":
    opt = get_args()
    test(opt)
