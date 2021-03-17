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
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas
import pickle

import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
import cv2
from PIL import Image

from fista_para import *
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
    parser.add_argument("--trials", type=int, default=3, help= "num test trials")
    parser.add_argument("--max_distance", type=int, default=1800, help= "win condition before trial stops")
    parser.add_argument("--save_detector", type=str, default="detector_models/dual_detector")
    parser.add_argument("--action_max", type=int, default=1)


    parser.add_argument("--use_gpu", type=str2bool, default=False)
    args = parser.parse_args()
    return args


def test(opt):
    im_goomba=plt.imread('image_template/goomba.png')
    im_tube=plt.imread('image_template/tube.png')
    im_koopa=plt.imread('image_template/koopa.png')
    img_combo = [im_goomba,im_tube,im_koopa]
    fig, axsfigure = plt.subplots(4,3,figsize=(20, 20))  
    plt.rcParams['font.size'] = '1' 
    plt.setp(axsfigure, xlim=(-220,130), ylim=(-130,130))
    plt.gca().invert_yaxis()
    color = ['red','blue','yellow','k','g','pink','c','m','y','lime','tan','brown']
    line_color_loop=0
    figure_patch = {}

    

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 3
    video_filename = 'entire_image.avi'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (2500, 1600),True)
    
    
    
    
    
    Dual_Memory = {}
    figure_index = {}
    fig_index = 0
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

    super_list = []
    # simplifies look up later on
    index_tracker = []
    tracker_counter = 0
    seq_counter = 1

    current_policy_list = []
    current_coord_list = []
    first_input = True
    trash_next = True

#    max_marios = 1
#    max_goombas = 5
#    max_pipes = 4
#
#    for i in range(max_marios):
#        for j in range(max_goombas):
#            for k in range(max_pipes):
#                super_list.append([[i+1, j, k]])
#                index_tracker.append([i+1, j, k])
#    
#    print(super_list)
    # max amount of each object that can be in frame
    num_mario_range = 1
    num_goomba_range = 5
    num_pipe_range= 4
    num_koopa_range = 1

    # will house [coord, policy] tuples, divided by object amount
    # 1 0 1 1: [[coord, policy] [coord, policy]]
    # 1 0 1 2: [[coord, policy] [coord, policy]]
    super_list = []
    # simplifies lookup later on
    index_tracker = []

    for i in range(num_mario_range +1):
        for j in range(num_goomba_range + 1):
            for k in range(num_pipe_range + 1):
                for l in range(num_koopa_range + 1):
                    super_list.append([[i, j, k, l]])
                    index_tracker.append([i, j, k, l])
    # running_averages =  np.zeros(results_to_keep)
#    goomba_detect = np.load('template_item1.npy')
#    pipe_detect = np.load('template_item2.npy')
#
#    mario_template = cv2.imread('mario_templete.png', 0)
#    mario_template = mario_template.astype(np.uint8)


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


        img_gray = rgb2gray(unprocessed_state[3])
        img_gray = img_gray.astype(np.uint8)  
        image_with_obj = unprocessed_state[3]
        image_canvas = np.ones_like(image_with_obj)*255
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
            # print(coords_set)
        else:
            coords_set = []
            in_range = False    

#        plt.imshow(image_with_obj)
#        plt.show()
        
        if update_state == True:
            # add last seq to super list, start new sequence
            # print(seq_counter)
            seq_counter = 1     
            # trash next tells us if the sequence we are about to save
            # started with a valied coordinate detection, if it did
            # not we will just ignore, and go to the next
            if not first_input:
#                idx = index_tracker.index(current_num_each)
                '''
                print("idx:            ",index_tracker[idx])
                print("current num each", current_num_each)
                if index_tracker[idx] != current_num_each:
                    print("BLAAAAAAAAAAAAHHHHHH")
                print()
                '''
                print(current_num_each_list)
                print(current_coord_list)            
                for i in range(len(current_num_each_list)):  
                    in_range_sig = in_range_list[i]

                    if in_range_sig:
                        num_each = current_num_each_list[i]
                        current_coord = current_coord_list[i] 
                        if num_each[0]!=0 and num_each[1:]!=[0,0,0]:
                            
                            current_coords =current_coord_list[i]
                            key_list = []
                            for item in current_coords:
                                for item1 in item:
                                    key_list.append(tuple(item1))
                                    
                            if tuple(num_each) not in Dual_Memory:
                                figure_index[tuple(num_each)]=fig_index
                                axsfigure[fig_index//3][fig_index%3].set_title("("+str(fig_index//3)+","+str(fig_index%3)+")")
                                fig_index +=1
                                Dual_Memory[tuple(num_each)] = {}
                                figure_patch[tuple(num_each)] = [key_list]
                                draw_ankor(num_each[1:], current_coords[1:],figure_index[tuple(num_each)],img_combo,fig, axsfigure)               
                            else:
                                if current_coords not in figure_patch[tuple(num_each)]:
                                    figure_patch[tuple(num_each)].append(key_list)

                            key_tuple= tuple(key_list)
                            color_point = torch.argmax(current_policy_list[i]).item()
                            draw_action(key_tuple[0], color[color_point],axsfigure,figure_index[tuple(num_each)])
    #                        print('i',i,len(current_coord_list),len(current_num_each_list))
    #                        print(current_coord_list,'current_num_each_list',current_num_each_list)
                            if i == len(current_coord_list)-1:
                                line_color_loop+=1
                                Dual_Memory[tuple(num_each)][key_tuple] = [current_policy_list[i],None]
                            elif  (i< (len(current_coord_list)-1) and current_num_each_list[i]!=current_num_each_list[i+1]):
                                line_color_loop+=1
                                Dual_Memory[tuple(num_each)][key_tuple] = [current_policy_list[i],None]
                                break
                            else:
                                key_list = []
                                for item in current_coord_list[i+1]:
                                    for item1 in item:
                                        key_list.append(tuple(item1))
                                key_tuple_next= tuple(key_list)    
                                draw_line(key_tuple[0], key_tuple_next[0],color[line_color_loop%10],axsfigure,figure_index[tuple(num_each)])
                                Dual_Memory[tuple(num_each)][key_tuple] = [current_policy_list[i],key_tuple_next]
                                
                                
                                print(key_tuple[0], key_tuple_next[0])
                                print(i,'line_color_loop',line_color_loop)
                            fig.suptitle('External Memory \n  Number of Memory Clusters:'+str(fig_index), fontsize=35)
#                            "   Writing to subfigure: ("+str(fig_index//3)+","+str(fig_index%3)+")"
#                            
                            plt.savefig('image_template/external_memory')

                                       
            current_mario_list = mario_list
            current_goomba_list = goomba_list
            current_pipe_list = tube_list
            current_koopa_list = koopa_list

#            current_num_each = num_each
            current_coords = coords_set
            current_policy_list = [policy]
            current_coord_list = [current_coords]
            current_num_each_list = [current_num_each]
            in_range_list = [in_range]
            # just dont want to add an empty list the first time
            first_input = False
            # lets add the last shit first
        else:
            seq_counter+=1
            current_policy_list.append(policy)
            temp_coords = [mario_list, goomba_list, tube_list,koopa_list]
            current_coord_list.append(coords_set)
            current_num_each_list.append(current_num_each)
            in_range_list.append(in_range)
        
        '''
        mario_location = mario_location.tolist()
        goomba_location = goomba_location.tolist()
        pipe_location = pipe_location.tolist()

        combined_coords = [mario_location, goomba_location, pipe_location]
        '''
        
        # print(combined_coords)
        # print()
        # time.sleep(1)
    ###########################################################################     
        if opt.game =='Supermario':
            state, reward, raw_reward,done, info, unprocessed_state = env.step(action)
        else:
            state, reward, raw_reward,done, info = env.step(action,g_0_cnt,video_save=False)
 
#        if save:
#            print(reward,raw_reward)
        env.render()
        
        
        
        #make video

        im=np.zeros((1600,2500,3),dtype=np.uint8)
        title_raw_image = cv2.imread('image_template/rawimage.png')
        title_brain_canvas = cv2.imread('image_template/braincavas.png')
        height_title_raw_image= title_raw_image.shape[0]
        width_title_rawimage = title_raw_image.shape[1] 
        height_title_brain_canvas= title_brain_canvas.shape[0]
        width_title_brain_canvas = title_brain_canvas.shape[1] 
        
        arrow = cv2.imread('image_template/arrow.png')
        height_arrow= arrow.shape[0]
        width_arrow = arrow.shape[1] 
        im[700:700+height_arrow,280:280+width_arrow,:]=arrow
        im[700:700+height_arrow,800:800+width_arrow,:]=arrow
        try:
            external_memory_fig = cv2.imread('image_template/external_memory.png')
            height_memory = external_memory_fig.shape[0]
            width_memory = external_memory_fig.shape[1]        
            im[100:100+height_memory,1000:1000+width_memory,:]=external_memory_fig
            im[500:500+height_title_brain_canvas,450:450+width_title_brain_canvas,:]=title_brain_canvas
        except:
            print('no external_memory.png')
        height_raw_image= image_with_obj.shape[0]
        width_rawimage = image_with_obj.shape[1]
        im[500:500+height_title_raw_image,50:50+width_title_rawimage,:]=title_raw_image
        im[600:600+height_raw_image,10:10+width_rawimage,:]=image_with_obj
        im[600:600+height_raw_image,450:450+width_rawimage,:]=image_canvas
        RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out.write(RGB_img) 
    
    
    
        # time.sleep(0.3)
        print(info['x_pos'])

        score += raw_reward
        state = torch.from_numpy(state)
        if opt.use_gpu:
            state = state.cuda()
        cum_r = cum_r+reward
#        actions.append(action)
        
        x=info['x_pos']
        if x>=perfect_distance:
            done = True
#        print("x pos", x)
        time.sleep(.2)
        if g_0_cnt==0:
            # time.sleep(1)
            num_interaction+=1

        else:
            pass
            # print(g_0_cnt,num_interaction)
        if done:
            if opt.game=="Supermario":
                x=info['x_pos']
                print(trial_count, x,num_interaction)
                distance_counts.append(x)
                action_counts.append(num_interaction)
                trial_count+=1
        
        if trial_count== num_trials:
            out.release()
            with open('Dual_Memory.pickle', 'wb') as handle:
                pickle.dump(Dual_Memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('figure_patch.pickle', 'wb') as handle:
                pickle.dump(figure_patch, handle, protocol=pickle.HIGHEST_PROTOCOL)
            break

        
#    for idx in super_list:
#        for value in idx:
#            print(value)
#
#    path = opt.save_detector + ".pickle"
#    print(path)
#    with open(path, 'wb') as fp:
#        # pickle dump length
#        pickle.dump(super_list, fp)
#        fp.close()

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
    dis = 0
    for i in range(len(co1)):
        dis = dis + pow(abs(co1[i] - co2[i]), 2)
    return  dis

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
#    start_loc=[start_loc[1],-start_loc[0]]
#    end_loc=[end_loc[1],-end_loc[0]]
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
