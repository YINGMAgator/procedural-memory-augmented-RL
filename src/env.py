#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:08:13 2020

@author: yingma
"""

"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import gym_super_mario_bros
import gym
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
import copy


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None,final_step=5000):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.final_step=final_step
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
#        self.STATE=np.zeros((1500,240,256,3))
#        self.i=0
#        self.j=0
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        raw_reward = copy.deepcopy(reward) 
        unprocess_state=copy.deepcopy(state)  ## CHANGED
#        self.STATE[self.i]=state
#        self.i+=1        
#        if self.i>1087:
#            np.save('STATE'+str(self.j),self.STATE)
#        print(self.i)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
#        print(info["score"],reward,(info["score"] - self.curr_score) / 40.)
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50

        if info["x_pos"]>self.final_step:
            done=True
            reward += 50
        return state, reward / 10., raw_reward,done, info,unprocess_state

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomReward_atari(Wrapper):
    def __init__(self, env=None, monitor=None,save_path=None,final_step=5000):
        super(CustomReward_atari, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.final_step=final_step
        self.current_live=3
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.STATE=[]
        self.CNT=[]
        self.save_path=save_path
    def step(self, action,g_0_cnt=0,video_save=False):
        state, reward, done, info = self.env.step(action)
        raw_reward = copy.deepcopy(reward)    
#        print('raw_reward',raw_reward)
        if video_save:
            self.STATE.append(state)
            self.CNT.append(g_0_cnt)
            if done:
                np.save(self.save_path+'/STATE',self.STATE)
                np.save(self.save_path+'/CNT',self.CNT)
                self.STATE=[]
                self.CNT=[]
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward=reward
        if info['ale.lives']==2 or info['ale.lives']==1:
            aaaaa=1
            pass
#            print(self.current_live)
        if info['ale.lives']==self.current_live-1:
            self.current_live=info['ale.lives']
            reward -= 50

        return state, reward / 50.,raw_reward, done, info

    def reset(self):
        self.current_live = 3
        return process_frame(self.env.reset())
    
    
    

class CustomSkipFrame(Wrapper):
    def __init__(self, env, game,skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.game= game
        
    def step(self, action,g_0_cnt=0,video_save=False):
        total_reward = 0
        total_raw_reward = 0
        states = []
        STATE=[] ## CHANGED
        if self.game=="Supermario":
            state, reward, raw_reward,done, info,unprocess_state = self.env.step(action) ## CHANGED
        else:
            state, reward, raw_reward,done, info = self.env.step(action,g_0_cnt,video_save)
        total_reward += reward
        total_raw_reward += raw_reward
        states.append(state) 
        STATE.append(unprocess_state)      ## CHANGED  
        for i in range(self.skip-1):

            if not done:
                if self.game=="Supermario":
                    state, reward, raw_reward,done, info ,unprocess_state= self.env.step(action)
                else:
                    state, reward, raw_reward,done, info = self.env.step(action,g_0_cnt,video_save)
                total_reward += reward
                total_raw_reward += raw_reward
                states.append(state)
                STATE.append(unprocess_state)  ## CHANGED
            else:
                states.append(state)
                STATE.append(unprocess_state)  ## CHANGED
#            print(done, total_raw_reward)
#            print(reward, total_reward)
        states = np.concatenate(states, 0)[None, :, :, :]
#        print('raw_total_reward',total_raw_reward)
        return states.astype(np.float32), total_reward, total_raw_reward,done, info,STATE

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)


def create_train_env(world, stage, action_type, final_step,output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    game="Supermario"
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor,final_step)
    env = CustomSkipFrame(env,game)
    return env, env.observation_space.shape[0], len(actions)

def create_train_env_atari(game,save_path,output_path=None):
#    game=opt.game
    env = gym.make(game,frameskip=4)
    if output_path:
        monitor = Monitor(210, 160, output_path)
    else:
        monitor = None
    
    env = CustomReward_atari(env, monitor,save_path)
    env = CustomSkipFrame(env,game)    
    return env, env.observation_space.shape[0], env.action_space.n