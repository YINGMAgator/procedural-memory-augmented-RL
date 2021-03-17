#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:46:03 2019

@author: yingma
"""
import torch
from src.env import create_train_env,create_train_env_atari


from src.model import ActorCritic_seq
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
#from tensorboardX import SummaryWriter
import timeit
import numpy as np

def local_test_iter(opt,env,local_model_test,Cum_reward,SCORE,X,Num_interaction,videosave,action_max,gate_max): 


    curr_step_test = 0
    cum_r=0
    actions = deque(maxlen=opt.max_actions)
    
    with torch.no_grad():
        h_0 = torch.zeros((1, 512), dtype=torch.float)
        c_0 = torch.zeros((1, 512), dtype=torch.float)
        g_0_ini = torch.ones((1))
        
        g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
        
        env.reset()
        # randomly go through random steps for a specific interval
        if opt.start_initial =='random':
            for i in range(opt.start_interval):
                if opt.game =='Supermario':
                    state, reward, _,done, info = env.step(env.action_space.sample())
                else:
                    state, reward, _,done, info = env.step(env.action_space.sample(),0,video_save=videosave)
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
#    if save:
#        print('1111111111111111')
    while True:
        env.render()

        curr_step_test += 1
        with torch.no_grad():
            h_0 = h_0.detach()
            c_0 = c_0.detach()
    
        if gate_max:
            logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = local_model_test(state, h_0, c_0,g_0,g_0_ini,certain=True)
        else:
            logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = local_model_test(state, h_0, c_0,g_0,g_0_ini)
        g_0_ini = torch.zeros((1))
        if opt.use_gpu:
            g_0_ini = g_0_ini.cuda()         
        policy = F.softmax(logits, dim=1)
        

        if action_max:
            action = torch.argmax(policy).item()
        else:
            m = Categorical(policy)
            action = m.sample().item()  
            
            
        if opt.game =='Supermario':
            state, reward, raw_reward,done, info = env.step(action)
        else:
            state, reward, raw_reward,done, info = env.step(action,g_0_cnt,video_save=videosave)
 
#        if save:
#            print(reward,raw_reward)
#            env.render()
#            time.sleep(0.5)
        score += raw_reward
        state = torch.from_numpy(state)
        if opt.use_gpu:
            state = state.cuda()
        cum_r = cum_r+reward
        actions.append(action)
        if g_0_cnt==0:
            num_interaction+=1
        if curr_step_test > opt.max_test_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        
        if done:
            if opt.game=="Supermario":
                x=info['x_pos']
               
                X.append(x)
            else:
                x=score
            SCORE.append(score)
            Cum_reward.append(cum_r)   
            Num_interaction.append(num_interaction)     

            env.close()               
            break

    return Cum_reward,SCORE,X,Num_interaction,x


#def local_test2_allmax(opt,env,local_model_test,Cum_reward,X,Num_interaction,save): 
#
#
#    curr_step_test = 0
#    cum_r=0
#    actions = deque(maxlen=opt.max_actions)
#    
#    with torch.no_grad():
#        h_0 = torch.zeros((1, 512), dtype=torch.float)
#        c_0 = torch.zeros((1, 512), dtype=torch.float)
#        g_0_ini = torch.ones((1))
#
#        g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
#        env.reset()
#        if opt.start_initial =='random':
#            for i in range(opt.start_interval):
#                state, reward,_, done, info = env.step(env.action_space.sample())
#                if done:
#                    env.reset()       
#            state=torch.from_numpy(state)
#        else:
#            state = torch.from_numpy(env.reset())
#        if opt.use_gpu:
#            state = state.cuda()        
#            h_0 = h_0.cuda()
#            c_0 = c_0.cuda()
#            g_0_ini = g_0_ini.cuda() 
#            g_0 = g_0.cuda()
#        
#    num_interaction=1
#    score = 0
#    while True:
#        curr_step_test += 1
#        with torch.no_grad():
#            h_0 = h_0.detach()
#            c_0 = c_0.detach()
#    
#        logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = local_model_test(state, h_0, c_0,g_0,g_0_ini,certain=True)
#        g_0_ini = torch.zeros((1))
#        if opt.use_gpu:
#            g_0_ini = g_0_ini.cuda()         
#        policy = F.softmax(logits, dim=1)
#        action = torch.argmax(policy).item()
#        state, reward, raw_reward,done, info = env.step(action)
#        score += raw_reward
#        state = torch.from_numpy(state)
#        if opt.use_gpu:
#            state = state.cuda()
#        cum_r = cum_r+reward
#        actions.append(action)
#        if g_0_cnt==0:
#            num_interaction+=1
#        if curr_step_test > opt.max_test_steps or actions.count(actions[0]) == actions.maxlen:
#            done = True
#        
#        if done:
#            if opt.game=="Supermario":
#                x=info['x_pos']
#            else:
#                x=score
#            X.append(x)
#            Cum_reward.append(cum_r)   
#            Num_interaction.append(num_interaction)                    
#            break
#    
#    return Cum_reward,X,Num_interaction,x
#
#
#def local_test3_actionpro_gatemax(opt,env,local_model_test,Cum_reward,X,Num_interaction,save): 
#
#
#    curr_step_test = 0
#    cum_r=0
#    actions = deque(maxlen=opt.max_actions)
##    if save:
##        print('3333333333333333333')
#    with torch.no_grad():
#        h_0 = torch.zeros((1, 512), dtype=torch.float)
#        c_0 = torch.zeros((1, 512), dtype=torch.float)
#        g_0_ini = torch.ones((1))
#        g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
#        env.reset()
#        if opt.start_initial =='random':
#            for i in range(opt.start_interval):
#                state, reward,_, done, info = env.step(env.action_space.sample())
#                if done:
#                    env.reset()       
#            state=torch.from_numpy(state)
#        else:
#            state = torch.from_numpy(env.reset())
#        if opt.use_gpu:
#            state = state.cuda()        
#            h_0 = h_0.cuda()
#            c_0 = c_0.cuda()
#            g_0_ini = g_0_ini.cuda() 
#            g_0 = g_0.cuda()
#            
#    num_interaction=1
#    score=0
#    while True:
#        curr_step_test += 1
#        with torch.no_grad():
#            h_0 = h_0.detach()
#            c_0 = c_0.detach()
#    
#        logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = local_model_test(state, h_0, c_0,g_0,g_0_ini,certain=True)
#        g_0_ini = torch.zeros((1))
#        if opt.use_gpu:
#            g_0_ini = g_0_ini.cuda() 
#            
#            
#        policy = F.softmax(logits, dim=1)
#        
#        m = Categorical(policy)
#        action = m.sample().item()
#
#        state, reward, raw_reward,done, info = env.step(action)
##        if save:
##            env.render()
##            time.sleep(0.2)
#        score+=raw_reward
##        print(score)
#        state = torch.from_numpy(state)
#        if opt.use_gpu:
#            state = state.cuda()
#        cum_r = cum_r+reward
#        actions.append(action)
#        if g_0_cnt==0:
#            num_interaction+=1
#        if curr_step_test > opt.max_test_steps or actions.count(actions[0]) == actions.maxlen:
#            done = True
#        
#        if done:
#            if opt.game=="Supermario":
#                x=info['x_pos']
#            else:
#                x=score
#            X.append(x)
#            Cum_reward.append(cum_r)   
#            Num_interaction.append(num_interaction)                    
#            break
#    return Cum_reward,X,Num_interaction,x



def local_train(index, opt, global_model, optimizer, save=False):
#    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
#    writer = SummaryWriter(opt.log_path)
    if not opt.saved_path:
        if opt.game == "Supermario":
            saved_path="{}_{}_{}_{}".format(opt.game,opt.num_sequence,opt.internal_reward,opt.world,opt.stage)
        else:
            saved_path="{}_{}".format(opt.game,opt.num_sequence)
    else:
        saved_path=opt.saved_path
    if opt.game == "Supermario":
        env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:

        env, num_states, num_actions = create_train_env_atari(opt.game,saved_path,output_path=None)    
    local_model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    
    loss_matrix = []
    Cum_reward1 = []
    SCORE1 = []
    X1 = []
    Num_interaction1=[]
    
    if opt.game == "Supermario":
        env1, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        env1, num_states, num_actions = create_train_env_atari(opt.game,saved_path,output_path=None)    
    local_model1 = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if opt.use_gpu:
        local_model1.cuda()
    local_model1.eval() 

    Cum_reward2 = []
    SCORE2=[]
    X2 = []
    Num_interaction2=[]
    
    if opt.game == "Supermario":
        env2, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        env2, num_states, num_actions = create_train_env_atari(opt.game, saved_path,output_path=None)              
    local_model2 = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if opt.use_gpu:
        local_model2.cuda()
    local_model2.eval() 

    Cum_reward3 = []
    SCORE3=[]
    X3 = []
    Num_interaction3=[]
    if opt.game == "Supermario":
        env3, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        env3, num_states, num_actions = create_train_env_atari(opt.game, saved_path,output_path=None)
        
    local_model3 = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if opt.use_gpu:
        local_model3.cuda()
    local_model3.eval() 

    
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                if opt.game=='Supermario':
#                    torch.save(global_model.state_dict(),
#                               "{}/a3c_seq_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
                    torch.save(global_model.state_dict(),saved_path+"/trained_model")    
                else:
                    torch.save(global_model.state_dict(),saved_path+"/trained_model")                    
#            print("Process {}. Episode {}".format(index, curr_episode),done)
        
        if curr_episode%opt.log_interval==0:
 
            if opt.game=='Supermario':
#                local_model1.load_state_dict(global_model.state_dict())            
#                Cum_reward1,X1,Num_interaction1,x_arrive_all_pro = local_test_iter(opt,env1,local_model1,Cum_reward1,X1,Num_interaction1,save) 
#            local_model2.load_state_dict(global_model.state_dict())
 #           Cum_reward2,SCORE2,X2,Num_interaction2,x_arrive_all_pro = local_test_iter(opt,env2,local_model2,Cum_reward2,SCORE2,X2,Num_interaction2,videosave=False,action_max=False,gate_max=False)    
                local_model2.load_state_dict(global_model.state_dict())            
                Cum_reward2,SCORE2,X2,Num_interaction2,x_arrive_all_max = local_test_iter(opt,env2,local_model2,Cum_reward2,SCORE2,X2,Num_interaction2,videosave=False,action_max=True,gate_max=True)
            
                local_model3.load_state_dict(global_model.state_dict())            
                Cum_reward3,SCORE3,X3,Num_interaction3,x_arrive_actionpro_gatemax = local_test_iter(opt,env3,local_model3,Cum_reward3,SCORE3,X3,Num_interaction3,videosave=False,action_max=False,gate_max=True)
                # print("Here")
                print(curr_episode,x_arrive_all_max,x_arrive_actionpro_gatemax )
            else:
                local_model1.load_state_dict(global_model.state_dict())
                Cum_reward1,SCORE1,X1,Num_interaction1,x_arrive_all_pro = local_test_iter(opt,env1,local_model1,Cum_reward1,SCORE1,X1,Num_interaction1,videosave=False,action_max=False,gate_max=False)
                print(curr_episode,x_arrive_all_pro)

                
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
#        g_0_cnt = 0 
        if done:
            g_0_ini = torch.ones((1))
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
            cum_r=0
            g_0_cnt = 0 
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
#            g_0 = g_0.detach()
        
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            g_0_ini = g_0_ini.cuda()
            g_0 = g_0.cuda()

        log_policies = []
        log_gates = []
        values = []
        rewards = []
        reward_internals = []
        entropies = []

        for aaaaa in range(opt.num_local_steps):
            curr_step += 1
            g_pre = g_0
            g_pre_cnt = g_0_cnt

            logits, value, h_0, c_0, g_0,g_0_cnt,gate_flag1, gate_flag2= local_model(state, h_0, c_0,g_0,g_0_ini)

            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()
            state, reward,raw_reward, done, info = env.step(action)
            reward_internal = reward
            
            if g_0_ini==1:

                log_gate = torch.zeros((), dtype=torch.float)
                if opt.use_gpu:
                    log_gate = log_gate.cuda()
            elif gate_flag1:

#                log_gate = log_gate
                log_gate = torch.zeros((), dtype=torch.float)
            elif gate_flag2:
                  
#                log_gate = log_gate + torch.log(1-g_pre[0,g_pre_cnt]) 
                log_gate = torch.log(1-g_pre[0,g_pre_cnt]) 
            else:
#                log_gate = log_gate+torch.log(g_0[0,g_0_cnt-1])
                log_gate = torch.log(g_0[0,g_0_cnt-1])
                if reward>0:
                    reward_internal = reward+opt.internal_reward
            g_0_ini = torch.zeros((1))
            if opt.use_gpu:
                g_0_ini = g_0_ini.cuda()
#            if save:
#                env.render()
#                print(reward)
#                time.sleep(1)  
            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True
                print('max glabal step achieve')

            if done:

                curr_step = 0

                env.reset()
                if opt.start_initial =='random':
                    for i in range(opt.start_interval):
                        state, reward, _,done, info = env.step(env.action_space.sample())
                        if done:
                            env.reset()       
                    state=torch.from_numpy(state)
                else:
                    state = torch.from_numpy(env.reset())
                if opt.use_gpu:
                    state = state.cuda()            
            
            

            values.append(value)
            log_policies.append(log_policy[0, action])
            log_gates.append(log_gate)
            rewards.append(reward)
            reward_internals.append(reward_internal)
            entropies.append(entropy)
            cum_r+=reward
            if done:
                break
#        print(log_policies,log_gates)
        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ ,_,_,gate_flag1, gate_flag2= local_model(state, h_0, c_0,g_0,g_0_ini,gate_update=False)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        
#        next_value = R
#        for value, log_policy, log_gate, reward, reward_internal, entropy in list(zip(values, log_policies, log_gates, rewards,reward_internals, entropies))[::-1]:
#            gae = gae * opt.gamma * opt.tau
#            gae = gae + reward_internal + opt.gamma * next_value.detach() - value.detach()
#            next_value = value
#            actor_loss = actor_loss + (log_policy+log_gate) * gae
#            R = R * opt.gamma + reward
#            critic_loss = critic_loss + (R - value) ** 2 / 2
#            entropy_loss = entropy_loss + entropy
        
# estimate internal reward directly      
        if not (gate_flag1 or gate_flag2):
            if R >0:    
                R=R+opt.internal_reward
        next_value = R  
        for value, log_policy, log_gate, reward, reward_internal, entropy in list(zip(values, log_policies, log_gates, rewards,reward_internals, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward_internal + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + (log_policy+log_gate) * gae
            R = R * opt.gamma + reward_internal
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
            
# estimate external reward      

#        next_value = R  
#        for value, log_policy, log_gate, reward, reward_internal, entropy in list(zip(values, log_policies, log_gates, rewards,reward_internals, entropies))[::-1]:
#            gae = gae * opt.gamma * opt.tau
#            gae = gae + reward_internal-0.01* + opt.gamma * next_value.detach() - value.detach()
#            next_value = value
#            actor_loss = actor_loss + (log_policy+log_gate) * gae
#            R = R * opt.gamma + reward
#            critic_loss = critic_loss + (R - value) ** 2 / 2
#            entropy_loss = entropy_loss + entropy
            
 


        if opt.value_loss_coef:
            total_loss = -actor_loss + critic_loss*opt.value_loss_coef - opt.beta * entropy_loss
        else:
            total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
#        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        if opt.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), opt.max_grad_norm)
        
        
        loss_matrix.append(total_loss.detach().cpu().numpy())
        
        
        if curr_episode%opt.save_interval==0:
#            print('aaaaaaaaaaa',X,Cum_reward)
            if opt.game=='Supermario':  
                np.save(saved_path+"/X1{}".format(index), X1)
                np.save(saved_path+"/X2{}".format(index), X2)
                np.save(saved_path+"/X3{}".format(index), X3)
                                  
            np.save(saved_path+"/loss{}".format(index), loss_matrix)
            np.save(saved_path+"/Cum_reward1{}".format(index), Cum_reward1)
            np.save(saved_path+"/SCORE1{}".format(index), SCORE1)
            np.save(saved_path+"/Num_interaction1{}".format(index), Num_interaction1)

            np.save(saved_path+"/Cum_reward2{}".format(index), Cum_reward2)
            np.save(saved_path+"/SCORE2{}".format(index), SCORE2)
            np.save(saved_path+"/Num_interaction2{}".format(index), Num_interaction2)

            np.save(saved_path+"/Cum_reward3{}".format(index), Cum_reward3)
            np.save(saved_path+"/SCORE3{}".format(index), SCORE3)
            np.save(saved_path+"/Num_interaction3{}".format(index), Num_interaction3)           
             
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    
    if opt.game == "Supermario":
        env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        
        if not opt.saved_path:
            saved_path="{}_{}_{}_{}".format(opt.game,opt.num_sequence,opt.internal_reward,opt.lr)
        env, num_states, num_actions = create_train_env_atari(opt.game,saved_path,output_path="test111")
        
        
    local_model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    local_model.eval()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    Cum_reward = []
    X = []
    i=0
    
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
                g_0_ini = torch.ones((1))
                state = torch.from_numpy(env.reset())
                cum_r=0
                g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
                score=0
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = local_model(state, h_0, c_0,g_0,g_0_ini)
        #print(g_0,g_0_cnt)
        g_0_ini = torch.zeros((1))
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward,raw_reward, done, info = env.step(action)
#        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        cum_r = cum_r+reward
        if done:

            i=i+1
            curr_step = 0
            actions.clear()
            state = env.reset()
            if opt.game == "Supermario":
                x=info['x_pos']
            else:
                x=score                
            print(i,'test_certain',x)  
            X.append(info['x_pos'])
            Cum_reward.append(cum_r)
                
                
        state = torch.from_numpy(state)
        
        if i%100==0:

            np.save("{}/Cum_reward_test".format(opt.saved_path), Cum_reward)
            np.save("{}/X_test".format(opt.saved_path), X)
            
def local_test_certain(index, opt, global_model):
    torch.manual_seed(123 + index)
    
    if opt.game == "Supermario":
        env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        if not opt.saved_path:
            saved_path="{}_{}_{}_{}".format(opt.game,opt.num_sequence,opt.internal_reward,opt.lr)
        env, num_states, num_actions = create_train_env_atari(opt.game, saved_path,output_path=None)
        
    local_model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    local_model.eval()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    Cum_reward = []
    X = []
    i=0

    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
                g_0_ini = torch.ones((1))
                state = torch.from_numpy(env.reset())
                cum_r=0
                g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
                score=0
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = local_model(state, h_0, c_0,g_0,g_0_ini,certain=True)
        #print(g_0,g_0_cnt)
        g_0_ini = torch.zeros((1))
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward,raw_reward, done, info = env.step(action)
        score+=raw_reward
#        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        cum_r = cum_r+reward
        if done:
            
            i=i+1
            curr_step = 0
            actions.clear()
            state = env.reset()
            if opt.game == "Supermario":
                x=info['x_pos']
            else:
                x=score
                
            print(i,'test_certain',x)
            X.append(x)
            Cum_reward.append(cum_r)
    
        state = torch.from_numpy(state)      
        
        if i%100==0:
            np.save("{}/Cum_reward_test_certain".format(opt.saved_path), Cum_reward)
            np.save("{}/X_test_certain".format(opt.saved_path), X)
