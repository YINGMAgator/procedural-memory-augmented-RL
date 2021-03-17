"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import copy
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx



class ActorCritic_seq(nn.Module):
    def __init__(self, num_inputs, num_actions,num_sequence):
        super(ActorCritic_seq, self).__init__()
        self.num_sequence = num_sequence
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        
        
        self.conv1_gate = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2_gate = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3_gate = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4_gate = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        if self.num_sequence!=0:
            self.gate_linear = nn.Linear(32 * 6 * 6, self.num_sequence)
        self.counter = 0

#        self.bnl = Bernoulli (0.5)
#        self.g = torch.zeros((1, self.num_sequence), dtype=torch.float)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx,g,g_ini,gate_update=True,certain=False):
        update_state = False
        if g_ini==1 or self.counter == self.num_sequence:
            seq_ini_flag1 = True
            seq_ini_flag2 = False
        else:
            seq_ini_flag1 = False
            if certain:
                bnl = Bernoulli (g.data[0][self.counter].round())
            else:
                bnl = Bernoulli (g.data[0][self.counter])
            gate_sample=bnl.sample()
#            print(gate_sample)
            if gate_sample==1:
#                print("yes")
                seq_ini_flag2 = False
                if gate_update:
                    self.counter+=1
            else:
                seq_ini_flag2 = True
        if gate_update:       
            if seq_ini_flag1 or seq_ini_flag2:
                update_state = True
                self.counter = 0
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))                  
                self.x_pre = x
           
                if self.num_sequence!=0: 
                    g = torch.sigmoid(self.gate_linear(x.view(x.size(0),-1)))
                else:
                    self.g = torch.zeros((1, self.num_sequence), dtype=torch.float)
            # print("Update State", update_state)                  
            hx, cx = self.lstm(self.x_pre.view(self.x_pre.size(0), -1), (hx, cx))   
        else:
            if seq_ini_flag1 or seq_ini_flag2:
                update_state= True
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))  
                hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))   
            else:
                hx, cx = self.lstm(self.x_pre.view(self.x_pre.size(0), -1), (hx, cx)) 
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx, g,self.counter,seq_ini_flag1,seq_ini_flag2, update_state
