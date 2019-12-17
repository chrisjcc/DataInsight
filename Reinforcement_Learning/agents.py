import numpy as np
import torch as T
#from deep_q_network import DeepQNetwork
from dqn import DeepQNetwork, DuelingDeepQNetwork
from replay_memory import ReplayBuffer


class Agent(object):
    '''
    Agent base class
    '''
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions  = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name  = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        # agents memory
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
    
    def choose_action(self, observation): # Example: epsilon-greedy behavior policy for action selection
        raise NotImplementedError

    def store_transition(self, state, action, reward, state_, done):
        '''
           Storing transitions in the agent's memory, sampling those transitions and converting imput into tensors, 
           and decay epsilon, and replacing target network.
        '''
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()


    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        raise NotImplementedError


# Deep Q-Network Agent
class DQNAgent(Agent):
    '''
    Agent based on Deep Q-Network Agent (DQN)
    '''
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # define Q-evaluation network and target Q-network for the agent
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+'_'+self.algo+'_q_eval',
                                   chkpt_dir=self.chkpt_dir)

        # we will never perform gradient descent or backpropagation with Q next network
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name+'_'+self.algo+'_q_next',
                                   chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
         
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # zero out previous gradient calculations
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        # calculate Q-predicted and Q-target values (gives action values for batch states)
        '''
          dims --> batch_size x n_actions
          what the target network has to say about the values of the new states that results from the agent's actions.
          we want to know what are teh values of the maximal actions for that articular set of states.
          we find that by taking the max along the action dimension 
        '''
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0] # 0 max value, 1 index of max value

        # done flag as a type of mask
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward() # backprogate the loss
        self.q_eval.optimizer.step() # step the optimizer to update weight
        self.learn_step_counter += 1 # do this to remember to update target network to the right frequency


# Double Deep Q-Network Agent (rename to DoubleDQN)
class DDQNAgent(Agent):
    '''
    Agent based on Double Deep Q-Nework (Double-DQN)
    '''
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # zero out previous gradient calculations
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        # done flag as a type of mask
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward() # backprogate the loss

        self.q_eval.optimizer.step() # step the optimizer to update weight
        self.learn_step_counter += 1 # do this to remember to update target network to the right frequency

        self.decrement_epsilon()


# Dueling Deep Q-Network Agent 
class DuelingDQNAgent(Agent):
    '''
    Agent based Dueling Deep Q-Network (Dueling DQN)
    '''
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                        input_dims=self.input_dims,
                        name=self.env_name+'_'+self.algo+'_q_eval',
                        chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                        input_dims=self.input_dims,
                        name=self.env_name+'_'+self.algo+'_q_next',
                        chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # zero out previous gradient calculations
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        indices = np.arange(self.batch_size)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                        (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0] 

        # done flag as a type of mask
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next


        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward() # backprogate the loss
        self.q_eval.optimizer.step() # step the optimizer to update weight
        self.learn_step_counter += 1 # do this to remember to update target network to the right frequency

        self.decrement_epsilon()


# Dueling Double Deep Q-Network Agent 
class DuelingDDQNAgent(Agent):
    '''
    Agent based on Dueling Double Deep Q-Network (DDDQN)
    '''
    def __init__(self, *args, **kwargs):
        super(DuelingDDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=self.env_name+'_'+self.algo+'_q_eval',
                                          chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=self.env_name+'_'+self.algo+'_q_next',
                                          chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = np.array([observation], copy=False, dtype=np.float32)
            state_tensor = T.tensor(state).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(state_tensor)

            action = T.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # zero out previous gradient calculations
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        # done flag as a type of mask
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward() # backprogate the loss
        self.q_eval.optimizer.step() # step the optimizer to update weight
        self.learn_step_counter += 1 # do this to remember to update target network to the right frequency

        self.decrement_epsilon()
