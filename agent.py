from torchvision.models import mobilenet_v3_small, vgg16
from torch.utils.data import DataLoader, TensorDataset
from environment import Environment
import setup
from collections import deque
from torch import nn
import torch
import random
import numpy as np
import time
from threading import Thread
import os
from tqdm import tqdm

REPLY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MEMORY_FRACTION = 0.9
MIN_REWARD = -200
EPISODES = 100
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 # 0.99975
MIN_EPSILON = 0.001
AGGRAGATE_STATS_EVERY = 10

class DQNAgent():
    def __init__(self, IMG_HEIGHT, IMG_WIDTH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.replay_memory = deque(maxlen=REPLY_MEMORY_SIZE)
        
        self.terminate = False
        self.training_initialized = False
        
    def create_model(self):
        
        model = mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardshrink(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, 3)
        )
        
        """ model = vgg16(weights=None)
        
        # 3 classes (right, forward, left)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 3)
        ) """
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        return model.to(self.device)
        
        
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)
        
    def train(self):
        X = []
        y = []
        
        self.model.train()
        
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

    # Obtem os Q values para o estado atual para todas as ações possíveis e os Q values do estado futuro (depois da action)
    # Por cada transição, calcula o novo Q value e atualiza o Q value para a ação tomada no estado atual
        current_states = (torch.tensor([transition[0] for transition in minibatch])/255).to(self.device)
        current_qs_list = self.model(current_states) # Get the Q-value for all possible actions
            
        new_current_states = (torch.tensor([transition[3] for transition in minibatch])/255).to(self.device)
        future_qs_list = self.target_model(new_current_states)
        
        
        
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                # Compute the new q-value
                max_future_q = torch.max(future_qs_list[index]) # The computation of the new_q is based on the new_state Q-values
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = new_q # update the Q-value for the action taken
            
            X.append(current_state)
            y.append(current_qs)

            
        # TRAINING  
        X = torch.tensor(X/255).to(self.device)
        y = torch.tensor(y).to(self.device)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=False)

        final_loss = 0
        for (minibach_X, minibach_y) in dataloader:
            
            self.optimizer.zero_grad()
            
            y_pred = self.model(minibach_X)
            loss = self.loss_fn(y_pred, minibach_y)
            
            loss.backward()
            self.optimizer.step()
            
            final_loss += loss
            
        final_loss /= len(dataloader)
        print("Loss: ", final_loss)

        # Update target model weights every episodes ->         TESTAR EM CADA 5 EPISODIOS
        self.target_model.load_state_dict(self.model.state_dict())
        
            
    def get_qs(self, state):
        state_clone = state.clone().detach().requires_grad_(True)
        state_clone = torch.tensor(state_clone/255).reshape(-1, *state_clone.shape)[0]
        state_clone = state_clone.to(self.device)
        
        return self.model(state_clone)
    
    
    # SUPSOTAMENTE É PRECISO FAZER UM TRAIN DE 1 ANTES DE FAZER O TRAIN EM LOOP...
    def train_in_loop(self):
        X = torch.rand(1, self.IMG_HEIGHT, self.IMG_WIDTH, 3).to(self.device)
        y = torch.rand(1, 3).to(self.device)
                
        X = X.permute(0, 3, 1, 2)
                
        self.model.train()
        
        self.optimizer.zero_grad()
        
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        
        loss.backward()
        self.optimizer.step()
        
        self.training_initialized = True
        
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)        
        
        
if __name__ == "__main__":
    FPS = 20
    ep_rewards = [-200]
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    env = Environment()
    agent = DQNAgent(env.IMG_HEIGHT, env.IMG_WIDTH)
    
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    
    while not agent.training_initialized:
        time.sleep(0.01)
        
    state = torch.ones((env.IMG_HEIGHT, env.IMG_WIDTH, 3))
    state = state.unsqueeze(0)
    state = state.permute(0, 3, 1, 2)
    
    agent.get_qs(state)
    
    for episode in tqdm(range(1, EPISODES+1), unit="episodes"):
        env.collision_hist = []
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()
        
        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state)) # Get the action with the highest Q-value
            else:
                action = np.random.randint(0, 3) # Get a random action (1, 2, 3)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            
            step += 1
                
            if done:
                break
        
        setup.destroy_actors(env.actor_list)

        
    agent.terminate = True
    trainer_thread.join()
    agent.model.save("models/model.model")