from tkinter import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Constants
GAME_WIDTH = 500
GAME_HEIGHT = 500
SPACE_SIZE = 40
SPEED = 30
SNAKE_COLOR = "#00FF00"
FOOD_COLOR = "#FF0000"
BACKGROUND_COLOR = "#000000"


# =========================================🧠 MODEL LOGIC==========================================
# DQN model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.fc(x)

def state_to_tensor(state):
    head = state["snake_head"]
    food = state["food"]
    direction = state["direction"]
    body = state["snake_body"]

    return torch.FloatTensor([
        int(food[0] < head[0]),
        int(food[0] > head[0]),
        int(food[1] < head[1]),
        int(food[1] > head[1]),
        int(direction == "left"),
        int(direction == "right"),
        int(direction == "up"),
        int(direction == "down"),
        int(head in body),
        int(head[0] < 0 or head[0] >= GAME_WIDTH),
        int(head[1] < 0 or head[1] >= GAME_HEIGHT)
    ])
# ======================================================================================================


# ==============================================🎮 ENVIRONMENT LOGIC ========================================
class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [[SPACE_SIZE * 5, SPACE_SIZE * 5]]
        self.direction = "right"
        self.spawn_food()
        self.score = 0
        self.done = False
        return self.get_state()

    def spawn_food(self):
        while True:
            x = random.randint(0, (GAME_WIDTH // SPACE_SIZE) - 1) * SPACE_SIZE
            y = random.randint(0, (GAME_HEIGHT // SPACE_SIZE) - 1) * SPACE_SIZE
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    def step(self, action):
        if action == 0 and self.direction != "down":
            self.direction = "up"
        elif action == 1 and self.direction != "up":
            self.direction = "down"
        elif action == 2 and self.direction != "right":
            self.direction = "left"
        elif action == 3 and self.direction != "left":
            self.direction = "right"

        x, y = self.snake[0]
        if self.direction == "up":
            y -= SPACE_SIZE
        elif self.direction == "down":
            y += SPACE_SIZE
        elif self.direction == "left":
            x -= SPACE_SIZE
        elif self.direction == "right":
            x += SPACE_SIZE

        new_head = [x, y]
        self.snake.insert(0, new_head)

        if x < 0 or x >= GAME_WIDTH or y < 0 or y >= GAME_HEIGHT or new_head in self.snake[1:]:
            self.done = True
            return self.get_state(), -10, self.done

        if new_head == self.food:
            self.score += 1
            self.spawn_food()
            return self.get_state(), 10, self.done
        else:
            self.snake.pop()
            return self.get_state(), 0, self.done

    def get_state(self):
        return {
            "snake_head": self.snake[0],
            "snake_body": self.snake[1:],
            "food": self.food,
            "direction": self.direction
        }

# =======================================================================================================

# RL Setup
env = SnakeEnv()
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
memory = deque(maxlen=1000)
epsilon = 1.0
scores = []

# GUI Setup
root = Tk()
root.title("Snake AI with Live Score")

frame = Frame(root)
frame.pack(side=LEFT)

canvas = Canvas(frame, bg=BACKGROUND_COLOR, height=GAME_HEIGHT, width=GAME_WIDTH)
canvas.pack()

score_label = Label(root, text="Score: 0", font=("Arial", 16))
score_label.pack()

episode_label = Label(root, text="Episode: 0", font=("Arial", 16))
episode_label.place(x=GAME_WIDTH + 50, y=2)


# Matplotlib Plot in Tkinter
fig, ax = plt.subplots(figsize=(4, 4))
score_plot, = ax.plot([], [], 'b-')
ax.set_title("Episode Scores")
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
ax.set_ylim(0, 30)

plot_canvas = FigureCanvasTkAgg(fig, master=root)
plot_canvas.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=1)
plot_canvas.draw()



# ============================================🚀 MAIN TRAINING LOOP======================================
episode = 0
def train_one_episode():
    global epsilon, episode
    episode_label.config(text=f"Episode: {episode}")

    state = env.reset()
    total_reward = 0
    canvas.delete("all")
    episode += 1
    episode_label.config(text=f"Episode: {episode}") 


    def step_loop():
        global epsilon 
        nonlocal state, total_reward

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            q_vals = model(state_to_tensor(state))
            action = torch.argmax(q_vals).item()

        next_state, reward, done = env.step(action)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        canvas.delete("snake")
        canvas.delete("food")

        for x, y in env.snake:
            canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=SNAKE_COLOR, tag="snake")

        fx, fy = env.food
        canvas.create_oval(fx, fy, fx + SPACE_SIZE, fy + SPACE_SIZE, fill=FOOD_COLOR, tag="food")

        score_label.config(text=f"Score: {env.score}")

        if done:
            scores.append(env.score)
            update_plot()
            if len(memory) >= 64:
                batch = random.sample(memory, 64)
                for s, a, r, s_, d in batch:
                    q_vals = model(state_to_tensor(s))
                    target = q_vals.clone().detach()
                    target[a] = r if d else r + 0.9 * torch.max(model(state_to_tensor(s_))).item()
                    loss = criterion(q_vals, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            epsilon = max(0.1, epsilon * 0.995)
            root.after(500, train_one_episode)  #  next episode starts
        else:
            root.after(SPEED, step_loop)

    step_loop()

# ====================================================================================================

def update_plot():
    score_plot.set_data(range(len(scores)), scores)
    ax.set_xlim(0, max(10, len(scores)))
    ax.set_ylim(0, max(10, max(scores) + 5))
    plot_canvas.draw()

# Start training 
train_one_episode()
root.mainloop()
