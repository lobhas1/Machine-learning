
# Import libraries
import pygame
import numpy as np
import os
import random
import pickle
from multiprocessing import Process,freeze_support, Queue
from functools import partial
import cv2
import pygame.surfarray
#constants
WIDTH, HEIGHT = 800, 800
BACKGROUND_COLOR = (0, 0, 0)
STARTBUTTONAREA =200
TILE_REWARD = 20
COMPLETE_REWARD = 400
MOVE_REWARD = 50
DISCOUNT = 0.95
LEARNING_RATE = 0.1
EPISODES = 25000
TILE_DESTROY_REWARD = 50
BOUNCE_REWARD = 20
GAME_OVER_REWARD = -1000
GAME_COMPLETE_REWARD = 5000
Game_complete = False
Game_over = False
#game setup


# Define the class for the tiles of the game
class Tile:

    # Set the dimensions of the tiles
    Width = 160
    Height = 32

    # Define the generator
    def __init__(self,x,y,tile_type = 0):
        # Initiallize the tiles position and dimension
        # tile_type changs the colour of the tile    
        self.x = x
        self.y = y
        self.width = self.Width
        self.height = self.Height
        self.tile_type = tile_type
        
    # Define a function to change tile type
    def change_tile_type(self,tile_type):
        self.tile_type = tile_type
    def draw(self,screen):
        if self.tile_type == 0:          # normal
            self.tile_colour = (255,255,255)
        elif self.tile_type == 1:        # start
            self.tile_colour = (0,255,0)
        elif self.tile_type == 2:        # stop
            self.tile_colour = (255,0,0)
        elif self.tile_type == 3:        # seen
            self.tile_colour = (255,255,0)
        elif self.tile_type == 4:        # wall
            self.tile_colour = (0,0,0)   
        elif self.tile_type == 5:        # searched
            self.tile_colour = (160,32,240) 
        elif self.tile_type == 6:
            self.tile_colour = (0,0,255)
        
        pygame.draw.rect(screen,self.tile_colour,(self.x,self.y,self.Width,self.Height))

    def __sub__(self,other):
        return (abs(self.x-other.x)//self.Width,abs(self.y-other.y)//self.Height)
    def __lt__(self,other):
        return False
    

def rect_circle_collision(circle_center, circle_radius, rect):
    cx, cy = circle_center
    rx, ry, rw, rh = rect.x,rect.y,rect.width,rect.height
  

    # Find the closest point on the rectangle to the circle's center
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))

    # Calculate the distance between the circle's center and this point
    distance_x = cx - closest_x
    distance_y = cy - closest_y

    # Check if the distance is less than or equal to the radius
    distance_squared = distance_x ** 2 + distance_y ** 2
    return distance_squared <= circle_radius ** 2

def ball_player_collision(circle_center, circle_radius, rect):
    cx, cy = circle_center
    rx, ry, rw, rh = rect
  

    # Find the closest point on the rectangle to the circle's center
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))

    # Calculate the distance between the circle's center and this point
    distance_x = cx - closest_x
    distance_y = cy - closest_y

    # Check if the distance is less than or equal to the radius
    distance_squared = distance_x ** 2 + distance_y ** 2
    return distance_squared <= circle_radius ** 2

# Define class of player
class Player:
    def __init__(self,x,y):
        # Define position and dimensions
        self.x = x
        self.y = y
        self.width = 100
        self.height = 5
        self.actions = [self.move_left,self.move_right,self.stay]

    # Define action functions
    def move_left(self):
        newx = self.x -  10
        if newx < 0:
            self.x = self.x
        else:
            self.x = newx
        
    def stay(self):
        
        pass
    def move_right(self):
        newx = self.x +  10
        if newx > WIDTH - self.width:
            self.x = self.x
        else:
            self.x = newx
        
    def update(self):
        pass
    # Define a function to check distance between player and ball
    def __sub__(self,other):
        return (abs(self.x - other.x) ,abs(self.y - other.y))
    
    
    def draw(self,screen):
        pass
        pygame.draw.rect(screen,(0,255,0),(self.x,self.y,self.width,self.height))
  

# Define class of ball
class Ball:
    def __init__(self, x, y, vx, vy, grid, player):
        # Initialize position
        self.x = x
        self.y = y
        
        # Radius of the ball
        self.radius = 8
        
        # Velocity components
        self.vx = vx
        self.vy = vy
        self.max_vx = abs(vx)
        self.max_vy = abs(vy)
        # Center of the ball (initial)
        self.center = (self.x, self.y)
        
        # Reference to the game grid/environment
        self.grid = grid
        
        # Reference to the player (likely used for collision or scoring)
        self.player = player

    def update(self):
        # Update position based on velocity
        self.x += self.vx
        self.y += self.vy
        
        # Reflect horizontally if ball hits left or right walls
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.vx = -self.vx
        
        # Reflect vertically if ball hits the top wall
        if self.y - self.radius < 0:
            self.vy = -self.vy 

    def __sub__(self, other):
        # Define subtraction behavior: returns distance in x and y
        return (abs(self.x - other.x), abs(self.y - other.y))

    def draw(self,screen):
        # Placeholder for drawing logic
        pass

        pygame.draw.circle(screen,(255,0,0),(self.x,self.y),self.radius)

        
class Handler:
    # Class-level lists to store objects
    things = []         # List to hold primary objects (likely game elements)
    other_things = []   # Unused in this snippet, could be for future use

    def update(self):
        # Update all objects in the 'things' list by calling their update method
        i = 0
        for i in range(len(self.things)):
            self.things[i].update()

    def draw(self):
        # Draw all objects in the 'things' list by calling their draw method
        i = 0
        for i in range(len(self.things)):
            self.things[i].draw()

    def add(self, a):
        # Add an object to the 'things' list
        self.things.append(a)

    def remove(self, a):
        # Remove an object from the 'things' list
        self.things.remove(a)


def draw_seperatelines(screen, screen_width, screen_height):
    # Draw horizontal lines across the upper half of the screen
    for i in range(int((screen_height // Tile.Height) / 2)):
        # Each line is spaced by Tile.Height
        pygame.draw.line(
            screen, 
            color="black", 
            start_pos=(0, i * Tile.Height), 
            end_pos=(screen_width, i * Tile.Height)
        )

    # Draw vertical lines across the full width of the screen
    for i in range(int(screen_width // Tile.Width)):
        # Each line is spaced by Tile.Width
        pygame.draw.line(
            screen, 
            color="black", 
            start_pos=(i * Tile.Width, 0), 
            end_pos=(i * Tile.Width, screen_height / 2)
        )


def bounce(ball,player):
    player_center_x = player.x + player.width/2
    distance_from_center = ball.x - player_center_x
    # if neg then left side, if pos then right side
    reduction_factor = (player.width/2)/ball.max_vx
    x_vel = distance_from_center/reduction_factor
    ball.vx = x_vel
    
    ball.vy = -abs(ball.vy)  

    

def main_show(q_table = None):
    pygame.init()
    video_width, video_height = WIDTH, HEIGHT
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")    
    out = cv2.VideoWriter("breakout_agent_gameplay.avi", fourcc, 30, (video_width, video_height))
    clock = pygame.time.Clock()
    # Flag to control the main loop
    running = True

    # List to hold tile positions (currently unused)
    tile_positions = []

    # Filename for loading the pre-trained Q-table (used for reinforcement learning)
    start_q_table = q_table

    # Load the Q-table from file
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

    # Calculate number of tiles horizontally and vertically
    GRID_WIDTH, GRID_HEIGHT = WIDTH // Tile.Width, HEIGHT // Tile.Height

    # Create a 2D grid of Tile objects for the upper half of the screen
    grid = [
        [Tile(x * Tile.Width, y * Tile.Height) for x in range(GRID_WIDTH)]
        for y in range(int(GRID_HEIGHT / 2))
    ]

    # Flatten the 2D grid into a single list for easier access
    flattened_grid = [item for sublist in grid for item in sublist]
    player_startx = random.randint(100, WIDTH - 100)
    ball_startx = random.randint(100, WIDTH - 100)
    ball_startvx = random.choice([-5,5])
    ball_startvy = random.choice([-5,5])
    # Initialize the player at position (200, 700)
    player = Player(player_startx, 700)

    # Initialize the ball at position (250, 500) with velocity (-5, -5)
    ball = Ball(ball_startx, 500, ball_startvx, ball_startvy, flattened_grid, player)

    # Create a handler object to manage game entities
    handler = Handler()

    # Append the 2D grid (as a list of lists) to the handler's things list
    handler.things.append(grid)

    # Add player and ball to the handler's other_things list
    handler.other_things.append(player)
    handler.other_things.append(ball)

    # Calculate delta time using the clock (divided for scaling)
    dt = clock.tick(60) / 10000000.0

    # Convert the 2D grid into a NumPy array for potential processing
    grid_array = np.array(grid)

    # Output the number of tiles in the grid
    print(grid_array.size)


    

        # Get total number of tiles (flattened size of grid)
    number_of_tiles = grid_array.size

    # Main game loop
    while running:

        # Handle events (e.g., quit button pressed)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Exit the loop when window is closed

        # Update all objects stored in handler.other_things (like player, ball, etc.)
        for things in handler.other_things:
            things.update()

        # Flag to check if collision occurred
        is_overlapping = False

        # Check collision between ball and player
        if ball_player_collision(
            (ball.x, ball.y),             # Ball position (center)
            ball.radius,                  # Ball radius
            (player.x, player.y,          # Player's position
             player.width, player.height) # Player's size
        ):
            is_overlapping = True

        # If collision detected, reflect the ball's vertical direction upward
        if is_overlapping:
            print("Overlapping")
            bounce(ball,player)  # Ensure ball always bounces upward after hitting the player

                
               # Collision flag for the current frame
        collided = False

        # Loop through the grid of tiles
        for i in range(len(grid)):
            if not collided:  # Only check until the first collision in this frame
                for j in range(len(grid[i])):

                    # Only check active tiles (non-zero)
                    if grid[i][j] != 0:

                        # Check if the ball collides with this tile
                        if rect_circle_collision((ball.x, ball.y), ball.radius, grid[i][j]):

                            # Handle collision based on direction of impact
                            if abs(ball.y - grid[i][j].y) < grid[i][j].height and ball.vy > 0:
                                # Ball hit tile from the top, bounce upward
                                ball.vy = -abs(ball.vy)

                            elif abs(ball.y - (grid[i][j].y + grid[i][j].height)) < grid[i][j].height and ball.vy < 0:
                                # Ball hit tile from the bottom, bounce downward
                                ball.vy = abs(ball.vy)

                            # Remove tile by setting it to 0 (destroyed)
                            handler.things[0][i][j] = 0
                            collided = True  # Prevent multiple bounces in one frame
                            number_of_tiles -= 1  # Decrease the count of remaining tiles
                            break  # Exit inner loop once a collision is handled

        # Calculate difference in x and y between ball and player
        diffx, diffy = ball - player

        # Determine which quadrant the ball is in relative to the player's center
        if ball.x < player.x + player.width / 2 and ball.y > player.y + player.height / 2:
            quadrant = 2  # Bottom-left
        elif ball.x > player.x + player.width / 2 and ball.y > player.y + player.height / 2:
            quadrant = 3  # Bottom-right
        elif ball.x > player.x + player.width / 2 and ball.y < player.y + player.height / 2:
            quadrant = 0  # Top-right
        elif ball.x < player.x + player.width / 2 and ball.y < player.y + player.height / 2:
            quadrant = 1  # Top-left

        # Create observation tuple used as a key in the Q-table
        obs = (int(diffx // 30), int(diffy // 30), quadrant)

        # Choose the best action based on the Q-table
        action = np.argmax(q_table[obs])

        # Execute the chosen action on the player
        player.actions[action]()

                        
                        
            
           
       
         
                

        
                # --- Draw Section ---

        # Flatten the 2D grid of tiles into a 1D list for easy iteration
        flattened = np.array(handler.things[0]).flatten().tolist()

        # Clear the screen by filling it with black
        screen.fill((0, 0, 0))

        # Draw each tile if it hasn’t been destroyed (i.e., not zero)
        for i in range(len(flattened)):
            if flattened[i] != 0:
                flattened[i].draw(screen)

        # Draw grid lines on top of tiles for better visual separation
        draw_seperatelines(screen, WIDTH, HEIGHT)

        # Draw all other objects managed by the handler (like the player and ball)
        for things in handler.other_things:
            things.draw(screen)

        # Update the full display surface to the screen
        pygame.display.flip()
        # Get the screen buffer and convert it for OpenCV
        # Capture the frame from Pygame
        frame = pygame.surfarray.array3d(pygame.display.get_surface())

        # Transpose from (width, height, 3) to (height, width, 3)
        frame = np.transpose(frame, (1, 0, 2))

        # Flip vertically to correct upside-down issue
        frame = np.flipud(frame)

        # Convert from RGB to BGR (for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write to video
        out.write(frame)

        # Cap the frame rate to 30 frames per second
        clock.tick(30)

    # Once the main loop ends, close the game window
    pygame.quit()
    out.release()


# Entry point for the game
# Uncomment the lines below to run the game directly from this script
# if __name__ == "__main__":
#     main()



def training(
        run_id,           # Identifier for this training run (used for saving logs, models, etc.)
        start_q_table,    # Filename of the starting Q-table (None for training from scratch)
        queue,       # to return a value back to the main process
        first_run=True
):
    # Grid resolution divisions (used to discretize the state space)
    divisions = 30

    # List to store rewards from each episode
    episode_rewards = []

    # Exploration rate for ε-greedy policy
    epsilon = 0.3

    # Number of discrete quadrants in the state space (to represent ball-player spatial relation)
    number_of_quadrant = 4

    # Total number of bricks or tiles in the game
    number_of_tiles = 60
    # Define the state space dimensions:
    # - X difference divided into WIDTH/divisions
    # - Y difference divided into HEIGHT/divisions
    # - Quadrant index
    state_space = (WIDTH // divisions + 1, HEIGHT // divisions + 1, number_of_quadrant)

    # Number of actions the agent can take (e.g., left, right, stay)
    action_space = 3

    

    # Initialize Q-table
    if start_q_table is None:
        # No existing Q-table, so create a new one with random initial values
        q_table = np.random.uniform(
            low=-5, high=0, size=(state_space + (action_space,))
        )
        print(q_table.shape)  # Print shape for verification

    else:
        # Load the pre-existing Q-table from file
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    try:
        # Initialize total reward for this episode
        episode_reward = 0
        max_tiles_broken = 0
        max_broken_reward = (0,-10000)  # Tuple to track max broken tiles and corresponding reward
        # Main training loop for 50,000 episodes
        for episode in range(50000):

            # Whether to render or show this episode (e.g., every 5000 episodes)
            show = False
            if episode % 5000 == 0:
                show = True
            if show:
                print(episode)  # Print the episode number as a checkpoint

            # Calculate tile grid dimensions based on screen size
            GRID_WIDTH, GRID_HEIGHT = WIDTH // Tile.Width, HEIGHT // Tile.Height

            # Reset episode rewards list (can be used to track per-step rewards)
            episode_rewards = []

            # Create a new grid of Tile objects for this episode
            grid = [
                [Tile(x * Tile.Width, y * Tile.Height) for x in range(GRID_WIDTH)]
                for y in range(int(GRID_HEIGHT / 2))
            ]

            # Flatten the grid into a 1D list for easier access
            flattened_grid = [item for sublist in grid for item in sublist]

            # Randomly choose a starting x-position for the player within screen bounds
            player_startx = random.randint(100, WIDTH - 100)

            # Randomly choose a starting x-position and velocity for the ball
            ball_startx = random.randint(100, WIDTH - 100)
            ball_startvx = random.choice([-5, 5])  # Horizontal direction
            ball_startvy = random.choice([-5, 5])  # Vertical direction

            # Initialize player and ball with randomized positions and velocity
            player = Player(player_startx, 650)
            ball = Ball(ball_startx, 500, ball_startvx, ball_startvy, flattened_grid, player)

            # Game control flags
            Game_complete = False  # True if the game is won (e.g., all tiles cleared)
            Game_over = False      # True if the game is lost (e.g., ball falls below screen)


            # Adjust epsilon (exploration rate) over time:
            # Higher in early training to encourage exploration
            if first_run:
                if episode < 25000:
                    epsilon = 0.8
                else:
                    epsilon = 0.3
            else: 
                epsilon = 0.1

            # Reset per-episode variables
            episode_reward = 0                 # Total reward for the episode
            number_of_tiles = 60              # Total active tiles at start
            number_of_tiles_broken = 0        # Counter for broken tiles
            
            limit = 40000
            k = 0
            # Run game simulation until it's either completed or lost
            while not Game_complete and not Game_over:  
                k+=1
                if k>limit:
                    print(f"Over the limit for process {run_id} on episode {episode}")
                    break

                # Compute distance between ball and player
                diffx, diffy = ball - player

                # Discretize the x and y distances to fit into Q-table state space
                BP_distance_x = diffx // divisions
                BP_distance_y = diffy // divisions

                # Determine quadrant: where the ball is in relation to the player center
                if ball.x < player.x + player.width / 2 and ball.y > player.y + player.height / 2:
                    quadrant = 2  # Ball is bottom-left of player
                elif ball.x > player.x + player.width / 2 and ball.y > player.y + player.height / 2:
                    quadrant = 3  # Ball is bottom-right of player
                elif ball.x > player.x + player.width / 2 and ball.y < player.y + player.height / 2:
                    quadrant = 0  # Ball is top-right of player
                elif ball.x < player.x + player.width / 2 and ball.y < player.y + player.height / 2:
                    quadrant = 1  # Ball is top-left of player

                # Reset collision flag for current frame
                is_overlapping = False

                # Construct observation tuple (state) for the Q-table lookup
                obs = (int(BP_distance_x), int(BP_distance_y), quadrant)

                # ε-greedy action selection:
                # With (1 - ε), choose the best known action
                # Otherwise, explore randomly
                if np.random.random() > epsilon:
                    action = np.argmax(q_table[obs])  # Exploit
                else:
                    action = np.random.randint(0, 3)  # Explore

                # Perform the chosen action (e.g., move left, right, or stay)
                player.actions[action]()

                # --- Update Game Entities ---
                player.update()  # Update player position based on chosen action
                ball.update()    # Update ball position based on current velocity

                # --- Collision: Ball and Player ---
                # Check if the ball is overlapping with the player paddle
                if ball_player_collision(
                    (ball.x, ball.y),               # Ball position
                    ball.radius,                    # Ball radius
                    (player.x, player.y,            # Player's position
                     player.width, player.height)   # Player's size
                ):
                    is_overlapping = True

                # If collision detected, bounce the ball upward and assign a bounce reward
                if is_overlapping:
                    bounce(ball,player)    # Ensure vertical velocity is upward
                    reward = BOUNCE_REWARD      # Assign reward for successful bounce

                # --- Collision: Ball and Tiles ---
                collided = False  # Flag to stop after the first tile collision

                # Check each tile in the grid for collision
                for i in range(len(grid)):
                    if not collided:  # Skip remaining tiles once a collision is processed
                        for j in range(len(grid[i])):

                            # Only process active tiles (non-zero)
                            if grid[i][j] != 0:

                                # Check collision between ball and tile
                                if rect_circle_collision((ball.x, ball.y), ball.radius, grid[i][j]):

                                    # Bounce behavior based on which side the ball hits
                                    if abs(ball.y - grid[i][j].y) < grid[i][j].height and ball.vy > 0:
                                        ball.vy = -abs(ball.vy)  # Hit from top, bounce up
                                    elif abs(ball.y - (grid[i][j].y + grid[i][j].height)) < grid[i][j].height and ball.vy < 0:
                                        ball.vy = abs(ball.vy)   # Hit from bottom, bounce down

                                    # Remove the tile (set to 0)
                                    grid[i][j] = 0

                                    # Mark collision processed for this frame
                                    collided = True

                                    # Update tile tracking variables
                                    number_of_tiles -= 1
                                    number_of_tiles_broken += 1
                                    
                                    
                                    break  # Exit inner loop after first hit

                # --- Check for Terminal Conditions ---

                # If all tiles are destroyed, the game is complete (win condition)
                if number_of_tiles == 0:
                    Game_complete = True
                    print(f"For Process {run_id} at {episode}: Game Complete")
                    print("Number of tiles broken: ", number_of_tiles_broken)

                # If the ball falls below the bottom of the screen, it's game over (loss condition)
                if ball.y > HEIGHT:
                    Game_over = True

                # --- Reward System ---

                # Default reward for each step (will be overridden if events occur)
                reward = 0

                # Add bounce reward if the ball hit the player's paddle
                if is_overlapping:
                    reward += BOUNCE_REWARD

                # Add tile destruction reward if the ball hit a tile
                if collided:
                    reward += TILE_DESTROY_REWARD

                # Add additional reward for clearing all tiles
                if Game_complete:
                    reward += GAME_COMPLETE_REWARD

                # Add penalty (or specific reward) for game over
                if Game_over:
                    reward += GAME_OVER_REWARD

                # If no special event happened, apply a small negative reward to discourage idleness
                if reward == 0:
                    reward = -MOVE_REWARD

                

                # --- State Transition and Q-Table Update ---

                # Calculate new distances between ball and player after the action
                BP_new_distance_x, BP_new_distance_y = ball - player

                # Discretize the new distances to determine the next state
                new_obs_ballx = BP_new_distance_x // divisions
                new_obs_bally = BP_new_distance_y // divisions

                # Determine the new quadrant relative to the player
                if ball.x < player.x + player.width / 2 and ball.y > player.y + player.height / 2:
                    quadrant = 2
                elif ball.x > player.x + player.width / 2 and ball.y > player.y + player.height / 2:
                    quadrant = 3
                elif ball.x > player.x + player.width / 2 and ball.y < player.y + player.height / 2:
                    quadrant = 0
                elif ball.x < player.x + player.width / 2 and ball.y < player.y + player.height / 2:
                    quadrant = 1

                # Form the new observation/state tuple
                new_obs = (int(new_obs_ballx), int(new_obs_bally), quadrant)

                # --- Q-Value Update Rule (Q-Learning) ---

                try:
                    # Find the maximum Q-value for the next state (best expected future reward)
                    max_future_Q = np.max(q_table[new_obs])
                except IndexError:
                    # If the new_obs is out of Q-table bounds, print debug info and break the episode
                    print("Index error: ", ball.x, ball.y, player.x, player.y, new_obs_ballx, new_obs_bally)
                    break

                # Get the current Q-value for the chosen action from the current state
                current_Q = q_table[obs][action]

                # Determine new Q-value based on the outcome
                if reward == GAME_OVER_REWARD:
                    # If the game is over, set Q directly to terminal penalty
                    new_q = GAME_OVER_REWARD
                elif reward == GAME_COMPLETE_REWARD:
                    # If the game is completed, set Q to terminal bonus
                    new_q = GAME_COMPLETE_REWARD
                
                else:
                    # Standard Q-learning update formula
                    new_q = (1 - LEARNING_RATE) * current_Q + LEARNING_RATE * (reward + DISCOUNT * max_future_Q)

                # Update the Q-table with the new value
                q_table[obs][action] = new_q

                # Accumulate total reward for this episode
                episode_reward += reward

            if number_of_tiles_broken > max_tiles_broken:
                
                max_tiles_broken = number_of_tiles_broken
                max_broken_reward = (number_of_tiles_broken,episode_reward)
            elif number_of_tiles_broken == max_tiles_broken and episode_reward > max_broken_reward[1]:
                max_broken_reward = (number_of_tiles_broken,episode_reward)

            # --- End of Episode Logging ---
            if show:
                print(f"For Process {run_id}:- Episode: {episode} Reward: {episode_reward} "
                      f"Number of tiles broken: {number_of_tiles_broken} Game Over: {Game_over}")

            # Track final reward of the episode
            episode_rewards.append(reward)

        # Update the maximum reward across all episodes
        max_rewards = max(episode_rewards)



        # --- Save Q-Table to Disk After Training ---
        with open(f"{run_id}.pickle", "wb") as f:
            pickle.dump(q_table, f)  # Save the trained Q-table with the run ID as filename

    # --- Graceful Handling of Early Termination (e.g., CTRL+C) ---
    except KeyboardInterrupt:
        with open(f"{run_id}.pickle", "wb") as f:
            print("Interrupted, saving q_tables")
            pickle.dump(q_table, f)  # Save progress before exiting
    # Store the run _id, the maximum number of tiles broken and the corresponding reward of that run
    print(run_id,max_broken_reward)
    queue.put((run_id, max_broken_reward))

    


def main_train(number_of_cycles,q_table = None):
    # Initialize maximum rewards (not shared across processes)
    
    first_run = True
    
    # In this we will runt the training process 3 times in parallel
    # Each time we will check which process has the best policy 
    # and use that policy to then train the next two processes
    for i in range(number_of_cycles):
        result_queue1 = Queue()
        result_queue2 = Queue()

        # Create two training processes using Python's multiprocessing
        # Each one starts from scratch (start_q_table = None) and uses a different run_id
        p1 = Process(target=partial(training, run_id=1, start_q_table=q_table, queue = result_queue1,first_run = first_run))
        p2 = Process(target=partial(training, run_id=2, start_q_table=q_table, queue = result_queue2,first_run = first_run))

        # Start both processes
        p1.start()
        p2.start()

        # Wait for both to finish
        p1.join()
        p2.join()

        first_run = False
        # Retrieve (run_id, (tiles_broken, other_metric)) tuples from each result queue
        run_id1, max_broken_reward1 = result_queue1.get()
        run_id2, max_broken_reward2 = result_queue2.get()

        # --- Compare results from the two processes to choose the better Q-table ---

        # If process 1 broke more tiles, choose its Q-table
        if max_broken_reward1[0] > max_broken_reward2[0]:
            q_table = f"q_table{run_id1}.pickle"
            os.remove(f"q_table{run_id2}.pickle")
            print(f"Process {run_id1} broke more tiles than Process {run_id2}")
            print(f"Process {run_id1} broke {max_broken_reward1[0]} and Process {run_id2} broke {max_broken_reward2[0]}")
        # If process 2 broke more tiles, choose its Q-table
        elif max_broken_reward1[0] < max_broken_reward2[0]:
            q_table = f"q_table{run_id2}.pickle"
            os.remove(f"q_table{run_id1}.pickle")
            print(f"Process {run_id2} broke more tiles than Process {run_id1}")
            print(f"Process {run_id1} broke {max_broken_reward1[0]} and Process {run_id2} broke {max_broken_reward2[0]}")
        # If both broke the same number of tiles, apply a tiebreaker
        elif max_broken_reward1[0] == max_broken_reward2[0]:
            print("Both broke the same number of tiles")
            print(f"Process {run_id1} broke {max_broken_reward1[0]} and Process {run_id2} broke {max_broken_reward2[0]}")
            # Use secondary metric (e.g., cumulative reward, efficiency, etc.)
            if max_broken_reward1[1] > max_broken_reward2[1]:
                q_table = f"q_table{run_id1}.pickle"
                os.remove(f"q_table{run_id2}.pickle")
                print(f"Process {run_id1} has higher reward than Process {run_id2}")
                print(f"Process {run_id1}'s reward {max_broken_reward1[1]} and Process {run_id2}'s reward {max_broken_reward2[1]}")
            else:
                q_table = f"q_table{run_id2}.pickle"
                os.remove(f"q_table{run_id1}.pickle")
                print(f"Process {run_id2} has higher reward than Process {run_id1}")                
                print(f"Process {run_id1}'s reward {max_broken_reward1[1]} and Process {run_id2}'s reward {max_broken_reward2[1]}")

    
    

if __name__ == "__main__":
    # On Windows, calling this ensures the child process can
    # safely import the main module.
    freeze_support()
    # Run this to train
    #main_train(number_of_cycles=7)

    # Run this to show the result 
    main_show("q_table2.pickle")

