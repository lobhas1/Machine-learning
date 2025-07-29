In this file, I will summarise the breakout game reinforcement agent that I made. 
I have used a Q-learning method to train an RL model for playing the popular Atari game "Breakout".
I have also made the game by myself, and hence it might be a bit buggy.
One of the issues of the game is that the tiles are too large in width, which causes some weird bounces of the ball.
You can change this by decreasing the height of the tiles in the Tiles class in ![this file](BallPong_multiprocess.py).

The logic of training is as follows:
1) Two processes use the train() function to train two different q-tables.
   Each Q-table is trained for 50,000 episodes each cycle. There is a set limit of 40,000 steps for each episode, so it doesn't get stuck in a loop.
2) After a cycle, the maximum number of broken tiles for Process 1 and Process 2 is checked.
   The one with better performance is kept, and the other q-table is deleted.
3) The next cycle then uses the dominant Q-table and makes two more processes to train on that.
4) This keeps on repeating for the specified number of cycles.

The ![logfile](logfile.txt) keeps track of what has happened over the training time.
It tracks and prints the episode number, the reward for that episode, the number of tiles broken in that episode and whether the agent lost or not.
It also keeps track of any episode where the agent actually managed to break all the tiles in the game.
It can be noticed that as the number of tiles that are broken increases, the episode reward decreases, which is counterintuitive.
But this is because we get a negative reward for each step taken without getting any other reward, which demotivates the agent from rapidly moving.

As you can see from the video that the agent still oscillates very often, which means the move punishment should be higher.

The result of this training can be seen in the videos.
![breakoutagentgameplay1](breakoutagentgameplay1.mp4)
![breakoutagentgameplay2](breakoutagentgameplay2.mp4)

Problems with this code.
1) The first problem is very apparent; there needs to be more time spent on training the model.
2) Because the q_table does not track the number of tiles left, there is a state where the reward is randomly very huge, but the agent
   doesn't know that it is because the number of tiles is 0 and not because of the position of the ball and paddle.
   This causes it to take moves which do not make sense when that state (which is the distance between the ball and paddle and the relative position in the quadrant)
   is reached.
   To fix this, we need to add the number of tiles in the state space. But doing this can increase the training time exponentially, as reaching those states
   It's gonna take an extremely longer time.

These problems are definetly fixable, but my laptop wont be able to handle this, and hence I will settle for the agent that I have right now.
Maybe I will train it a bit more over the next few days but the second problem wont be fixed by that.

