This is part of a project I worked on to train a robotic arm to catch a ball, from start to finish. 

To do this, I had to generate some training data for the arm to learn from. The initial problem is that, to my knowledge, there are no open-source datasets on trajectory data, so I had to make my own. For that, I had to build a simple physics simulator. The trajectory data you see in the file is subject to both gravity and air-resistance.

After that, I trained the model and wrote this script to actually see if it learned how to move correctly. The python file is for showing both the movement of the ball through the air, and to see how the arm changes its orientation to get closer to it.
