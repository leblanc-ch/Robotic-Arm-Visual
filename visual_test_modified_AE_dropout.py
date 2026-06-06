import numpy as np
import tensorflow as tf
import math
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
import pyarrow.parquet as pq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from tensorflow.keras import regularizers
import tensorflow as tf

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Initialize the optimizer and model
eps_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)

input_val = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_val)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)

output_layer_1 = tf.keras.layers.Dense(1, activation='sigmoid')
output_layer_2 = tf.keras.layers.Dense(1, activation='sigmoid')
output_layer_3 = tf.keras.layers.Dense(1, activation='sigmoid')
output_layer_4 = tf.keras.layers.Dense(1, activation='sigmoid')

angle_out_1 = output_layer_1(x)
angle_out_2 = output_layer_2(x)
angle_out_3 = output_layer_3(x)
angle_out_4 = output_layer_4(x)

angle_estimate = tf.keras.Model(inputs = input_val, outputs = [angle_out_1, angle_out_2, angle_out_3, angle_out_4])
angle_estimate.compile(optimizer=eps_optimizer, loss='huber')

angle_estimate.load_weights('AE_dropout_L2_50810.h5')

theta1 = 0.0
theta2 = 0.25
theta3 = 1.5708
theta4 = 2.75

data = """0.390179 1.96935 -0.021415 false
0.375075 1.919374 0.017543 false
0.359967 1.869353 0.053788 false
0.344856 1.819287 0.087318 false
0.32974 1.769175 0.11813 false
0.314619 1.719017 0.146219 false
0.299495 1.668814 0.171585 false
0.284367 1.618564 0.194224 false
0.269234 1.568269 0.214134 false
0.254097 1.517928 0.231314 false
0.238956 1.467541 0.245761 false
0.223811 1.417108 0.257475 false
0.208662 1.366628 0.266455 false
0.193508 1.316103 0.272699 false
0.178351 1.26553 0.276208 false
0.163189 1.214912 0.276981 false
0.148023 1.164246 0.275018 false
0.132853 1.113534 0.270318 false
0.117678 1.062775 0.262882 false
0.1025 1.01197 0.252708 false
0.087317 0.961117 0.239795 false
0.07213 0.910218 0.224141 false
0.056939 0.859271 0.205746 false
0.041744 0.808278 0.184606 false
0.026544 0.757236 0.160721 false
0.011341 0.706148 0.134086 false
-0.003867 0.655012 0.1047 false
-0.019079 0.603829 0.072559 false
-0.034295 0.552598 0.03766 false
-0.049516 0.501319 0.0 true"""

# Split the data into rows
rows = data.split('\n')

# Initialize lists for storing x_pos, y_pos, and z_pos for the ball trajectory
x_pos = []
y_pos = []
z_pos = []

for row in rows:
    values = row.split()
    x_pos.append(float(values[0]))
    y_pos.append(float(values[1]))
    z_pos.append(float(values[2]))

x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create an empty scatter plot object for the animation
ball_position, = ax.plot([], [], [], 'bo', markersize=5)
arm_displacement, = ax.plot([], [], [], 'r--')



# ************************************* New Stuff
joint_one, = ax.plot([], [], [], 'k-', label='Joint 1')
joint_two, = ax.plot([], [], [], color='#6e6e67', linestyle='-', label='Joint 2')
joint_three, = ax.plot([], [], [], 'k-', label='Joint 3')
joint_four, = ax.plot([], [], [], color='#6e6e67', linestyle='-', label='Joint 4')

ax.scatter(x_pos, y_pos, z_pos, color='blue', label='Ball Position')

theta1 =  0.0000000
theta2 = 1.134464
theta3 = 1.7017
theta4 = 1.439901

ball_path = np.column_stack((x_pos,y_pos,z_pos))

def joint_one_out(theta_1):
    link_1 = .115
    rotation_1_to_2 = np.array([[math.cos(theta_1), -math.sin(theta_1), 0],
                        [math.sin(theta_1),  math.cos(theta_1), 0],
                        [0,0,1]
                        ])
    
    displace_1_to_2 = np.array([[0],[0],[link_1]])

    H_1_2 = np.concatenate((rotation_1_to_2, displace_1_to_2), 1)
    H_1_2 = np.concatenate((H_1_2, [[0,0,0,1]]), 0)

    return H_1_2[0:3,3]

def joint_two_out(theta_1, theta_2):
    link_1 = .115
    link_2 = .2285

    rotation_1_to_2 = np.array([[math.cos(theta_1), -math.sin(theta_1), 0],
                            [math.sin(theta_1),  math.cos(theta_1), 0],
                            [0,0,1]
                            ])

    rotation_2_to_3 = np.array([[1,0,0],
                            [0, math.cos(theta_2), -math.sin(theta_2)],
                            [0, math.sin(theta_2), math.cos(theta_2)]
                            ])
    
    displace_1_to_2 = np.array([[0],[0],[link_1]])
    
    H_1_2 = np.concatenate((rotation_1_to_2, displace_1_to_2), 1)
    H_1_2 = np.concatenate((H_1_2, [[0,0,0,1]]), 0)

    displace_2_to_3 = np.array([[0],
                    [link_2 * math.sin(theta_2)],
                    [link_2 * math.cos(theta_2)]
                    ])

    H_2_3 = np.concatenate((rotation_2_to_3, displace_2_to_3), 1)
    H_2_3 = np.concatenate((H_2_3, [[0,0,0,1]]), 0)

    return np.dot(H_1_2, H_2_3)[0:3,3]


def joint_three_out(theta_1, theta_2, theta_3):
    link_1 = .115
    link_2 = .2285
    link_3 = .2285

    rotation_1_to_2 = np.array([[math.cos(theta_1), -math.sin(theta_1), 0],
                            [math.sin(theta_1),  math.cos(theta_1), 0],
                            [0,0,1]
                            ])

    rotation_2_to_3 = np.array([[1,0,0],
                            [0, math.cos(theta_2), -math.sin(theta_2)],
                            [0, math.sin(theta_2), math.cos(theta_2)]
                            ])

    rotation_3_to_4 = np.array([[1,0,0],
                            [0, math.cos(theta_3), -math.sin(theta_3)],
                            [0, math.sin(theta_3), math.cos(theta_3)]
                            ])
    
    displace_1_to_2 = np.array([[0],[0],[link_1]])

    H_1_2 = np.concatenate((rotation_1_to_2, displace_1_to_2), 1)
    H_1_2 = np.concatenate((H_1_2, [[0,0,0,1]]), 0)

    displace_2_to_3 = np.array([[0],
                    [link_2 * math.sin(theta_2)],
                    [link_2 * math.cos(theta_2)]
                    ])

    H_2_3 = np.concatenate((rotation_2_to_3, displace_2_to_3), 1)
    H_2_3 = np.concatenate((H_2_3, [[0,0,0,1]]), 0)

    displace_3_to_4 = np.array([[0],
                    [link_3 * math.sin(theta_3)],
                    [link_3 * math.cos(theta_3)]
                    ])

    H_3_4 = np.concatenate((rotation_3_to_4, displace_3_to_4), 1)
    H_3_4 = np.concatenate((H_3_4, [[0,0,0,1]]), 0)

    return np.dot(np.dot(H_1_2, H_2_3), H_3_4)[0:3, 3]




def displacement_function(theta_1, theta_2, theta_3, theta_4):
    # the links represent the length of the joints in meters
    link_1 = .115
    link_2 = .2285
    link_3 = .2285
    link_4 = .24

    rotation_1_to_2 = np.array([[math.cos(theta_1), -math.sin(theta_1), 0],
                            [math.sin(theta_1),  math.cos(theta_1), 0],
                            [0,0,1]
                            ])

    rotation_2_to_3 = np.array([[1,0,0],
                            [0, math.cos(theta_2), -math.sin(theta_2)],
                            [0, math.sin(theta_2), math.cos(theta_2)]
                            ])

    rotation_3_to_4 = np.array([[1,0,0],
                            [0, math.cos(theta_3), -math.sin(theta_3)],
                            [0, math.sin(theta_3), math.cos(theta_3)]
                            ])

    rotation_4_to_5 = np.array([[1,0,0],
                            [0, math.cos(theta_4), -math.sin(theta_4)],
                            [0, math.sin(theta_4), math.cos(theta_4)]
                            ])

    displace_1_to_2 = np.array([[0],[0],[link_1]])

    H_1_2 = np.concatenate((rotation_1_to_2, displace_1_to_2), 1)
    H_1_2 = np.concatenate((H_1_2, [[0,0,0,1]]), 0)

    displace_2_to_3 = np.array([[0],
                    [link_2 * math.sin(theta_2)],
                    [link_2 * math.cos(theta_2)]
                    ])

    H_2_3 = np.concatenate((rotation_2_to_3, displace_2_to_3), 1)
    H_2_3 = np.concatenate((H_2_3, [[0,0,0,1]]), 0)

    displace_3_to_4 = np.array([[0],
                    [link_3 * math.sin(theta_3)],
                    [link_3 * math.cos(theta_3)]
                    ])

    H_3_4 = np.concatenate((rotation_3_to_4, displace_3_to_4), 1)
    H_3_4 = np.concatenate((H_3_4, [[0,0,0,1]]), 0)

    displace_4_to_5 = np.array([[0],
                    [link_4 * math.sin(theta_4)],
                    [link_4 * math.cos(theta_4)]
                    ])

    H_4_5 = np.concatenate((rotation_4_to_5, displace_4_to_5), 1)
    H_4_5 = np.concatenate((H_4_5, [[0,0,0,1]]), 0)

    return np.dot(np.dot(np.dot(H_1_2, H_2_3), H_3_4), H_4_5)[0:3,3]


# Use the actor to predict actions for the new states
arm_x = []
arm_y = []
arm_z = []

joint_one_x = []
joint_one_y = []
joint_one_z = []

joint_two_x = []
joint_two_y = []
joint_two_z = []

joint_three_x = []
joint_three_y = []
joint_three_z = []

joint_four_x = []
joint_four_y = []
joint_four_z = []

prediction_error = ball_path[0] - displacement_function(theta1, theta2, theta3, theta4)

for ball_index in range(0, len(x_pos)):
    # Current State is the ball position (x, y, z) and the current arm position (x, y, z)
    # curr_state = np.concatenate([ball_path[ball_index].flatten(), displacement_function(theta1, theta2, theta3, theta4), prediction_error])
    
    
    
    # action = np.array(actor(np.expand_dims(curr_state, axis=0)))

    # theta1 += action[0,0]
    # theta2 += action[0,1] 
    # theta3 += action[0,2]
    # theta4 += action[0,3]

    ball_location = np.array([ball_path[ball_index, 0:3]]).flatten()
    ball_actual = tf.convert_to_tensor(np.expand_dims(ball_location, axis=0), dtype = tf.float32)

    unscaled_angle_1, unscaled_angle_2, unscaled_angle_3, unscaled_angle_4 = angle_estimate(ball_actual)
    scaled_action_1 = tf.squeeze((unscaled_angle_1 * (2.0944)) - 1.0472)
    scaled_action_2 = tf.squeeze((unscaled_angle_2 * (3.0)))
    scaled_action_3 = tf.squeeze((unscaled_angle_3 * (3.0)))
    scaled_action_4 = tf.squeeze((unscaled_angle_4 * (3.0)))

    arm_actual = displacement_function(scaled_action_1, scaled_action_2, scaled_action_3, scaled_action_4)

    theta1 = scaled_action_1
    theta2 = scaled_action_2
    theta3 = scaled_action_3
    theta4 = scaled_action_4

    j1_x, j1_y, j1_z = joint_one_out(theta1)
    joint_one_x = np.append(joint_one_x , j1_x)
    joint_one_y = np.append(joint_one_y , j1_y)
    joint_one_z = np.append(joint_one_z , j1_z)

    j2_x, j2_y, j2_z = joint_two_out(theta1, theta2)
    joint_two_x = np.append(joint_two_x , j2_x)
    joint_two_y = np.append(joint_two_y , j2_y)
    joint_two_z = np.append(joint_two_z , j2_z)

    j3_x, j3_y, j3_z = joint_three_out(theta1, theta2, theta3)
    joint_three_x = np.append(joint_three_x , j3_x)
    joint_three_y = np.append(joint_three_y , j3_y)
    joint_three_z = np.append(joint_three_z , j3_z)

    x,y,z = displacement_function(theta1, theta2, theta3, theta4)
    arm_x = np.append(arm_x, x)
    arm_y = np.append(arm_y, y)
    arm_z = np.append(arm_z, z)

arm_x = np.array(arm_x)
arm_y = np.array(arm_y)
arm_z = np.array(arm_z)

joint_one_x = np.array(joint_one_x)
joint_one_y = np.array(joint_one_y)
joint_one_z = np.array(joint_one_z)

joint_two_x = np.array(joint_two_x)
joint_two_y = np.array(joint_two_y)
joint_two_z = np.array(joint_two_z)

joint_three_x = np.array(joint_three_x)
joint_three_y = np.array(joint_three_y)
joint_three_z = np.array(joint_three_z)

def update(frame):
    if frame < len(arm_x):
        # Update ball position
        ball_position.set_data(x_pos[:frame+1], y_pos[:frame+1])
        ball_position.set_3d_properties(z_pos[:frame+1])

        # Update the arm and joints
        arm_displacement.set_data([0.0, arm_x[frame]], [0.0, arm_y[frame]])
        arm_displacement.set_3d_properties([0.0, arm_z[frame]])

        joint_one.set_data([0.0, joint_one_x[frame]], [0.0, joint_one_y[frame]])
        joint_one.set_3d_properties([0.0, joint_one_z[frame]])

        joint_two.set_data([joint_one_x[frame], joint_two_x[frame]], [joint_one_y[frame], joint_two_y[frame]])
        joint_two.set_3d_properties([joint_one_z[frame], joint_two_z[frame]])

        joint_three.set_data([joint_two_x[frame], joint_three_x[frame]], [joint_two_y[frame], joint_three_y[frame]])
        joint_three.set_3d_properties([joint_two_z[frame], joint_three_z[frame]])

        joint_four.set_data([joint_three_x[frame], arm_x[frame]], [joint_three_y[frame], arm_y[frame]])
        joint_four.set_3d_properties([joint_three_z[frame], arm_z[frame]])

        # Adjust the limits of the axes
        ax.set_xlim(min(np.min(x_pos), np.min(arm_x)), max(np.max(x_pos), np.max(arm_x)))
        ax.set_ylim(min(np.min(y_pos), np.min(arm_y)), max(np.max(y_pos), np.max(arm_y)))
        ax.set_zlim(min(np.min(z_pos), np.min(arm_z)), max(np.max(z_pos), np.max(arm_z)))

    return ball_position, arm_displacement, joint_one, joint_two, joint_three, joint_four

# Create the animation
ani = FuncAnimation(fig, update, frames=len(x_pos), blit=True, interval=100)

# Show the plot
plt.show()