import serial
import time
import numpy as np

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)

print("Connecting...")

init_lines = 0

while init_lines < 2:
    line=ser.readline().decode('utf-8').rstrip()
    if line:
        print(line)
        init_lines+=1

#This script includes z-height, so that the robot can reach objects at any height

#phi is the base angle, 0 is when the limit switch is directly opposite home [-180, 180]
#theta1 is the angle between L1 and the horizontal
#theta2 is the angle between L1 and L2
#theta3 is the angle between L2 and L3

# Arm segment lengths
L1 = 240.0 
L2 = 290.0 
L3 = 150.0 

theta1_initial = np.radians(60.0)
theta2_initial = np.radians(120.0)
theta3_initial = np.radians(120.0)


def forward_kinematics(theta1, theta2, theta3, L1, L2, L3):

    # End effector position in terms of joint angles

    # B is the joint B angle from the horizontal
    # C is the joint C angle from the horizontal
    # D is the joint D angle from the horizontal
    B = theta1
    C = B + theta2 - np.pi
    D = C + theta3 - np.pi

    r = (L1 * np.cos(B)) + (L2 * np.cos(C)) + (L3 * np.cos(D))
    z = (L1 * np.sin(B)) + (L2 * np.sin(C)) + (L3 * np.sin(D))
    return np.array([r, z])


def jacobian(theta1, theta2, theta3, L1, L2, L3):

    B = theta1
    C = B + theta2 - np.pi
    D = C + theta3 - np.pi

    # Differentiate forward kinematics position function wrt theta to find jacobian matrix
    J = np.array([
        [- (L1 * np.sin(B)) - (L2 * np.sin(C)) - (L3 * np.sin(D)), - (L2 * np.sin(C)) - (L3 * np.sin(D)), - (L3 * np.sin(D))], #dr/d(theta1), dr/d(theta2), dr/d(theta3)
        [(L1 * np.cos(B)) + (L2 * np.cos(C)) + (L3 * np.cos(D)), (L2 * np.cos(C)) + (L3 * np.cos(D)), (L3 * np.cos(D))] #dz/d(theta1), dz/d(theta2), dz/d(theta3)
    ])
    return J

def enforce_elbow_up(theta):
    #reflect the angle back into [0, pi]
    if theta > np.pi:
        theta = 2 * np.pi - theta
    return theta
    

def normalise_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def inverse_kinematics(target_x, target_y, target_z, theta1_initial, theta2_initial, theta3_initial, L1, L2, L3, tol=1e-4, max_iters=1000):
    
    target_r = np.sqrt(target_x**2 + target_y**2)
    maxLen = L1+L2+L3
    if maxLen < target_r:
        raise ValueError(f"Target ({target_x}, {target_y}, {target_z}) is unreachable. Max reach is {maxLen:.2f} mm.")

    theta1 = theta1_initial
    theta2 = theta2_initial
    theta3 = theta3_initial

    # Base rotation
    phi = np.arctan2(target_y, target_x)

    # Joints B, C, D
    for i in range(max_iters):
        current_position = forward_kinematics(theta1, theta2, theta3, L1, L2, L3)
        error = np.array([target_r, target_z]) - current_position
        
        if np.linalg.norm(error) < tol:
            break
        
        J = jacobian(theta1, theta2, theta3, L1, L2, L3)
        delta_theta = np.linalg.pinv(J).dot(error)

        theta1 += delta_theta[0]
        theta2 += delta_theta[1]
        theta3 += delta_theta[2]

        theta1 = normalise_angle(theta1)
        theta2 = normalise_angle(theta2)
        theta3 = normalise_angle(theta3)
        

        #theta2 = enforce_elbow_up(theta2)
        #theta3 = enforce_elbow_up(theta3)
    
        theta2 = max(theta2, np.pi / 2 + 0.01)
        theta3 = max(theta3, np.pi / 2 + 0.01)

    else:
        raise RuntimeError("Inverse kinematics did not converge within the maximum iterations.")

    return [np.degrees(phi), np.degrees(theta1), np.degrees(theta2), np.degrees(theta3)]


target_x = int(input("Input target x-coord: "))
target_y = int(input("Input target y-coord: "))
target_z = int(input("Input target z-coord: "))

angles = inverse_kinematics(target_x, target_y, target_z, theta1_initial, theta2_initial, theta3_initial, L1, L2, L3)

print(f"Joint A (phi): {angles[0]:.2f} degrees")
print(f"Joint B angle (theta1): {angles[1]:.2f} degrees")
print(f"Joint C angle (theta2): {angles[2]:.2f} degrees")
print(f"Joint D angle (theta3): {angles[3]:.2f} degrees")



angles_deg = [positive_deg(a) for a in angles]
comm = ",".join(map(str, angles_deg))

ser.write(f"{comm}\n".encode())

while True:
    line = ser.readline().decode('utf-8').rstrip()
    if line:
        print("Arduino: ", line)
