#This script has FIXED Z-HEIGHT, and horizontal L3 

#phi is the base angle, 0 is when the limit switch is directly opposite home [-180, 180]
#theta1 is the angle between L1 and the horizontal
#theta2 is the angle between L1 and L2
#theta3 is the angle between L2 and L3

import numpy as np

# arm segment lengths
L1 = 240.0 
L2 = 290.0 
L3 = 150.0 

target_z = 53.1

def wrist_pos(target_x, target_y, target_z, L1, L2, L3):
    target_r = np.sqrt(target_x**2 + target_y**2)

    maxLen = L1+L2+L3
    if maxLen < target_r:
        raise ValueError(f"Target ({target_x}, {target_y}, {target_z}) is unreachable. Max reach is {maxLen:.2f} mm.")

    r = target_r - L3
    z = target_z
    return np.array([r, z])

def inverse_kinematics(target_x, target_y, target_z, L1, L2, L3):
    # base rotation
    phi = np.arctan2(target_y, target_x)

    # wrist position
    coords = wrist_pos(target_x, target_y, target_z, L1, L2, L3)

    # theta2
    cosTheta2 = (coords[0]**2 + coords[1]**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cosTheta2 = np.clip(cosTheta2, -1.0, 1.0)
    theta2 = np.arccos(cosTheta2)

    # theta1
    alpha = np.arctan2(coords[1], coords[0])

    beta = np.arcsin(L2 * (np.sin(theta2)/ np.sqrt(coords[0]**2 + coords[1]**2)))
    theta1 = alpha + beta

    # theta3 
    theta3 = 2*np.pi - (theta1 + (np.pi - theta2))

    # normalise angles
    theta1 = normalise_angle(theta1)
    theta2 = normalise_angle(theta2)
    theta3 = normalise_angle(theta3)

    return [phi, theta1, np.pi - theta2, theta3]

def normalise_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def positive_deg(theta):
    deg = np.degrees(theta)
    return deg % 360

target_x = int(input("Input target x-coord: "))
target_y = int(input("Input target y-coord: "))

angles = inverse_kinematics(target_x, target_y, target_z, L1, L2, L3)

print(f"Base rotation angle (phi): {positive_deg(angles[0]):.2f} degrees")
print(f"Joint B angle (theta1): {positive_deg(angles[1]):.2f} degrees")
print(f"Joint C angle (theta2): {positive_deg(angles[2]):.2f} degrees")
print(f"Joint D angle (theta3): {positive_deg(angles[3]):.2f} degrees")
