from six.moves import cPickle as pickle #for performance
import numpy as np 
import math


def load_dict( filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

result = load_dict("Town05.pkl")



def save_dict( di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
        
        
def get_yaw(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    yaw_vector = (point2 - point1)
    vector = [1, 0]
    unit_vector_1 = yaw_vector / np.linalg.norm(yaw_vector)
    unit_vector_2 = vector / np.linalg.norm(vector)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    if yaw_vector[1] < 0:
        yaw = -math.degrees(angle)
    else:
        yaw = math.degrees(angle)
    return yaw
    

obstacle_data = load_dict(f"./Town05.pkl")


scenario_info = obstacle_data[0]
obstacle_route_list = scenario_info[2] 
obstacle_info = scenario_info[1]

# get ego transform 

# get first two point 
point1 = obstacle_route_list[0][0]
point2 = obstacle_route_list[0][1]
yaw = get_yaw(point1, point2)

print(point1)
print(yaw)



print(len(obstacle_route_list))


print(len(obstacle_route_list[1:]))

print(obstacle_info)






# get ego transform 

# get other vehicle transform 

