# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:11:18 2020

@author: DINGMAN
"""
import gym
from gym import spaces
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

action_limit = [-1., 1.]
length=[100,100]
dt = 0.15 #update rate for chosen action
AREA_WIDTH = 80.
GOAL_WIDTH = 60.
#GOAL_COUNTS = 1
AREA_REWARD, GOAL_REWARD, WRONG_INTERCEPT, DIST_REWARD = -1., 1., -0.8, 0

class State(object):
        
    def __init__(self, angle=[np.pi/3, np.pi/3], goal=None, area_interest=None, base=[200.,200.], valid_side=None):
        '''
        Parameters
        ----------
        length : list
            length of the each segment of the arm.
        angle : list
            Angle of the each joint
        base : tuple
            location of the base joint
        '''
        self.arm = {}
        self.arm['length'] = np.array(length)
        self.arm['angle'] = np.array(angle)
        self.loc_base = np.array(base)
        self.prev_loc_tip = self.get_location()
        self.distances = [0] 
        self.goal = goal
        self.area_interest = area_interest
        self.valid_side = valid_side
        
    def get_arm(self):
        '''
        get the information about the arm, length of the arm and the 
        angle

        Returns
        -------
        arm: dict
        '''
        return self.arm
    
    def get_location(self):
        '''
        get locations of the two joints and tip of the arm

        Returns
        ------- 
        Location of base joint: 1x2 array
        location of middle joint: 1x2 array
        location of arm tip: 1x2 array
        '''
        #len_1 refer to the base joint, len_2 middle joint, same as angle_1, angle_2
        len_1, len_2 = self.get_arm()['length']
        angle_1, angle_2 = self.get_arm()['angle']
    
        #location of middle joint
        loc_middle = self.loc_base + len_1 * np.array([np.cos(angle_1), np.sin(angle_1)])
        loc_tip = loc_middle + len_2 * np.array([np.cos(angle_1+angle_2), np.sin(angle_1+angle_2)])
        
        return loc_middle, loc_tip
    
    def distance_to_goal(self, goal_x, goal_y, goal_w, loc_tip):
        '''
        calculate the distance from the current position to goal
        position of the goal is defined as mid point of the valid side.

        Returns
        -------
        distance: float
        '''
        if self.valid_side == 0:
            goal_mid = [goal_x, goal_y+goal_w/2]
        if self.valid_side == 1:
            goal_mid = [goal_x+goal_w/2, goal_y+goal_w]
        if self.valid_side == 2:
            goal_mid = [goal_x+goal_w/2, goal_y]
        if self.valid_side == 3:
            goal_mid = [goal_x+goal_w, goal_y+goal_w/2]
        distance = np.sqrt(np.sum(np.square(goal_mid - loc_tip)))
        
        return distance
    
    def status_update(self):
        '''
        check if the arm has reached the goal or area of interest
        Returns
        -------
        reach_goal : bool
        reach_area : bool
        distance: shortest distance from current position to goal    
        '''
        epsilon = 10**-6
        reach_goal = False
        reach_area = False
        wrong_intercept = False
        distance = None
        loc_tip = self.get_location()[-1]
        area_x, area_y, area_w = [self.area_interest.get(key) for key in ['x', 'y', 'width']]
        goal_x, goal_y, goal_w = [self.goal.get(key) for key in ['x', 'y', 'width']]
        #check if any part of the upper arm intercept with the area
        intercepts = self.check_intersection()
        #print(intercepts)
        if np.any(np.array(intercepts)==True):            
        #only access via valid side
            if intercepts.pop(self.valid_side)==True and np.all(np.array(intercepts)==False):
            #check if the tip has reached the goal
                if (goal_x-epsilon <= loc_tip[0] <= goal_x + goal_w+epsilon)and \
                    (goal_y-epsilon <= loc_tip[1] <= goal_y + goal_w+epsilon):
                    reach_goal = True                    
            #check if the upper arm intercept with area of interest (should be penalized)
                elif (area_x-epsilon <= loc_tip[0] <= area_x + area_w + epsilon)and \
                    (area_y-epsilon <= loc_tip[1] <= area_y + area_w + epsilon):
                    reach_area = True
                else:
                    wrong_intercept = True
            else:
                wrong_intercept = True
                
        if reach_goal==False and reach_area==False:
            #no intersection with the area, calculate distance to the goal
            distance = self.distance_to_goal(goal_x, goal_y, goal_w, loc_tip)
            if self.distances[-1] != 0:
                self.dist_diff = (self.distances[-1] - distance)/self.distances[-1]
            else:
                self.dist_diff = 0
            self.distances.append(distance)
            
        return reach_goal, reach_area, wrong_intercept

    def check_intersection(self):
        '''
        check if the upper arm intersects with the correct side of the area of interest

        Returns
        -------
        intercept boolean
        '''        
        loc_tip = self.get_location()[-1]
        loc_middle = self.get_location()[0]
        x1, y1 = loc_middle       
        x2, y2 = loc_tip
        x_BL, y_BL, area_w = [self.area_interest.get(key) for key in ['x', 'y', 'width']]
        x_TR, y_TR = x_BL + area_w, y_BL + area_w
        corners = [[x_BL,y_BL], [x_BL+area_w, y_BL], [x_BL, y_BL+area_w], [x_TR, y_TR]]
        segments = [corners[0]+corners[2], corners[2]+corners[3], corners[0]+corners[1], corners[1]+corners[3]]
        intercepts = []
        for segment in segments:
            intercepts.append(self.intersect(loc_middle, loc_tip, segment[:2], segment[2:]))            
        return intercepts
    
    def intersect(self, a,b,c,d):
        '''
        check if two line segments intersects

        Returns
        -------
        intercept: boolean
        '''
        def direction(a, b, c):
            #colinear:0, anticlockwise:2, clockwise:1
            direction = (b[1]-a[1])*(c[0]-b[0])-(b[0]-a[0])*(c[1]-b[1])
            if direction == 0:
                return 0
            elif direction < 0:
                return 2
            return 1
    
        dir1 = direction(a, b, c)
        dir2 = direction(a, b, d)
        dir3 = direction(c, d, a)
        dir4 = direction(c, d, b)
        
        if(dir1 != dir2 and dir3 != dir4):
          return True             
        return False;         
    
    def valid_action(self, action):
        '''
        make sure the action is within 2*pi range.
        '''
        action = np.clip(action, *action_limit)
        return action
    
class Viewer():
    def __init__(self, arm, goal, area_interest):
        self.loc_base = np.array([200.,200.])
        self.goal = goal
        self.area_interest = area_interest
        self.image = self.plot(arm)
        
    def get_location(self, arm):
        self.arm_lengths = np.array(arm['length'])
        self.arm_angles = np.array(arm['angle'])
        len_1, len_2 = self.arm_lengths
        angle_1, angle_2 = self.arm_angles
        loc_middle = self.loc_base + len_1 * np.array([np.cos(angle_1), np.sin(angle_1)])
        loc_tip = loc_middle + len_2 * np.array([np.cos(angle_1+angle_2), np.sin(angle_1+angle_2)])
        return loc_middle, loc_tip
        
    def plot(self, arm, show=True):
        loc_middle, loc_tip = self.get_location(arm)
        lower_arm = plt.Line2D((self.loc_base[0], loc_middle[0]), (self.loc_base[1], loc_middle[1]),
                              lw=2, marker='.', markersize=15, markerfacecolor='r', markeredgecolor='r',
                              alpha=0.6)
        upper_arm = plt.Line2D((loc_middle[0], loc_tip[0]), (loc_middle[1], loc_tip[1]),
                              lw=2, marker='.', markersize=15, markeredgecolor='r', alpha=0.6)
        
        area_x, area_y, area_w = [self.area_interest.get(key) for key in ['x', 'y', 'width']]
        goal_x, goal_y, goal_w = [self.goal.get(key) for key in ['x', 'y', 'width']]
        area = Rectangle((area_x, area_y), area_w, area_w, color='yellow')
        goal_rect = Rectangle((goal_x, goal_y), goal_w, goal_w, color='green')
        fig, ax = plt.subplots()
        current_axis = plt.gca()
        current_axis.add_patch(area)
        current_axis.add_patch(goal_rect)       
        current_axis.add_line(lower_arm)
        current_axis.add_line(upper_arm)
        plt.xlim([0, 400])
        plt.ylim([0, 400])
        #plt.grid()
        if show == False:
            plt.close();
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)        
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
        return image
        
    def render(self, arm, show):            
        image = self.plot(arm, show)
        return image
        
class ArmEnv(gym.Env):
    def __init__(self, angle=[0.,0.], goal=None, area_interest=None, valid_side=None, loc_base=[200.,200,]):
        '''
        Parameters
        ----------
        loc_base : list
            location of base joint. The default is [200.,200,].
        goal : dict
            location and width of the goal
        area_interest : dict
            location and width of the area of interest
        '''
        self.loc_base = loc_base
        self.done = False
        self.viewer = None
        self.goal_count = 0
        self.action_space = spaces.Tuple([spaces.Discrete(1), spaces.Discrete(2)])
                
    def _setWorld(self, goal, area_interest, angle, valid_side=None):
        
        def valid_sides(area_interest, loc_base=[200.,200,]):
            '''
            check which side of the area of interest to place the goal
            make sure the robot arm can indeed reach the goal.
            Parameters
            ----------
            area_interest : dict
            loc_base : list
    
            Returns
            -------
            valid_sides: list
    
            '''
            x, y, w = [area_interest.get(key) for key in ['x', 'y', 'width']]
            valid_sides = [0,1,2,3] #0-left side, 1-upper, 2-lower, 3-right
            #upper arm length 100
            if x < 100 :
                #furthest side cannot be reached: left side
                valid_sides.remove(0)
            if (x > loc_base[1] + 100 - w):
                #right side cannot be reached
                valid_sides.remove(3)
            if y < 100:
                #furthest side cannot be reached: right side
                valid_sides.remove(2)
            if y > loc_base[1] + 100 - w:
                #upper side cannot be reached
                valid_sides.remove(1)        
            return valid_sides
    
        def set_goal(side, area_x, area_y, goal):
            '''       
            Place the goal on the chosen side 
            
            Parameters
            ----------
            side : int
                chosen side of the area of interest
            area_x, area_y: int
                x, y coordinates of area of interest
    
            Returns
            -------
            goal: dict
            '''
            goal = {}
            goal['width'] = GOAL_WIDTH
            if side == 0:
                # |
                goal['x'] = area_x
                goal['y'] = area_y + (AREA_WIDTH - GOAL_WIDTH) / 2                
            elif side == 1:
                #-
                goal['x'] = area_x + (AREA_WIDTH - GOAL_WIDTH) / 2
                goal['y'] = area_y + AREA_WIDTH - GOAL_WIDTH                
            elif side ==2:
                #_
                goal['x'] = area_x + (AREA_WIDTH - GOAL_WIDTH) / 2
                goal['y'] = area_y                
            else:
                # |
                goal['x'] = area_x + AREA_WIDTH - GOAL_WIDTH 
                goal['y'] = area_y + (AREA_WIDTH - GOAL_WIDTH) / 2
            
            return goal
        
        
        valid = True
        #the maximum range is (20<x, y<340)
        valid_range = list(set(np.arange(20., 400.-AREA_WIDTH)) 
                           .difference(set(np.arange(20, self.loc_base[0]+0.5*AREA_WIDTH))) 
                           .union(set(np.arange(20, self.loc_base[0]-1.5*AREA_WIDTH))))
        if area_interest is None:
            #randomly select both area of interest and goal            
            #randomly generate coordinates for area of interest
            area_interest = {}
            area_interest['width'] = AREA_WIDTH
            #set the area_interest not too close to the location of base
            area_x, area_y = np.random.choice(valid_range, 2)   
            area_interest['x'], area_interest['y'] = area_x, area_y  
            #randomly generate a goal inside the area of interest, accessible from one side  
        if area_interest is not None:
            #no need to generate area_interest
            #assert(type(area_interest['x']) == float and type(area_interest['y']) == float)
            area_x, area_y, area_w = [area_interest.get(key) for key in ['x', 'y', 'width']]
            if area_x not in valid_range and area_y not in valid_range:
                valid = False
                area_x, area_y = np.random.choice(valid_range, 2)   
                area_interest['x'], area_interest['y'] = area_x, area_y 
                
        valid_sides = valid_sides(area_interest, self.loc_base)
        if not(valid_side is None):
            if valid_side not in valid_sides:
                valid = False
                #raise ValueError('Not a valid side')            
        else:
            valid_side = np.random.choice(valid_sides,1)[0]
        self.goal = set_goal(valid_side, area_x, area_y, goal)
        self.area_interest = area_interest
        self.valid_side = valid_side
        if angle == [0.,0.]:            
            self.angle = np.random.rand(2) * 2 * np.pi
        else:
            self.angle = angle
        self.world = State(self.angle, self.goal, self.area_interest, self.loc_base, self.valid_side)
        #Todo: check if the model works without area_interest, i.e. area_interest=None, goal!=None
        return valid
    
    def reset(self, goal=None, area_interest=None, angle=[0,0], valid_side=None):
        self.done = False
        #randomly the initial angle the arm as starting point        
        # Initialize data structures
        self.valid = self._setWorld(goal, area_interest, angle, valid_side)
        if self.viewer is not None:
            self.viewer = None        
        return self.world.arm['angle']
    
    def step(self, action):
        action = self.world.valid_action(action)
        self.world.arm['angle'] += action * dt
        self.world.arm['angle'] %= np.pi * 2    # normalization
        reach_goal, reach_area, wrong_intercept = self.world.status_update()
        #print(self.world.status_update())

        if reach_goal == True:
            reward = GOAL_REWARD            
            self.done = True
            
        if reach_area == True:
            reward = AREA_REWARD
            
        if wrong_intercept == True:
            reward = WRONG_INTERCEPT
            
        if reach_goal == False and reach_area == False and wrong_intercept == False:
            reward = self.world.dist_diff * DIST_REWARD
        
        return self.world.arm['angle'], reward, self.done
     
    def render(self, show=True):
        if self.viewer is None:
            self.viewer = Viewer(self.world.arm, self.goal, self.area_interest)
        image = self.viewer.render(self.world.arm, show)
        return image     
        
    def sample_action(self):
        #sample uniformly
        return np.random.rand(2)-0.5
            
if __name__ == '__main__':
    env = ArmEnv()
    env.reset()
    for i in range(10):
        s,r,done = env.step(env.sample_action())
        print(s,r,done)
        env.render()