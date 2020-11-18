# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:02:51 2020

@author: DINGMAN
"""
import numpy as np
import unittest
import arm_gym as Arm_Env

AREA_REWARD, GOAL_REWARD, WRONG_INTERCEPT, DIST_REWARD = -1., 1., -0.8, 0
#case1
area_interest1_1 = {'x': 205., 'y': 205., 'width': 80.}
area_interest1_2 = {'x': 160., 'y': 250., 'width': 80.}
goal1 = {'x': 205., 'y': 210., 'width': 60.}
arm1 = {'length': [100, 100], 'angle': [np.pi/2, np.pi*(2-0.5-1/6)]}
arm1_2 = {'length': [100, 100], 'angle': [np.pi*1., -1.8]}
loc_base=[200.,200,]

#case2
area_interest2 = {'x': 310., 'y': 310., 'width': 80.}
goal2 = {'x': 315., 'y': 310., 'width': 60.}
arm2_1 = {'length': [100, 100], 'angle': [0.7, 0.3]}
arm2_goal = {'length': [100, 100], 'angle': [0.5, 0.5]}
arm2_3 = {'length': [100, 100], 'angle': [0.29, 1.03]}
arm2_4 = {'length': [100, 100], 'angle': [0.7, 1.5]}
valid_side2_left = 0
valid_side2_down = 2
invalid_side2_top = 1
invalid_side2_right = 3
loc_base=[200.,200,]

#case2
class ArmTests(unittest.TestCase):
    def test_valid_area_range(self):
        env1 = Arm_Env.ArmEnv()
        env1.reset(goal1, area_interest1_1, arm1['angle'])
        self.assertEqual(env1.valid, False)
        env1.reset(goal1, area_interest1_2, arm1['angle'])
        #env1.render()
        self.assertEqual(env1.valid, True)
               
    def test_valid_sides(self):
        env1 = Arm_Env.ArmEnv()
        env1.reset(goal2, area_interest1_2, arm1['angle'], valid_side2_left)
        self.assertEqual(env1.valid, True)
        env1.reset(goal2, area_interest1_2, arm1['angle'], invalid_side2_top)
        self.assertEqual(env1.valid, False)
        env1.reset(goal2, area_interest2, arm1['angle'], invalid_side2_top)
        # env1.render()
        self.assertEqual(env1.valid, False)        
        env1.reset(goal2, area_interest2, arm1['angle'], invalid_side2_right)
        self.assertEqual(env1.valid, False)
                
    def test_action_bad1(self):
        #touch the goal from wrong side
        env1 = Arm_Env.ArmEnv()
        env1.reset(goal2, area_interest2, arm2_1['angle'], valid_side2_down)
        s, r, done = env1.step([0.1,0.1])
        env1.render()
        self.assertEqual(done, False)
        self.assertAlmostEqual(r, WRONG_INTERCEPT)
        
    def test_action_goal(self):
        #action that reaches the goal
        env1 = Arm_Env.ArmEnv()
        env1.reset(goal2, area_interest2, arm2_goal['angle'], valid_side2_down)
        s, r, done = env1.step([0.1,0.1])
        #env1.render()
        self.assertEqual(done, True)
        self.assertAlmostEqual(r, GOAL_REWARD)
        
    def test_action_area_bad2(self):
        #reach the constrained area from the correct side
        env1 = Arm_Env.ArmEnv()
        env1.reset(goal2, area_interest2, arm2_3['angle'], valid_side2_down)
        s, r, done = env1.step([0.1,0.1])
        #env1.render()
        self.assertEqual(done, False)
        self.assertAlmostEqual(r, AREA_REWARD)
        
    def test_action_penetrate_bad3(self):
        #penetrate the area
        env1 = Arm_Env.ArmEnv()
        env1.reset(goal2, area_interest1_2, arm2_4['angle'], valid_side2_down)
        s, r, done = env1.step([0.1,0.1])
        #env1.render()
        self.assertEqual(done, False)
        self.assertAlmostEqual(r, WRONG_INTERCEPT)
    
    # def test_action_approaching(self):
        #optional, if the area and goal are too small, might need reward
    #     #not within the area, but approaching in the right direction
    #     #entitle to a small positive reward based on the distance to goal
    #     env1 = Arm_Env.ArmEnv()
    #     env1.reset(goal2, area_interest1_2, arm1_2['angle'], valid_side2_left)
    #     env1.step([-0.5,-0.5])
    #     s, r, done = env1.step([-0.5,-0.5])
    #     self.assertEqual(done, False)
    #     self.assertAlmostEqual(r, 0.023436037)
        
    # def test_action_away(self):
    #     #further away from the area, entitled to a small negative reward based 
    #     #on the distance to goal
    #     env1 = Arm_Env.ArmEnv()
    #     env1.reset(goal2, area_interest1_2, arm1_2['angle'], valid_side2_left)
    #     env1.step([0.5,0.5])
    #     s, r, done = env1.step([0.5,0.5])
    #     self.assertEqual(done, False)
    #     self.assertAlmostEqual(r, -0.027103630)
        
     #TODO: more test cases  

if __name__ == '__main__':
    unittest.main()
