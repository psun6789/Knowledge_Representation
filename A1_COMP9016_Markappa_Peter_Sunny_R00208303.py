# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:21:08 2021

@author: Peter Sunny Shanthveer Markappa

Final Submitting on 26-10-2021
"""


'''
    Reference Websites are here
    for this assignment i have referred this page
    https://stackoverflow.com/questions/56697930/how-can-i-change-this-to-use-a-q-table-for-reinforcement-learning
    
'''


import os
import sys
import inspect
from agents import *
from search import *
from logic import *
import random
import math
import numpy as np_array
import time

cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)




class DroneBattery(Thing):
    pass


class DronePickItem(Thing):
    pass


class DroneObstacle(Thing):
    pass





'''The is the function where Environment is prepared for the drone '''
class Air_Coordinates_Map(GraphicEnvironment):
    '''
        I have copied this percept code from agents.ipynb
        https://github.com/aimacode/aima-python/blob/master/agents.ipynb
    '''
    def percept(self, agent):
        ''' This will return the things that are in our Drone current co-ordinate '''
        drone_thing = self.list_things_at(agent.location)
        return drone_thing

    '''
        in this part of section i have done reference from below mention page
        https://github.com/aimacode/aima-python/blob/master/agents.ipynb
    '''
    def execute_action(self, agent, action):
        ''' This will change state of Drone'''
        if action == "move_forward_coordinate":
            if agent.direction == "D":
                print('{}Drone has decided to move back because of move back instruction {}'.format(str(agent)[1:-1], agent.location))
            else:
                print('{} Drone decided to move {} at coordinate: {}'.format(str(agent)[1:-1], action, agent.location))
            if agent.move_forward_coordinate():
                if agent.direction == "D":
                    print('{} Drone moved behind with coordinate: {}'.format(str(agent)[1:-1], agent.location))
                else:
                    print('{} Drone decided to move {} its present coordinate is: {}'.format(str(agent)[1:-1], action, agent.location))
            else:
                print("{} Drone cannot move forward".format(agent))


        elif action == 'pick_item':
            print('{} Drone decided to stop  {} with coordinate location: {}'.format(str(agent)[1:-1], action, agent.location))
            if agent.pick_item():
                print('{} has picked the Package and its new coordinate is: {}'.format(str(agent)[1:-1], agent.location))
            else:
                print("{}cannot pick the package of some obstacles ".format(agent))

        elif action == 'move_to_left_coordinate':
            print('{} detectd obstacle so agent turning turn  {} with coordinate location: {}'.format(str(agent)[1:-1], action, agent.location))
            if agent.move_to_left_coordinate():
                print('{} Drone moved to Left side from obstacles coordinate and its new coordinate is: {}'.format(str(agent)[1:-1], agent.location))
            else:
                print("{} Drone cannot move right because of some obstacles".format(agent))

        # Some times drone has to stop in the air because of Signal problem
        elif action == "stop_signal":
            print('{}Drone decided to stop {} for some time {} '.format(str(agent)[1:-1], action, agent.stop_timing))
            if agent.stop_signal():
                print("{} Drone stopped for {} minute for Battery ".format(str(agent)[1:-1], agent.stop_timing))

        # Checking for Drone Destination
        elif action == 'drone_destination':
            print(" Drone has reached its destination with co-ordinate {} equal to destination_coordinate {} ".format(agent.location,agent.delivery_location))
            return

    '''
        I have copied this percept code from agents.ipynb and changed some variables
        https://github.com/aimacode/aima-python/blob/master/agents.ipynb
    '''
    def is_done(self):
        if (self.agents[0].location == self.agents[0].delivery_location).all() or (self.agents[0].drone_battery_level <= 0):
            return True
        return False







''' this class explains about agent which is drone is updates its current state of the drone'''

class DroneDeliveryAgent(Agent):
    location = np_array.array([0, 0])
    delivery_location = np_array.array([5, 5])
    stop_timing = 1
    drone_battery_level = 200
    drone_achievement = 0
    direction = "U"
    stage = {}

    def update_state(self, action):
        self.stage[tuple(self.location)] = action

    def check_current_state(self):
        if (self.location == self.delivery_location).all():
            return True
        return False

    '''
    I have referred below content to build this function
    https://stackoverflow.com/questions/56697930/how-can-i-change-this-to-use-a-q-table-for-reinforcement-learning
    '''
    def move_forward_coordinate(self):
        if self.direction == "U":
            self.drone_achievement += 1
            self.drone_battery_level -= 10
            self.location[0] += 1
            return True

        elif self.direction == "R":
            self.drone_achievement += 1
            self.drone_battery_level -= 10
            self.location[1] += 1
            return True

        elif self.direction == "L":
            self.drone_achievement += 1
            self.drone_battery_level -= 10
            self.location[1] -= 1
            return True

        elif self.direction == "D":
            self.drone_achievement -= 1
            self.location[0] -= 1
            return True



    def pick_item(self):
        if self.direction == "U":
            self.drone_achievement += 1
            self.drone_battery_level -= 10
            self.location[1] += 1
            self.direction = "R"
            return True

        elif self.direction == "L":
            self.drone_battery_level -= 10
            self.drone_achievement += 1
            self.location[0] += 1
            self.direction = "U"
            return True

        elif self.direction == "R":
            self.drone_achievement -= 1
            self.location[0] -= 1
            self.direction = "D"
            return True

        elif self.direction == "D":
            self.drone_achievement += 1
            self.drone_battery_level -= 10
            self.location[1] -= 1
            self.direction = "L"
            return True

    def move_to_left_coordinate(self):
        if self.direction == "U":
            self.drone_achievement += 1
            self.drone_battery_level -= 10
            self.location[1] -= 1
            self.direction = "L"
            return True

        elif self.direction == "R":
            self.location[0] += 1
            self.drone_battery_level -= 10
            self.drone_achievement += 1
            self.direction = "U"
            return True

        elif self.direction == "L":
            self.location[0] -= 1
            self.drone_achievement -= 1
            self.direction = "D"
            return True

        elif self.direction == "D":
            self.location[1] += 1
            self.drone_battery_level -= 10
            self.drone_achievement += 1
            self.direction = "R"
            return True

    def stop_signal(self):
        time.sleep(self.stop_timing)
        self.drone_achievement -= 1
        self.move_forward_coordinate()
        return True




# adding things to Drone environment
def drone_things(env, thing, location):
    env_size = np.array([env.width, env.height])
    if (location <= env_size).all():
        env.add_thing(thing, location)
    else:
        print(" drone cannot be added {} at this coordinate: {} as Drone environment size is: {}, choose different coordinate".format(thing, location, env_size))
        return


def DroneEnvironment(agent, env, runs):
    for i in range(runs):
        if not env.is_done():
            print("\nRun: {}: Drone present coordinate is: {}".format(i, agent.location))
            print(" Drone Performance is : ", agent.drone_achievement)
            print(" Drone Battery Level is : ", agent.drone_battery_level)
            env.step()
        else:
            if (agent.drone_battery_level <= 0):
                print("    {} has run out of drone battery level so it stopped moving now".format(agent))
                print(" Drone Performance is : ", agent.drone_achievement)
                return
            elif (agent.location == agent.delivery_location).all():
                print(" Drone reached its goal which is at {} and current location of agent is: {}".format(
                    agent.location, agent.delivery_location))
                print(" Drone Performance is : ", agent.drone_achievement)
                return





'''Drone actions is based on the destination(goal) when drone flies '''
def DroneGoalBased():
    drone_model = {}

    def program(percepts):
        print(" Drone Goal Based is : ", drone_model)
        print(" Drone Percept is:", percepts)

        for i in percepts:
            if str(i) in drone_model.keys():
                print("Drone Percept is Matched: ", i)
                drone_action = drone_model[str(i)]
                print("Drone Activity: ", drone_action)
                return drone_action
            if isinstance(i, DroneBattery):
                drone_action = "stop_signal"
                drone_model[str(i)] = drone_action
                return drone_action
            elif isinstance(i, DronePickItem):
                drone_action = "pick_item"
                drone_model[str(i)] = drone_action
                return drone_action
            elif isinstance(i, DroneObstacle):
                drone_action = "move_to_left_coordinate"
                drone_model[str(i)] = drone_action

            drone_present_status = i.check_current_state()
            if drone_present_status:
                return "drone_destination"
            else:
                if (i.location[0] < i.delivery_location[0]):
                    i.direction = "U"
                    return "move_forward_coordinate"
                elif (i.location[0] > i.delivery_location[0]):
                    i.direction = "D"
                    return "move_forward_coordinate"
                elif (i.location[1] < i.delivery_location[1]):
                    i.direction = "U"
                    return "pick_item"
                else:
                    i.direction = "U"
                    return "move_to_left_coordinate"

    return DroneDeliveryAgent(program)




def DroneGoalBasedAgentRun():
    air_coordinates_map_goal_based = Air_Coordinates_Map(100, 100)
    delivery_drone = DroneGoalBased()
    drone_battery = DroneBattery()
    pick_item_1 = DronePickItem()
    pick_item_2 = DronePickItem()
    drone_obstacle_1 = DroneObstacle()
    # drone_obstacle_2 = DroneObstacle()
    drone_things(air_coordinates_map_goal_based, delivery_drone, [0, 0])
    drone_things(air_coordinates_map_goal_based, pick_item_2, [3, 1])
    drone_things(air_coordinates_map_goal_based, drone_battery, [4, 1])
    drone_things(air_coordinates_map_goal_based, pick_item_1, [5, 1])
    drone_things(air_coordinates_map_goal_based, drone_obstacle_1, [2, 0])
    # drone_things(air_coordinates_map_goal_based, drone_obstacle_2, [5, 0])
    print("***************** Goal Based Agent Run Starts for Delivery Drone **************************\n")
    print("\nDrone is working is Model state... start....")
    DroneEnvironment(delivery_drone, air_coordinates_map_goal_based, 25)
    print("\n Drone has completed its model based agent run...")
    print("***************** Goal Based Run Over for Delivery Drone **************************\n")




def DroneModelBased():
    drone_model = {}
    def program(percepts):
        print(" Drone Model is : ", drone_model)
        print(" Drone Percept is:", percepts)
        for i in percepts:
            if str(i) in drone_model.keys():
                print("Drone Percept is Matched: ", i)
                action = drone_model[str(i)]
                print("Drone Activity: ", action)
                return action
            if isinstance(i, DroneBattery):
                action = "stop_signal"
                drone_model[str(i)] = action
                return action
            elif isinstance(i, DronePickItem):
                action = "pick_item"
                drone_model[str(i)] = action
                return action
            elif isinstance(i, DroneObstacle):
                action = "move_to_left_coordinate"
                drone_model[str(i)] = action
                return action
        return "move_forward_coordinate"

    return DroneDeliveryAgent(program)



def DroneModelAgentRun():
    air_coordinates_map_model = Air_Coordinates_Map(100, 100)
    delivery_drone = DroneModelBased()
    drone_battery = DroneBattery()
    pick_item_1 = DronePickItem()
    pick_item_2 = DronePickItem()
    pick_item_3 = DronePickItem()
    drone_obstacle_1 = DroneObstacle()
    drone_obstacle_2 = DroneObstacle()
    drone_things(air_coordinates_map_model, delivery_drone, [0, 0])
    drone_things(air_coordinates_map_model, drone_battery, [4, 0])
    drone_things(air_coordinates_map_model, pick_item_1, [3, 0])
    drone_things(air_coordinates_map_model, pick_item_2, [5, 2])
    # drone_things(air_coordinates_map_model, pick_item_3, [4, 3])
    drone_things(air_coordinates_map_model, drone_obstacle_1, [2, 4])
    drone_things(air_coordinates_map_model, drone_obstacle_2, [3, 6])
    print("***************** Model Based Agent Run Starts for Delivery Drone **************************\n")
    print("\nDrone is working is model Based state... start....")
    DroneEnvironment(delivery_drone, air_coordinates_map_model, 20)
    print("\n Drone has completed its Goal Based agent run...")
    print("***************** Model Based Run Over for Delivery Drone **************************\n")



# Simple Reflex Agent (actions based on the percept)
def DroneSimpleAgent():
    def program(percepts):
        for i in percepts:
            if isinstance(i, DroneBattery):
                return "stop_signal"
            elif isinstance(i, DronePickItem):
                return "pick_item"
            elif isinstance(i, DroneObstacle):
                return "move_to_left_coordinate"
        return "move_forward_coordinate"
    return DroneDeliveryAgent(program)


def DroneSimpleAgentRun():
    air_coordinates_map_1 = Air_Coordinates_Map(50, 50)
    delivery_drone = DroneSimpleAgent()
    drone_battery = DroneBattery()
    pick_item = DronePickItem()
    drone_obstacle = DroneObstacle()
    drone_things(air_coordinates_map_1, delivery_drone, [0, 0])
    drone_things(air_coordinates_map_1, drone_battery, [4, 0])
    drone_things(air_coordinates_map_1, pick_item, [2, 0])
    drone_things(air_coordinates_map_1, drone_obstacle, [2, 5])
    print("***************** Simple Agent Run Starts for Delivery Drone **************************\n")
    print("\nDrone is made to work in Simplex Reflex Model... start....")
    DroneEnvironment(delivery_drone, air_coordinates_map_1, 25)
    print("\nDrone Simplex Reflex Model ends here")
    print("***************** Simple Agent Run Over for Delivery Drone **************************\n")





'''Searching Methods Ends'''


'''Drone actions is based on the destination(goal) when drone flies '''

'''Searching Methods Starts'''

'''Here we are creating the Grid for 3x3 and 5x5 and apply different searching algorithms'''


def createMap():
    # 3x3 Grid where c1, c2, ..... refers as co-orodinates
    global dronemap
    dronemap = {
        "c1": {"c2": 5, "c4": 2},
        "c2": {"c1": 5, "c5": 3, "c3": 2},
        "c3": {"c6": 1, "c2": 2},
        "c4": {"c1": 5, "c5": 1, "c7": 5},
        "c5": {"c4": 1, "c6": 1, "c2": 3, "c8": 4},
        "c6": {"c3": 1, "c5": 1, "c9": 3},
        "c7": {"c4": 5, "c8": 4},
        "c8": {"c7": 4, "c5": 4, "c9": 5},
        "c9": {"c6": 3, "c8": 5}
    }
    # 3x3 Grid locations
    dronemap = UndirectedGraph(dronemap)
    dronemap.locations = {
        "c1": (0, 0),
        "c2": (0, 1),
        "c3": (0, 2),
        "c4": (1, 0),
        "c5": (1, 1),
        "c6": (1, 2),
        "c7": (2, 1),
        "c8": (2, 2),
        "c9": (2, 3)
    }
    return dronemap
    # 5x5 Grid where c1, c2, ..... refers as co-orodinates
    # dronemap = {
    #     "c1": {"c2": 1, "c6": 2},
    #     "c2": {"c1": 1, "c3": 3, "c7": 4},
    #     "c3": {"c2": 3, "c8": 3, "c4": 1},
    #     "c4": {"c3": 1, "c5": 5, "c9": 3},
    #     "c5": {"c4": 5, "c10": 4},
    #     "c6": {"c1": 2, "c7": 3, "c11": 1},
    #     "c7": {"c2": 1, "c6": 3, "c8": 4, "c12": 1},
    #     "c8": {"c7": 4, "c3": 3, "c9": 1, "c13": 2},
    #     "c9": {"c8": 1, "c4": 3, "c10": 4, "c14": 5},
    #     "c10": {"c5": 4, "c9": 4, "c15":2},
    #     "c11": {"c6": 1, "c12": 3},
    #     "c12": {"c11": 3, "c7": 1, "c13": 5},
    #     "c13": {"c12": 5, "c8": 2, "c14": 2},
    #     "c14": {"c13": 2, "c9": 5, "c15": 1},
    #     "c15": {"c14": 1, "c10": 2}
    # }

    # dronemap.locations = {
    #     "c1": (0, 0),
    #     "c2": (0, 1),
    #     "c3": (0, 2),
    #     "c4": (0, 3),
    #     "c5": (0, 4),
    #     "c6": (1, 0),
    #     "c7": (1, 1),
    #     "c8": (1, 2),
    #     "c9": {1, 3},
    #     "c10": {1, 4},
    #     "c11": {2, 0},
    #     "c12": {2, 1},
    #     "c13": {2, 2},
    #     "c14": {2, 3},
    #     "c15": {2, 4}
    # }
    # return dronemap


'''Depth First Search Start'''


def tree_breadth_search_for_vis(problem):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Don't worry about repeated paths to a state. [Figure 3.7]"""

    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    # Adding first node to the queue
    frontier = deque([Node(problem.initial)])

    node_colors[Node(problem.initial).state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    while frontier:
        # Popping first node of queue
        node = frontier.popleft()

        # modify the currently searching node to red
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            # modify goal node to green after reaching the goal
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        frontier.extend(node.expand(problem))

        for n in node.expand(problem):
            node_colors[n.state] = "orange"
            iterations += 1
            all_node_colors.append(dict(node_colors))

        # modify the color of explored nodes to gray
        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))

    return None


def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first."
    iterations, all_node_colors, node = tree_breadth_search_for_vis(problem)
    return (iterations, all_node_colors, node)




'''Depth First Search'''


def graph_search_for_vis(problem):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the first one. [Figure 3.7]"""
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    frontier = [(Node(problem.initial))]
    explored = set()

    # modify the color of frontier nodes to orange
    node_colors[Node(problem.initial).state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    while frontier:
        # Popping first node of stack
        node = frontier.pop()

        # modify the currently searching node to red
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            # modify goal node to green after reaching the goal
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)

        for n in frontier:
            # modify the color of frontier nodes to orange
            node_colors[n.state] = "orange"
            iterations += 1
            all_node_colors.append(dict(node_colors))

        # modify the color of explored nodes to gray
        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))

    return None


def depth_first_graph_search(problem):
    """Search the deepest nodes in the search tree first."""
    iterations, all_node_colors, node = graph_search_for_vis(problem)
    return (iterations, all_node_colors, node)

'''Depth First Search'''



''' Best First Search  '''


def depth_limited_search_graph(problem, limit=-1):
    '''
    Perform depth first search of graph g.
    if limit >= 0, that is the maximum depth of the search.
    '''
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    frontier = [Node(problem.initial)]
    explored = set()

    cutoff_occurred = False
    node_colors[Node(problem.initial).state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    while frontier:
        # Popping first node of queue
        node = frontier.pop()

        # modify the currently searching node to red
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            # modify goal node to green after reaching the goal
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        elif limit >= 0:
            cutoff_occurred = True
            limit += 1
            all_node_colors.pop()
            iterations -= 1
            node_colors[node.state] = "gray"

        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)

        for n in frontier:
            limit -= 1
            # modify the color of frontier nodes to orange
            node_colors[n.state] = "orange"
            iterations += 1
            all_node_colors.append(dict(node_colors))

        # modify the color of explored nodes to gray
        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))

    return 'cutoff' if cutoff_occurred else None


def depth_limited_search_for_vis(problem):
    """Search the deepest nodes in the search tree first."""
    iterations, all_node_colors, node = depth_limited_search_graph(problem)
    return (iterations, all_node_colors, node)

''' depth_limited_search_for_vis  '''

'''iterative_deepening_search_for_vis Start'''

def iterative_deepening_search_for_vis(problem):
    for depth in range(sys.maxsize):
        iterations, all_node_colors, node=depth_limited_search_for_vis(problem)
        if iterations:
            return (iterations, all_node_colors, node)


'''  Breadth First Graph Search Ends '''


def breadth_first_search_graph(problem):
    "[Figure 3.11]"

    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    node = Node(problem.initial)

    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return (iterations, all_node_colors, node)

    frontier = deque([node])

    # modify the color of frontier nodes to blue
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    explored = set()
    while frontier:
        node = frontier.popleft()
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        explored.add(node.state)

        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    node_colors[child.state] = "green"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))
                    return (iterations, all_node_colors, child)
                frontier.append(child)

                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

''' Breadth First Graph Search Start'''



'''Recersive Best First Search Start'''


'''Searching Methods Ends'''



def assignment2():
    print("----------- Different Search Method Implementations -------")
    search_list = ['BFTS', 'DFGS', 'DLS', 'IDS', 'RBFS', 'BFGS']
    for i in search_list:
        if i == 'BFTS':
            print("-----------Start of Breadth First Tree Search -------")
            a = createMap()
            all_node_colors = []
            # 5x5 Grid
            # drone_problem = GraphProblem('c1', 'c15', a)
            # 3x3 Grid
            drone_problem = GraphProblem('c1', 'c9', a)
            iteration, allnodes, final_node = breadth_first_tree_search(drone_problem)
            print(final_node.path())
            print("Path Cost : ", iteration)
            path_BFTS = int(iteration)
            print("-----------END of Breadth First Tree Search  -------")
            print("------------------------------------------------------------------")
            print("------------------------------------------------------------------")
        elif i == 'DFGS':
            print("-----------Start of Depth First Graph Search-------")
            a = createMap()
            all_node_colors = []
            # 5x5 Grid
            # drone_problem = GraphProblem('c1', 'c10', a)
            # 3x3 Grid
            drone_problem = GraphProblem('c1', 'c6', a)
            iteration, allnodes, final_node = depth_first_graph_search(drone_problem)
            print(final_node.path())
            print("Path Cost : ", iteration)
            path_DFGS = int(iteration)
            print("-----------END of Depth First Graph Search-------")
            print("------------------------------------------------------------------")
            print("------------------------------------------------------------------")
        elif i == 'DLS':
            print("-----------Start of Depth Limited Search-------")
            a = createMap()
            all_node_colors = []
            # 5x5 Grid
            # drone_problem = GraphProblem('c1', 'c15', a)
            # 3x3 Grid
            drone_problem = GraphProblem('c1', 'c9', a)
            iteration, allnodes, final_node = depth_limited_search_for_vis(drone_problem)
            print(final_node.path())
            print("Path Cost : ", iteration)
            path_DLS = int(iteration)
            print("-----------END of Depth Limited Search -------")
            print("------------------------------------------------------------------")
            print("------------------------------------------------------------------")
        elif i == 'IDS':
            print("-----------Start of Iterative Deepening Search-------")
            a = createMap()
            all_node_colors = []
            # 5x5 Grid
            # drone_problem = GraphProblem('c1', 'c12', a)
            # 3x3 Grid
            drone_problem = GraphProblem('c1', 'c8', a)
            iteration, allnodes, final_node = iterative_deepening_search_for_vis(drone_problem)
            print(final_node.path())
            print("Path Cost : ", iteration)
            path_IDS = int(iteration)
            print("-----------END of Iterative Deepening Search -------")
            print("------------------------------------------------------------------")
            print("------------------------------------------------------------------")
        elif i == 'BFGS':
            print("-----------Start of Breadth First Search Graph-------")
            a = createMap()
            all_node_colors = []
            # 5x5 Grid
            # drone_problem = GraphProblem('c1', 'c15', a)
            # 3x3 Grid
            drone_problem = GraphProblem('c1', 'c9', a)
            iteration, allnodes, final_node = breadth_first_search_graph(drone_problem)
            print(final_node.path())
            print("Path Cost : ", iteration)
            path_BFGS = int(iteration)
            print("-----------END of Breadth First Search Graph -------")
            print("------------------------------------------------------------------")
            print("------------------------------------------------------------------")


    compare_methods = [path_BFTS, path_DFGS, path_DLS, path_IDS, path_BFGS]
    compare_methods.sort()
    print(compare_methods)






if __name__ == "__main__":

    '''Assignment 1 question'''
    print("Here I'm Demonstrating Different Agents")

    print("Simple Reflex Agent Start\n")
    DroneSimpleAgentRun()
    print("Simple Reflex Agent End\n")
    print("------------------------------------------")


    print("Model Based Agent Start\n")
    DroneModelAgentRun()
    print("Model Based Agent End\n")
    print("------------------------------------------")

    print("Goal Based Agent Start\n")
    DroneGoalBasedAgentRun()
    print("Goal Based Agent End\n")
    print("------------------------------------------")

    print("------Different Searching Methods")
    assignment2()
    print("--------End of Both the Assignment")

