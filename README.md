# Knowledge_Representation
Here I'm implementing different agents and searching algorithm for my Assignment.

To run this file follow this steps

Step 1: Go to Github.com

Step 2: download aima/pyton using below link
        https://github.com/aimacode/aima-python
        or you can clone it using below link
        https://github.com/aimacode/aima-python.git

Step 3: add this file into the folder

Step 4: you can use pycharm to run this file

This is the Report for submission

Munster Technology University

Subject	:	Knowledge Representation
Subject Code	:	COMP9016
Name	:	Peter Sunny Shanthveer Markappa
Student Number	:	R00208303


1.1 BUILDING 2D World
Aim:
This 2D Game world is Drone Agent where it has to pick the package from the specified location and drop it to the delivery location.
This concept I have taken it considering the online food delivery system where our drone agent will pick food from specific restaurant and delivery it to destination location.  This agent gets the co-ordinates via GPS. This agent takes new coordinates (x, y) axis of the package to be picked and move to the destination direction in zig-zag manner. This agent has battery for moving.
The grid that is made for drone has 1) source 2) destination 3) pick-package 4) destination 5) danger zone
The problem can be formulated as:
1.	Source or Initial State: Agent will be in the (0,0) co-ordinated
2.	Actions to be performed: Pick the package, look for danger zone if it comes across danger zone it has to move from that co-ordinate and move to the destination which I have considered here as (5,5).
3.	Goal of Agent: Agent delivers the package before its battery gets drained out.
4.	Path cost: Moving through each grid will cost 10 units of battery drain out.
Below is the PEAS table1.1 
Agent Type	Performance Measure	Environment	Actuators	Sensors
Drone	If it picks the package and makes delivery to the given destination	2D matrix with
pick package location, drop location, danger zone	Jointed arm and hand	Camera, Sensors
Table 1.1: This above table is used by referencing the Book 3rd_Edition - Artificial Intelligence A Modern Approach-Stuart J. Russell, Peter_Norvig – Page No.:42


 In this 2D game I have implemented Simplex Reflex, Model Based, Goad Based Agent approach to see the performance of all the models.

1.	Critique the advantages and disadvantages of each agent type.
I have implemented Simplex Reflex, Model Based, and Goal Based so I will compare these agent type with each other starting from basic
a.	Simple Reflex
Simple-Reflex-Agent = current-Percept + (rules-Condition-Action)
Mechanism of this Drone world
1.	Drone will fly in random manner in the grid to pick and drop the package
2.	Drone cannot pick second package even if gets pick co-ordinates while delivering the first package.
Rules
1.	If the battery is drained out in the middle of delivery, then agent will stop working and dies.
2.	It consumes 10 units of battery.
Pro of this model:
1.	As drone is applying for the simple reflex model, it may find shortest route map and reach the destination before its battery is drained out.
Disadvantages of this model:
1.	Drone don’t fly in back direction, also it doesn’t give guarantee of finding shortest path.
2.	It fails to deliver the package if battery is drained out and delivery location is very far. 

		
	Destination 
	Package 
	 	
	
		
				
Source 	
		
Simple Agent Model for our Drone

 
Here drone stopped to pick the item

 
Here drone Moving to left coordinate when it comes across the obstacles

 
Drone has reached its destination and its performance is 10
b.	Model Based

Modal-Based-Agent = Simple-Reflex + Model(previous-state(s))

This agent is extension of simple reflex agent along with keeping the track of internal state. This agent can handle partially observable task environment.

Advantages: 
i.	It finds the shortest way to pick the package and move along side of the grid.
ii.	It can pick 2 items if it finds new pick-up package is either near to the destination or if it to the 1st package

Dis-advantages:
i.	Most of the time the agent may not be able to delivery the second package if it picks up as it may get drained out of the battery.
ii.	This drone does not estimate the battery level if it plans to pick the second item
  Package-2	
		Destination-1 

Package-1 			
	
		
				
Source 
			
Model Based Agent Grid
 
Drone has stopped to pick first package
 
Drone has found obstacles to moving to left co-ordinate
 
Drone has dropped its package
c.	Goal Based Agent

This agent worked smartly in this model to deliver the package partially observable task environment.

Advantages: 
i.	It finds optimal solution when compared to simple reflex and model based as this agent will learn if more number runs are given.
ii.	It finds the shortest way to pick the package and move along side of the grid but does not move back direction and diagonally it will move on right, left, forward.

Dis-advantages:
i.	I take more number runs to learn whether to pick the second package or not and deliver the first package first before battery dies.

  Package-2	
	
Destination-1 
	Package-1 			
	
		
				
Source 
			





 
Drone decided to pick the package-1
 
Drone decided to pick the package-2

 
Drone decided to pick the package-2 and drop the package

Performance of Different agents
Type of model	Drone Source	Destination	performance
Simple Reflex	(0,0)	5,5	10
Model Based	(0,0)	5,5	16
Goal Based	(0,0)	5,5	10

Conclusion of performance:
The performance of simple reflex and goal based seems same for above data but we must also consider that the goal based is picking more than one package while it is on the way to delivery for first package.


1.2 – Searching Methods
1.	Overview
Based on the environment we choose the searching methods are basically classified into 2 categories
a.	Informed
This type of search algorithm has information of their goal state which makes them more efficient than uniformed algorithm.
i.	A* Search
ii.	Recursive Best First Search
iii.	Best First Search
iv.	Uniform cost search
v.	Greedy Method
vi.	Graph search
b.	Uniformed
This type of search algorithm has no proper information of their goal state except one that is provided in the problem.
i.	Breadth First Search
ii.	Iterative Deepening Search
iii.	Depth First Search
iv.	Depth Limited Search

2.	Difference between searches
Informed Search	Uniformed Search
Low cost	High cost
Either they may complete or may not complete the search	This search will always complete the search
The time complexity of finding the solution is less	Time complexity of finding the solution is high when compared to informed search as surely, they will find the search complete
Implementation is not so lengthy	Implementation is lengthy
		
3.	Our Environment
I have built 3x3 and 5x3 grid for implementation of this methods where c1, c9 are the co-ordinate noted for our agent drone to travel implementing each search.

4.	Path cost of each method for different grid environment
	Breadth First Tree Search	Depth First Graph Search	Depth Limited Search	Iterative Deepening Search	Breadth First Search Graph
5x5 Grid Environment	Path Cost:  1838	Path Cost:  42	Path Cost:  35	Path Cost:  17	Path Cost:  41
3x3 Grid Environment	Path Cost:  177	Path Cost:  27	Path Cost:  22	Path Cost:  17	Path Cost:  23


** For above content and coding the references are mentioned in the last page



References
1.	https://github.com/aimacode/aima-python/blob/master/agents.ipynb
2.	https://github.com/aimacode/aima-python/blob/master/search.ipynb
3.	https://www.geeksforgeeks.org/difference-between-informed-and-uninformed-search-in-ai/
4.	https://stackoverflow.com/questions/56697930/how-can-i-change-this-to-use-a-q-table-for-reinforcement-learning
5.	https://www.javatpoint.com/types-of-ai-agents





