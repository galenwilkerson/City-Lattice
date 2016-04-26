# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:59:30 2016

@author: galen
@email: gjwilkerson@gmail.com

City Lattice


Investigate effects of small-world connections on shortest travel time paths

Compare shortest distance path to shortest travel time path

Represent City as Lattice

Set travel speed

Add small world connections

For a particular, randomly chosen start and destination node:

- calculate the shortest path by distance
- calculate the shortest path by time (based on speed)

- display both

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import pandas as pd
import pickle as pk


class City:
    '''
    City as a lattice with small-world connections    
    
    Note this is similar to the Labyrinth, except has small-world connections
    '''
    
    
    def __init__(self, rows, columns, speed):
        '''
        Init 2-D lattice without small-world connections

        store lattice as networkx graph
        assume all neighbors are connected        

        also set the speed of travel
        '''

        if (speed == 0):
            raise("Error, speed cannot be zero.")
            exit()

        self.lattice = nx.grid_2d_graph(rows, columns)

        self.speed = speed


    def addSmallWorldConnections(self, p):
        '''
        For each pair of nodes (intersections), 
        add a small world connection with probability p
        only consider non-neighbors to avoid duplicate edges
        '''
        for node1 in self.lattice.nodes():
            
            # find the non-neighbors of node1 (avoid duplicate edges)
            neighbors = self.lattice.neighbors((0,1))            
            non_neighbors = set(self.lattice.nodes()) - set(neighbors)
            
            for node2 in non_neighbors: #self.lattice.nodes():
                P = rnd.random()
                if (P < p):
                    self.lattice.add_edge(node1, node2)

    def euclideanDistance(self, nodeID1, nodeID2):
        '''
        input:  two node ids, each an (row, col) tuple of the node location
        output:  Euclidean distance float
        '''
        a = np.array(nodeID1)
        b = np.array(nodeID2)        
        distance = float(np.linalg.norm(a-b))
        return(distance)        


    def allNeighborsEuclideanDistance(self):
        '''
        set edge distances to be Euclidean distances between all pairs of nodes
        '''

        for edge in self.lattice.edges_iter():
            nodeA = edge[0]
            nodeB = edge[1]
            distance = float(self.euclideanDistance(nodeA, nodeB))
            self.lattice.edge[nodeA][nodeB]['distance'] = distance
            #print(self.lattice.edge[nodeA][nodeB]['distance'] )

#    def timeApartNeighbors(self, nodeID1, nodeID2):
#        '''
#        input: two NEIGHBOR NODES
#        returns: time apart float (using distance and velocity)
#
#        timeApart = distance/velocity
#        '''
#        
#        nodeA = edge[0]
#        nodeB = edge[1]        
#        distance = self.lattice.edge[nodeA][nodeB]['distance']
#        timeApart = distance/float(self.speed)
#        print(distance)
#
##        print(timeApart)
#        
#        return(timeApart)



    def allNeighborsTimeApart(self):
        '''
        set edge distances to be Euclidean distances between all pairs of nodes
        '''

        for edge in self.lattice.edges_iter():
            nodeA = edge[0]
            nodeB = edge[1]
            distance = self.lattice.edge[nodeA][nodeB]['distance']
#            print(distance)
            timeApart = distance/float(self.speed) #self.timeApartNeighbors(nodeA, nodeB)
#            print(timeApart)
            self.lattice.edge[nodeA][nodeB]['time_apart'] = timeApart


    def printCity(self):
        '''
        print the edges and any weight values
        '''
        print("nodeA nodeB distance")
        for edge in self.lattice.edges_iter():
            nodeA = edge[0]
            nodeB = edge[1]
            print(edge, self.lattice.edge[nodeA][nodeB])

    def drawCity(self, title = "city"):
        '''
        draw the city lattice, then the small world connections
        
        need to draw long-distance connections as arcs
        
        '''
        
        
        # get the position tuple
        pos = dict( (n, n) for n in self.lattice.nodes() )
#        labels = dict( ((i, j), i * 10 + j) for i, j in self.lattice.nodes() )
#        nx.draw_networkx(self.lattice, pos=pos, labels=labels, node_size = 1000/self.lattice.size())
        nx.draw_networkx(self.lattice, pos=pos, with_labels = False, node_size = 1000/self.lattice.number_of_nodes())
#        nx.draw_networkx_edge_labels(self.lattice, pos = pos)
        plt.axis('equal')
        plt.title(title)
        plt.axis('off')
        plt.show()


    def drawPaths(self, paths, labels, colors):
        '''
        draw the shortest path(s)
        
        inputs: 
        a list of paths, where one path is a list of node ids (in this case tuples),
        a list of labels (for the title)
        a list of colors (for each path)
        '''
        
        G = self.lattice

        # draw the graph
        pos = dict( (n, n) for n in G.nodes() )
        nx.draw_networkx(G, pos=pos, with_labels = False, node_size = 1000/G.number_of_nodes(), alpha = 0.3)

        # draw each path
        i = 0
        for path in paths:
            path_edges = list(zip(path,path[1:]))
            nx.draw_networkx_nodes(G,pos,nodelist=path,node_color=colors[i], node_size=10)
            nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color=colors[i],width=5, alpha = 0.7)
            i += 1
            
        plt.axis('equal')
        plt.axis('off')
        plt.title(", ".join(labels))
        plt.show()    


def main():
    '''
    Tester
    Create a city
    draw it
    
    add some small world connections
    draw it
    
    find the Euclidean distance between all neighbors
    
    using the velocity, find the time apart between all neighbors
    '''    

    # lattice size
    nRows = 30
    nCols = 30


    # small world connection probability
    p = .0001


    # now set the global spatial velocity
    speed = .9

    # create city, including lattice
    city1 = City(nRows, nCols, speed)

    city1.drawCity("city initialized " + str(nRows) + "x" + str(nCols))

    # add some small world connections
    city1.addSmallWorldConnections(p)   


    city1.drawCity("with small world connections, p = " + str(p))

    # for convenience, get the lattice graph (yes, this should use a 'getter', 
    # and data members should be private!)
    G = city1.lattice


    # find geographic distance    
    startNode = rnd.choice(G.nodes())
    destinationNode = rnd.choice(list(set(G.nodes()) - set(startNode)))
      
    distance = city1.euclideanDistance(startNode, destinationNode)
#    print(distance)    

    # compute geographic distances between all NEIGHBOR nodes -> label existing edges
    city1.allNeighborsEuclideanDistance()
 #   city1.printCity()
    
    # using velocity and distances, compute time distances between all NEIGHBOR nodes -> add another label to existing edges
    city1.allNeighborsTimeApart()
#    city1.printCity()
    

    # find the shortest path between start and destination using DISTANCE
    dist_path = nx.shortest_path(G,source=startNode,target=destinationNode, weight = 'distance')
    
#    city1.drawPath(path, 'shortest path - distance')

    print("Shortest Euclidean Distance path between ", startNode, "and", destinationNode, ":")
    print(dist_path)
 
    # find the shortest path between start and destination using TIME
    time_path = nx.shortest_path(G,source=startNode,target=destinationNode, weight = 'time_apart')

    

#    city1.drawPath(shortest_path, 'shortest path - timeApart')

    print("Shortest time path between ", startNode, "and", destinationNode, ":")
    print(time_path)

    city1.drawPaths([dist_path, time_path], ['shortest path - distance', 'shortest path - timeApart'], ['b','g'])


    # get the set difference
    path_difference = set(dist_path) - set(time_path)
    print("path difference is " + str(path_difference))
    

    ######################################

    # now search the speeds to find which gives the most path differences

    # keep track of path length difference (number of hops)
    max_set_difference_size = 0

    best_p = 0
    best_speed = 0

    best_city = None    
    
    
    # iterate over p values
    p_range = np.arange(0, .001, 0.0001)
    
    # itereate over the speed exponents
    min_speed_exp = -3
    max_speed_exp = 3
    speed_exponent_range = range(min_speed_exp, max_speed_exp)    
    
    # set the dataframe indices and column names
    row_indices = list(p_range)
    column_names = list(speed_exponent_range)
    column_names = ', '.join(map(str, column_names))
    column_names.split(", ")
     
    

    for p in p_range:
        
#        print("p: " + str(p))
        
        
        for speed_exponent in speed_exponent_range:
            
            #  REALLY SHOULD DO THIS MANY TIMES AND KEEP AVERAGE OR SO
            
            speed = 10**speed_exponent
            
            city1 = City(nRows, nCols, speed)
            
#            print("speed: " + str(speed))
            
            # add some small world connections
            city1.addSmallWorldConnections(p)   

            G = city1.lattice
            
            # find geographic distance    
            startNode = rnd.choice(G.nodes())
            destinationNode = rnd.choice(list(set(G.nodes()) - set(startNode)))
                      
            # compute geographic distances between all NEIGHBOR nodes -> label existing edges
            city1.allNeighborsEuclideanDistance()
            
            
            city1.allNeighborsTimeApart()
            
            # find the shortest path between start and destination using DISTANCE
            dist_path = nx.shortest_path(G,source=startNode,target=destinationNode, weight = 'distance')
    
            # find the shortest path between start and destination using TIME
            time_path = nx.shortest_path(G,source=startNode,target=destinationNode, weight = 'time_apart')
    
            # get the size of the difference between the paths
            path_difference = set(dist_path) - set(time_path)            
            size = len(path_difference)
                        
            if (size > max_set_difference_size):
                max_set_difference_size = size
                best_p = p
                best_speed = speed
                
                # keep a copy of the city
                best_city = city1
    
                print("Speed: " + str(speed) + " and p: " + str(p) + " gives max set difference of " + str(max_set_difference_size))
    
                print("Shortest time path between ", startNode, "and", destinationNode, ":")
                print(time_path)
            
                best_city.drawPaths([dist_path, time_path], ['shortest path - distance', 'shortest path - timeApart'], ['b','g'])




    pk.dump((max_set_difference_size, best_p, best_speed, best_city), open( "best_city_data.pkl", "wb" ))


def main_small():
    '''
    Tester
    Create a city
    draw it
    
    add some small world connections
    draw it
    
    find the Euclidean distance between all neighbors
    
    using the velocity, find the time apart between all neighbors
    '''    

    # lattice size
    nRows = 5
    nCols = 5


    # small world connection probability
    p = .01


    # now set the global spatial velocity
    speed = .9

    # create city, including lattice
    city1 = City(nRows, nCols, speed)

    city1.drawCity("city initialized " + str(nRows) + "x" + str(nCols))

    # add some small world connections
    city1.addSmallWorldConnections(p)   


    city1.drawCity("with small world connections, p = " + str(p))





if (__name__ == '__main__'):
    main()
    #main_small()
        

        
        