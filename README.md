# City Lattice

Here I am exploring a simple model of a city as a small-world lattice.
Especially, I am interested to see how highways affect mobility patterns, total travel time, and also energy.

This is a multi-graph, with two kinds of edges:
- Euclidean distance
- Time, calculated as a function: time = distance/speed

That is, first we see a city as a 2-dimensional lattice (grid), then we add some small-world connections (think of highways).

To search the parameter space (p - the probability of a small-world connection between any pair of nodes, and the speed, kept constant for highways and city streets), there is a loop that iterates over some p values and speeds:

- Euclidean distances are calculated between all neighbors in the network.
- Then, using the global velocity and these neighbor distances, travel times are calculated between all neighbors.
- Based on approximate real-world speed differences, small-world speeds are set to be 2x that of lattice speeds
- Finally, given a randomly chosen starting point and destination, the shortest path by distance and time is calculated.
- The "best city" (with the largest set-difference between time and distance paths) is saved to file, along with some other innformation.

