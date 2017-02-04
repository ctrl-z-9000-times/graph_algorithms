Common Graph Algorithms Library

Library of graph algorithms which operate directly on python data structures.

This library uses a novel API for representing graphs.  Graph vertexes can be 
any hashable python value and the connectivity between vertexes is
represented with a callback function.  This callback is named the 'adjacent' 
function.  The adjacent function has the following form:

def adjacent(vertex):
    '''
    This function returns all vertexes which the given vertex is connected to.
    '''
    return iterable-of-neighboring-vertexes



Contents:

depth_first_traversal()
    A lazy depth first traversal

depth_first_search()
    A depth first search

iterative_deepening_depth_first_search()
    Searching infinite graphs

a_star()
    Fast optimal pathfinding

topological_sort()
    Dependency resolution.

strongly_connected_components()
    Determines which areas of the graph can reach which other areas.

In the future I would like to implement more algorithms:
- Minimum Spanning Tree
- Min-cut/Max-flow
- Substructure Search


Installation note:
This package optionally uses numpy.
Numpy is used by some unit tests.
Numpy is used to calculate A-stars effective branching factor (EBF).
If numpy is not available then EBF is not reported.


Comments and feedback are welcome
Send to David McDougall  email: dam1784[at]rit.edu
