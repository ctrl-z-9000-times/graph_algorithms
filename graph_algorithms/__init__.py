"""Common Graph Algorithms Library

Library of graph algorithms which operate directly on python data structures.

This library uses many callback functions.  In this module's documentation 
callbacks are written in the following way:
    callback-name(argument-list) -> return-type


Callback Time Complexity:
All time complexities stated in this module assume that user supplied callback 
functions execute in a constant amount of time.


This module is safe for:
>>> from graph_algorithms import *
"""

"""
Remember:
Small optimizations are pointless in python.
Focus on asyptotic time complexity and simplifying things for the user.
"""



__version__ = "0.1"     # Simple "major.minor" versioning
__author__ = "David McDougall <dam1784[at]rit.edu>"
__license__ = "MIT"
__all__ = [
    "NoPathException",
    "CyclicGraphException",
    "depth_first_traversal",
    "depth_first_search",
    "iterative_deepening_depth_first_search",
    "a_star",
    "topological_sort",
    "strongly_connected_components",
]

import itertools
import heapq
import logging
logger = logging.getLogger(__name__)

try:
    # Using numpy is problematic for library code because not all installations have numpy.
    # A star's effective branching factor will only be reported if numpy is avaiable.
    import numpy
except ImportError:
    logger.info("import numpy failed, A-star effective branching factors will not be reported")
    numpy = None


class NoPathException(Exception):
    """Raised after the input graph has been fully searched and the goal could not be found."""

class CyclicGraphException(Exception):
    """Raised when a cycle is found by an algorithm which can not handle cycles."""



def depth_first_traversal(
    start_vertex,
    adjacent,
    preorder  = None,
    postorder = None,
    revisit   = None,):
    """
    Traverses a graph in depth first fashion.
    Returns an iterator over traversed objects.
    Returned iterator is evaluated lazily.

    A vertex can be any hashable value.

    A vertex is 'visited' when it is yielded to the caller as part of the result.
    Vertexes are visited in preorder.

    Arguments:
        start_vertex
            Starting point of the traversal

        adjacent(vertex) -> iterable-of-vertexes
            Callback function which returns all vertexes which the callbacks
            argument is connected to.  
            The returned iterable is evaluated lazily.

        There are three optional callbacks:
        preorder(vertex)
            Called before visiting every vertex.
        postorder(vertex)
            Called after all of adjacent(vertex) have been visited.
        revisit(vertex)
            Called when when two branches of the search tree touch.
            Called when vertex is seen and no other callbacks are called.
            Revisited vertexes are not yielded to the caller.

    Complexity:
    time  = O(|V| + |E|)
    space = O(|V|)
        where |V| = number of vertexes
              |E| = number of edges
    """
    if preorder is None:  preorder  = lambda vertex:None
    if postorder is None: postorder = lambda vertex:None
    if revisit is None:   revisit   = lambda vertex:None

    preorder(start_vertex)
    yield start_vertex

    # frontier = [(v0, adjacent(v0)), (v1, adjacent(v1)), ... (vN, adjacent(vN))]
    # FILO/Stack
    neighbors = adjacent(start_vertex)
    frontier = [(start_vertex, iter(neighbors))]
    # Visited does not include the frontier's neighbors
    visited = set([start_vertex])

    while frontier:
        parent, adj_iter = frontier[-1]
        try:
            vertex = next(adj_iter)
        except StopIteration:
            postorder(parent)
            frontier.pop()
        else:
            if vertex in visited:
                revisit(vertex)
            else:
                preorder(vertex)
                visited.add(vertex)
                yield vertex
                neighbors = adjacent(vertex)
                frontier.append((vertex, iter(neighbors)))



def depth_first_search(start_vertex, adjacent, goal):
    """
    Searches a graph in depth first fashion.
    Returns a path to the first goal vertex found.

    Arguments:
        start_vertex
            The search starts from this vertex.
            A vertex can be any hashable value.

        adjacent(vertex) -> iterable-of-vertexes
            Callback which returns all vertexes which the callbacks argument 
            connects to.  
            The returned iterable is evaluated immediately.

        goal(vertex) -> Boolean
            The first vertex encountered for which goal(vertex) == True is returned.

    Returns:
        tuple(start_vertex, ... goal_vertex)
        The returned value is the path from the start_vertex to the first 
        goal found.  It includes all intermediate vertexes which are passed
        through on the way to the goal.  

    Raises:
        NoPathException, if the goal is not found.

    Search Properties:
    Complete: No, this might not find a solution even if there is one.
    Optimal: No, this might not find the shortest path or the nearest solution.

    Complexity:
    time  = O(|V| + |E|)
    space = O(|V|)
        where |V| = number of vertexes
              |E| = number of edges
    """
    if goal(start_vertex):
        return (start_vertex,)

    # Note: parent_lookup doubles as the seen list. (AKA visited list)
    # Keep start_vertex in the seen list.
    # start_vertex has no parent, use a new unique object which is guarenteed not to be part of the input graph.
    # start_vertex is now in the seen list and its parent is NOT in the seen list.
    # It is OK for start_vertex's parent to not be in the seen list b/c it will never be traversed, its not part of the input graph.
    parent_lookup = {start_vertex: object()}
    search_stack = [start_vertex]

    while search_stack:
        vertex = search_stack.pop()

        # Process the neighbors immediately.
        # Do it now so that the neighbors are put into the parent-lookup before
        # a longer path to them is found and put in first.
        for neighbor in adjacent(vertex):
            # Is this the first time seeing vertex 'neighbor'?
            if neighbor not in parent_lookup:
                parent_lookup[neighbor] = vertex
                search_stack.append(neighbor)

                # Check the adjacent immediately
                if goal(neighbor):
                    # Make the path
                    path = []
                    vertex = neighbor
                    while vertex in parent_lookup:
                        path.append(vertex)
                        vertex = parent_lookup[vertex]
                    path.reverse()
                    return tuple(path)
    raise NoPathException()



def iterative_deepening_depth_first_search(start_vertex, adjacent, goal, max_depth=None):
    """
    Optimal pathfinding through infinite graphs.
    Returns the path to the first goal vertex found.

    A vertex can be any value.  Vertexes do NOT need to be hashable because 
    IDDFS does not keep track of which vertexes have been seen before.

    This algorithm is designed to work on infinite graphs, so it does not keep
    track of which vertexes it has seen before.  It treats the input graph as
    a tree.  This trades off speed for less memory needed.  

    Arguments:
        start_vertex
            The search starts from this vertex.

        adjacent(vertex) -> iterable-of-vertexes
            Callback which returns all vertexes which the callbacks argument 
            is connected to.  

        goal(vertex) -> Boolean
            Returns true only when passed a goal vertex.

        max_depth:
            Optional argument 'max_depth' limits the number of iterations.
            Returned paths will be at most 'max_depth' length.
            Must be an integer greater than or equal to 1 or left 
            default (no limit).

    Returns:
        tuple(start_vertex, ... goal_vertex)
        The returned value is the path from the start_vertex to the goal. 
        It includes all intermediate vertexes which are passed through on 
        the way to the goal.  

    Raises:
        NoPathException, if the goal is not found.

    Search Properties:
    Complete: Yes, if there is a solution this will eventually find it.
    Optimal: Yes, this finds the shortest path.

    Complexity:
    time  = O(b^p)
    space = O(p)
        where b = branching factor of graph
              p = length of path to nearest goal
    """
    if max_depth is not None and max_depth < 1:
        raise TypeError('max_depth must be an integer greater than 0')

    if goal(start_vertex):
        return [start_vertex]

    loop_control = itertools.count() if max_depth is None else range(max_depth-1)

    for depth_limit in loop_control:    # depth_limit = path length -1

        # Stack of all vertexes on the path and their partially searched adjacency iterators
        neighbors = adjacent(start_vertex)
        search_stack = [(start_vertex, iter(neighbors))]
        exausted = True         # Is the graph completely searched?

        while search_stack:
            parent, adj_iter = search_stack[-1]
            try:
                vertex = next(adj_iter)
            except StopIteration:
                search_stack.pop()
            else:
                if len(search_stack) > depth_limit:
                    # Only check goal on leaves
                    if goal(vertex):
                        path = [pair[0] for pair in search_stack]
                        path.append(vertex)
                        return tuple(path)
                    exausted = False
                else:
                    neighbors = adjacent(vertex)
                    search_stack.append((vertex, iter(neighbors)))
        if exausted:
            break
    raise NoPathException()



def a_star(start_vertex, adjacent, cost, heuristic, goal, max_depth=None):
    """
    Finds the shortest path through a weighted graph using a heuristic.
    Used for planning & pathfinding in low dimension spaces.

    Arguments:
        start_vertex
            The search starts from this vertex.
            A vertex can be any hashable value.

        adjacent(vertex) -> iterable-of-vertexes
            Callback which returns all vertexes which the callbacks argument 
            connects to.  

        cost(vertex1, vertex2) -> value
            Cost of traversing the edge from vertex1 to vertex2

        heuristic(vertex) -> value
            Estimated cost of getting from the vertex to the goal

        goal(vertex) -> Boolean
            Returns true only if vertex is a goal.

        max_depth
            Optional keyword argument "max_depth" limits the path length.
            For use on infinite graphs, by default there is no limit.

    Returns:
        (start_vertex, ... goal_vertex)

    Raises:
        NoPathException, if no goal is found.

    Logs:
        The following messages may be logged, at most once per call:

        DEBUG:graph_algorithms:Effective branching factor [EBF]

        WARNING:graph_algorithms:Detected inconsistent heuristic

        WARNING:graph_algorithms:Detected inadmissible heuristic
            Note only vertexes on the final path are checked for inadmissible
            heuristics.


    Explanation of A* (pronounced "A star") algorithm:
    Every vertex "n" has a value f(n):
        f(n) = g(n) + h(n)
            where: g(n) is the cost of reaching the vertex from the start
                   h(n) is the estimated cost of reaching the goal from this vertex
    f(n) represents the estimated total cost of reaching the goal through this vertex.

    A* works by repeatedly exploring the vertex with the lowest f(n) value.

    The heuristic h(n) should be admissible (never overestimate the true cost).
    If the heuristic overestimates the cost then A* will ignore potentially optimal paths.
    Overestimating cost by amount D will yield a path that is at most D units longer
    than the shortest/optimal path.  If the heuristic underestimates the cost then A* 
    will explore unfruitful paths which will cause it to take longer and use more memory.

    The heuristic should be consistent A.K.A. monotonic:
        h(n) <= c(n, n') + h(n')                    Triangle inequality
        where: n and n' are vertexes
               n is the parent of n'
               c(n, n') is the cost of traversing the edge from n to n'
    If the heuristic is consistent then f(n) is non-decreasing as A* moves down any path.
    Proof by induction:
        h(n) <= c(n, n') + h(n')                    Criterial for consistency
        g(n) + h(n) <= g(n) + c(n, n') + h(n')      Add the cost of reaching n to both sides
        g(n) + h(n) <= g(n') + h(n')                g(x) is total cost of the path to x
        f(n) <= f(n')
    A consequence of non-decreasing paths is if A* explores a vertex then it has found
    the shortest path to that vertex.

    All consistent heuristics are admissible. Not all admissible heuristics are consistent.

    The effective branching factor is the number of adjacent vertexes that are explored 
    on average.  The more accurate the heuristic is the more vertexes will be
    ruled out of the search resulting in a lower effective branching factor.


    Search Properties:
    Complete: Yes, if there is a solution A* will find it.
    Optimal: Yes, if the heuristic is consistent A* will find the shortest path.

    Complexity:
    time  = O(|E|) = O(b^d)
    space = O(|V|) = O(b^d)
        where: |E| = number of edges explored
               |V| = number of vertexes explored
               b = effective branching factor
               d = optimal path length

    Source:
    "Artificial Intelligence, A modern approach (3rd edition)" by Russel and Norvig
    """
    if max_depth is not None and max_depth <= 0:
        raise ValueError("Argument max_depth must be greater than zero")

    class InternalNode:
        '''
        value   = User data
        parent  = Parent's InternalNode (Or None if starting vertex)
        g       = Total cost of getting to this node
        h       = heuristic(vertex)
        f       = g + h
        depth   = Path length (in vertexes) to this node
        old     = Has a shorter path to this node been found?
        '''
        # Try slots with performance tests, measure both time and memory
        # __slots__ = ('value', 'parent', 'old', 'h', 'g', 'f', 'depth')
        def __init__(self, vertex, parent):
            self.value  = vertex
            self.parent = parent
            self.old    = False
            if parent is None:
                self.depth  = 1
                self.g      = 0
            else:
                self.depth  =  parent.depth + 1
                edge_cost   = cost(parent.value, vertex)
                self.g      = parent.g + edge_cost
            self.h = heuristic(vertex)
            self.f = self.g + self.h

        def __lt__(self, other):
            return self.f < other.f

    start_node = InternalNode(start_vertex, None)

    frontier = [start_node]
    # visited contains all of frontier
    visited = {start_vertex: start_node}
    inconsistent_heuristic = False
    revisits = 0    # num-explore != len(visited) b/c revists overwrite previous entry in visited.

    while frontier:
        node = heapq.heappop(frontier)
        if node.old:
            continue

        if max_depth is not None and node.depth > max_depth:
            break

        vertex = node.value
        if goal(vertex):
            # Make the path
            path = []
            while node is not None:
                path.append(node)
                node = node.parent

            # Check for inadmissibile heuristics along the final path
            total_path_cost = path[0].g
            for vertex in path:
                remaining_path_cost = total_path_cost - vertex.g
                if vertex.h > remaining_path_cost:
                    logger.warning("Detected inadmissible heuristic")
                    break

            calculate_EBF(len(visited) + revisits, len(path))

            return tuple(p.value for p in reversed(path))

        # Explore more of the graph
        for neighbor in adjacent(vertex):
            neighbor_node = InternalNode(neighbor, node)

            never_visited      = neighbor not in visited
            shorter_path_found = False
            if not never_visited:
                previous_visit = visited[neighbor]
                if previous_visit.g > neighbor_node.g:
                    shorter_path_found = True
                    previous_visit.old = True
                    revisits += 1
                    # Detect Negative cost cycles.
                    # TODO: Determine the time complexity of the following loop.
                    cursor = neighbor_node.parent
                    while cursor is not None:
                        if cursor is previous_visit:
                            raise CyclicGraphException("Negative Cost Cycle Detected")
                        cursor = cursor.parent

            if never_visited or shorter_path_found:
                # Visit this neighbor
                visited[neighbor] = neighbor_node
                heapq.heappush(frontier, neighbor_node)

                # Check for inconsistent heuristic (decreasing estimated total path cost)
                if node.f > neighbor_node.f:
                    if not inconsistent_heuristic:
                        inconsistent_heuristic = True
                        logger.warning("Detected inconsistent heuristic")

    raise NoPathException()


def calculate_EBF(explored, depth):
    """
    Helper function for a_star
    Calcluate and log the effective branching factor

    explored: number of vertexes visited
    depth: length of final path
    """
    if numpy is not None:
        # explored + 1 = sum(EBF ** z for z in range(depth+1))
        coefs = [1 for power in range(depth)]
        coefs.append(-explored)
        poly = numpy.poly1d(coefs)
        roots = (root for root in poly.r if root.real >= 0)     # Discard negative roots
        ebf = min(roots, key=lambda root: abs(root.imag))       # Find root with imag == 0
        logger.debug("Effective branching factor %f"%ebf.real)



def topological_sort(vertexes, adjacent):
    """
    Orders the vertexes of a directed acyclic graph by their dependancies.

    Arguments:
        vertexes
            An iterable of hashable python values.

        adjacent(vertex) -> iterable-of-vertexes
            A callback function. It accepts a vertex and returns an iterable of vertexes.
            Edges are directed as adjacent(from) -> (to1, ... toN)
            Note: this considers all vertexes reachable from the input 
            vertexes, even if they are not in the vertexes argument.

    Returns:
        (vertex0, vertex1, ... vertexN)
        Where each vertex has no edges to vertexes to the left of it in the tuple.

    Raises:
        CyclicGraphException, if there is no such ordering of vertexes.

    Complexity:
    time  = O(|V| + |E|)
    space = O(|V|)
        where |V| = number of vertexes
              |E| = number of edges
    """
    sorted_list = []        # Result, is topologically sorted.
    visited_set = set()     # visited_set = set(sorted_list)

    # This is a copy of depth first searches stack.
    on_stack = set()

    def _adj(vertex):
        """Filter visited vertexes out of the users adjacency function."""
        for neighbor in adjacent(vertex):
            if neighbor not in visited_set:
                yield neighbor

    def preorder(vertex):
        on_stack.add(vertex)

    def postorder(vertex):
        on_stack.remove(vertex)
        sorted_list.append(vertex)
        visited_set.add(vertex)

    def revisit(vertex):
        if vertex in on_stack:
            raise CyclicGraphException()

    for vertex in vertexes:
        if vertex in visited_set:
            continue
        dft = depth_first_traversal(vertex, _adj,
                                    preorder  = preorder,
                                    postorder = postorder,
                                    revisit   = revisit)
        for vertex in dft:
            pass

    return tuple(reversed(sorted_list))



def strongly_connected_components(vertexes, adjacent):
    """
    Finds all subsets/components of the graph which are fully connected. A connected component
    is a set of vertexes for which there is a path from every vertex to every other.

    Arguments:
        vertexes
            An iterable of hashable python values.

        adjacent(vertex) -> iterable-of-vertexes
            Callback which accepts a vertex and returns iterable of connected
            vertexes.

    Returns:
        frozenset(component1, component2, ... componentN)
        where a component is frozenset(vertex1, vertex2, ... vertexM)
        A path exists between every pair of vertexes in a component

    Complexity:
    time  = O(|V| + |E|)
    space = O(|V|)
        where |V| = number of vertexes
              |E| = number of edges
    """
    # Tarjan's algorithm, recursive implementation.
    output = []
    preorder_enumerator = itertools.count(0)

    class InternalData():
        def __init__(self, vertex):
            self.vertex    = vertex
            self.index     = next(preorder_enumerator)
            self.lowlink   = self.index
            self.on_stack  = True
        def __eq__(self, other):
            return other == self.user_node
        def __hash__(self):
            return hash(self.user_node)

    stack = []
    visited = {}    # visited[vertex] = InternalData(vertex)

    def connect(vertex):
        assert(vertex not in visited)
        vertex_data = InternalData(vertex)
        visited[vertex] = vertex_data
        stack.append(vertex_data)

        for neighbor in adjacent(vertex):
            if neighbor not in visited:
                neighbor_data = connect(neighbor)
                vertex_data.lowlink = min(vertex_data.lowlink, neighbor_data.lowlink)
            else:
                neighbor_data = visited[neighbor]
                if neighbor_data.on_stack:
                    vertex_data.lowlink = min(vertex_data.lowlink, neighbor_data.index)
        
        if vertex_data.lowlink == vertex_data.index:
            scc = []
            while True:
                top = stack.pop()
                top.on_stack = False
                scc.append(top.vertex)
                if top.index == vertex_data.index:
                    break
            output.append(frozenset(scc))

        return vertex_data

    for vertex in vertexes:
        if vertex not in visited:
            connect(vertex)

    return frozenset(output)





def minimum_spanning_tree(vertexes, adjacent, cost):
    """
    UNIMPLEMENTED!!!

    Finds the minimum spanning tree.

    Arguments:
        vertexes

        adjacent(vertex) -> iterable-of-vertexes

        cost(vertex1, vertex2) -> value

    Returns:
        tuple(vertexes, adjacent)
        Where vertexes is a frozenset
        
        graph is in the same format as it was given.
        adjacent(vertex) -> (adj1, adj2, ... adjN)

    Raises:

    """




def min_cut(source, sink, adjacent, flow):
    """
    UNIMPLEMENTED!!!

    Returns a frozenset of edges,
        Where each edge is of the form (V1, V2)
        Such that there is an edge from V1 to V2.
    """


def max_flow(source, sink, adjacent, flow):
    """
    UNIMPLEMENTED!!!
    """
    return sum(flow(x, y) for x, y in min_cut(source, sink, adjacent, flow))


