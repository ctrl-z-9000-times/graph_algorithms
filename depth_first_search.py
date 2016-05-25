#!/usr/bin/python3
'''
depth_first_search.py
'''
from collections.abc import Hashable


def depth_first_search(start_vertex, adjacent, goal):
    '''
    Search a graph in depth first fashion
    Returns the path to the first goal vertex found or None if it can't find the goal.
    eg: [start_vertex, ... goal_vertex]
        where: goal(goal_vertex) == True

    adjacent(vertex) -> iterable-of-vertexes
    Where a vertex can be any hashable python object.
    The adjacent list is evaluated immediately.

    goal(vertex) -> Boolean
    This callback should return True only when the given vertex is the quarry being searched for.

    Complexity:
    time  = O(v+e)
    space = O(v)
        where v = number of vertexes
              e = number of edges
    '''
    if not callable(adjacent): raise ValueError('Argument "adjacent" is not callable')
    if not callable(goal):     raise ValueError('Argument "goal" is not callable')
    if not isinstance(start_vertex, Hashable):
        raise ValueError('Argument "start_vertex" is not hashable, all vertexes must be hashable')

    if goal(start_vertex):
        return [start_vertex]

    parent_lookup = {}
    search_stack = [start_vertex]

    while search_stack:
        vertex = search_stack.pop()
        for neighbor in adjacent(vertex):
            if neighbor not in parent_lookup:
                parent_lookup[neighbor] = vertex
                search_stack.append(neighbor)

                # Check the adjacent immediately
                if goal(neighbor):
                    #
                    # Make the path
                    #
                    path = []
                    vertex = neighbor
                    while vertex in parent_lookup:
                        path.append(vertex)
                        vertex = parent_lookup[vertex]
                    path.append(start_vertex)
                    path.reverse()
                    return path
    return None



def unit_tests():
    import time
    from collections.abc import Iterable
    import random

    #
    # Test functionality
    #

    # Build a graph to search
    simple_graph = (
        (
            (1,2,3),
            (4,5,6),
        ),
        (
            (7,8,9),
            (10,11,12),
        ),
        (
            (13,14,15),
            (16,17,18),
        ),
    )
    def adjacent(vertex):
        if isinstance(vertex, Iterable):
            return vertex
        return []

    # Find the number ten
    path = depth_first_search(simple_graph, adjacent, lambda v: v==10)
    assert(path[0] is simple_graph)
    assert(path[1] is simple_graph[1])
    assert(path[2] is simple_graph[1][1])
    assert(path[3] is 10)

    # Test search when it can't find the goal
    path = depth_first_search(simple_graph, adjacent, lambda v: v==99)
    assert(path is None)

    #
    # Test on a cyclic graph
    #
    class Vertex:
        def __init__(self, depth, parent=None):      # Builds a tree
            self.neighbors = []
            self.parent = parent
            # Depth counts down to 1, vertexes start at max_depth through depth 1
            if depth-1 > 0:
                self.neighbors.append(Vertex(depth-1, self))
                self.neighbors.append(Vertex(depth-1, self))

            # Link every vertex with to parents siblings
            if self.parent:
                if self.parent.parent:
                    parent_uncle = self.parent.parent.neighbors
                    for v in parent_uncle:
                        if v is not self.parent:
                            self.neighbors.append(v)
                
        def adjacent(self):
            return self.neighbors

    complex_graph = Vertex(5)
    p1 = complex_graph.neighbors[1]
    p2 = p1.neighbors[0]
    p3 = p2.neighbors[1]
    goal = p3.neighbors[0]

    path = depth_first_search(complex_graph, Vertex.adjacent, lambda v: v is goal)

    assert(path[0] is complex_graph)
    assert(path[1] is p1)
    assert(path[2] is p2)
    assert(path[3] is p3)
    assert(path[4] is goal)

    path = depth_first_search(complex_graph, Vertex.adjacent, lambda v: False)
    assert(path is None)


    #
    # Test invalid arguments
    #

    # Test unhashable vertex
    unhashable = [[[1,2], [3,4]], [[5,6], [7,8]]]
    try:
        path = depth_first_search(unhashable, adjacent, lambda v:v == 1)
        print(path)
    except ValueError as e:
        pass
    else:
        assert(False)

    # Test invalid adjacent function
    try:
        depth_first_search(simple_graph, None, lambda v:False)
    except ValueError as e:
        pass
    else:
        assert(False)

    # Test invalid goal function
    try:
        depth_first_search(simple_graph, adjacent, None)
    except ValueError as e:
        pass
    else:
        assert(False)

    #
    # Test time complexity
    #
    # Measure the run times with small inputs and large inputs. Find the average per-vertex
    # processing times for each group and compare them.  In theory they should be the same,
    # in practive they should be within an order of magnitude of each other.
    #
    small_time = 0
    small_size = 0
    for max_depth in range(4, 9):
        graph = Vertex(max_depth)
        size = 2**max_depth - 1

        # Use a random leaf as goal
        goal = graph
        while len(goal.neighbors) > 1:  # Ignore the uncle edges
            goal = random.choice(goal.neighbors)

        start_time = time.time()
        path = depth_first_search(graph, Vertex.adjacent, lambda v: v is goal)
        small_time += time.time() - start_time

        small_size += size

        assert(path)
        assert(len(path) == max_depth)

    small_constant = small_time / small_size


    large_time = 0
    large_size = 0
    for max_depth in range(12, 17):
        graph = Vertex(max_depth)
        size = 2**max_depth - 1

        # Use a random leaf as goal
        goal = graph
        while len(goal.neighbors) > 1:  # Ignore the uncle edges
            goal = random.choice(goal.neighbors)

        start_time = time.time()
        path = depth_first_search(graph, Vertex.adjacent, lambda v: v is goal)
        large_time += time.time() - start_time
        large_size += size
        assert(path)
        assert(len(path) == max_depth)

    large_constant = large_time / large_size
    assert(small_constant / large_constant < 10)
    assert(large_constant / small_constant < 10)



if __name__ == '__main__':
    unit_tests()
