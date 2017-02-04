#!/usr/bin/python3

from graph_algorithms import *
# Define acronyms for long function names in graph_algorithms.
dft   = depth_first_traversal
dfs   = depth_first_search
iddfs = iterative_deepening_depth_first_search
scc   = strongly_connected_components
# Also defined in graph_algorithms:
# - a_star
# - topological_sort
import graph_algorithms
import collections.abc
import time
import random
import itertools
import copy
import io
import logging
import unittest


simple_tree = (
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

def simple_adjacent(vertex):
    """Use with simple_tree."""
    if isinstance(vertex, collections.abc.Iterable):
        # Don't shuffle adjacency list for simplest tree, it makes not simple...
        return iter(vertex)
    return iter([])

"""Use these three functions with simple_tree."""
simple_goal = lambda v: v == 14
simple_cost = lambda v1, v2: 1
simple_heuristic = lambda v: 0

# Invalid data structures for testing error messages
unhashable_tree = [
    [
        [1,2], 
        [3,4]
    ], 
    [
        [5,6], 
        [7,8]
    ]
]

def hidden_unhashable(v):
    """Everything past the start-vertex will be unhashable"""
    return [set()]

'''Returns a non-iterable adjacent list'''
invalid_adjacent = lambda vertex: 5



class TreeNode:
    """
    Binary tree for testing

    Simple binary tree for testing graph algorithms
    TreeNode(max_depth) -> graph-vertex
        Nodes are created at depths [max_depth, 1]
        depth == 0 is empty, this makes the subtree_size formula work
    """
    def __init__(self, depth):
        self.depth  = depth
        self.parent = None
        self.adj    = []
        if depth-1 > 0:
            for child_idx in range(2):
                child = type(self)(depth-1)
                child.parent = self
                self.adj.append(child)

    def adjacent(self):
        shuffled = list(self.adj)   # Shuffle a copy, other methods rely on the order of the adjacency list.
        random.shuffle(shuffled)    # Also use a copy b/c each iter returned by this method can't be 
                                    # looking at the same list (b/c the list gets shuffled each call to 
                                    # this method).
        return iter(shuffled)

    def subtree_size(self):
        return 2**self.depth - 1

    def random_leaf(self):
        if self.depth == 1:
            return self
        else:
            # Ignore extra links, go directly to the leaves
            return random.choice(self.adj[:2]).random_leaf()



class UncleNode(TreeNode):
    """
    Cyclic graph for testing

    Builds a binary tree and links every node to its parents siblings
    UncleNode(max_depth) -> graph-vertex
        Nodes are created at depths [max_depth, 1]
        depth == 0 is empty, this makes the subtree_size formula work

    In the adjacent list: first 2 edges are children, remaining are aunts & uncles
    """
    def __init__(self, depth):
        super().__init__(depth)
        if depth-2 > 0:
            for child in self.adj:
                siblings = [x for x in self.adj if x != child]
                for grandchild in child.adj:
                    grandchild.adj.extend(siblings)

    def random_leaf(self):
        if self.depth == 1:
            return self
        else:
            # Ignore uncle links, go directly to the leaves
            return random.choice(self.adj[:2]).random_leaf()



class Node:
    def __init__(self):
        self.adj = set()

    def adjacent(self):
        retval = list(self.adj)     # Shuffle a copy, not the master list.
        random.shuffle(retval)
        return iter(retval)


def randomAcyclicGraph(size):
    """
    Acyclic graph for testing
    Each node has a random number of random edges.
    Graph is guarenteed not to have any cycles.
    """
    graph = []
    for i in range(size):
        node = Node()
        num_edges = random.randrange(len(graph)) if graph else 0
        node.adj.update(random.sample(graph, num_edges))
        graph.append(node)
    random.shuffle(graph)
    return graph


def randomGraph(size, connectedness):
    """Sparse Random Directed Graph"""
    graph = [Node() for _ in range(size)]
    for node in graph:
        max_edges = size * connectedness
        if max_edges > 0:
            num_edges = random.randrange(int(size * connectedness))
        else:
            num_edges = 0
        node.adj.update(random.sample(graph, num_edges))
    random.shuffle(graph)
    return graph


def ringGraph(size):
    graph = [Node() for index in range(size)]
    for index, node in enumerate(graph):
        node.adj.add(graph[(index + 1) % size])
    random.shuffle(graph)
    return graph 


class WeightedNode(Node):
    """
    Undirected, weighted node.
    Adjacency list represenation.
    """
    def __init__(self):
        # _adj[node] = weight
        self._adj = {}

    def connect(self, other, weight=None):
        """
        Add a connection to the graph
        If no weight is given, uses a random weight in the range [0, 1]
        """
        if weight is None:
            weight = random.random()
        self._adj[other] = weight
        other._adj[self] = weight

    def adjacent(self):
        retval = list(self._adj.keys())
        random.shuffle(retval)
        return iter(retval)

    def weight(self, other):
        return self._adj[other]

    @classmethod
    def random_graph(cls, size):
        """Returns 2-tuple of (vertexes, weight-function)."""
        print(cls, size)
        graph = [cls() for index in range(size)]
        for node in graph:
            num_edges = random.randrange(size//2)
            for neighbor in random.sample(graph, num_edges):
                if neighbor != node:
                    node.connect(neighbor)
        return graph, lambda v1, v2: v1.weight(v2)


class SlidingPuzzle:
    """
    Puzzle for testing A*

    Sliding puzzle and machinery needed for driving A* puzzle solver

    puzzle.state[row][col] = value
    Zero value represents the empty square
    """
    def __init__(self, size, solvable=True):
        while True:
            # Make a random initial state
            self.size = size
            vec = list(range(size**2))
            random.shuffle(vec)
            self.state = [[vec[r*size + c] for c in range(size)] for r in range(size)]
            if self.is_solvable() == solvable:
                break
            # else try a new random puzzle

    def is_solvable(self):
        # Flatten the state into a vector
        vec = tuple(itertools.chain.from_iterable(self.state))
        # Count the number of inversions in the vec (ignoring the empty square)
        inversions = 0
        for idx in range(len(vec)):
            if vec[idx] == 0:       # Ignore the empty square
                continue
            for prev in vec[:idx]:
                if vec[idx] < prev:
                    inversions += 1
        # Even permutations are solvable
        return inversions % 2 == 0

    def force_solve(self):
        """Pry the peices out and rearrange them"""
        vec = list(range(self.size**2))
        self.state = [[vec[r*self.size + c] for c in range(self.size)] for r in range(self.size)]
        # Check the heuristics are zero here
        assert(self.heuristic_manhattan() == 0)
        assert(self.heuristic_displaced() == 0)
        assert(self.heuristic_inconsistent() == 0)

    def swap(self, idx1, idx2):
        temp = self.state[idx1[0]][idx1[1]]
        self.state[idx1[0]][idx1[1]] = self.state[idx2[0]][idx2[1]]
        self.state[idx2[0]][idx2[1]] = temp

    def adjacent(self):
        if hasattr(self, '_adj'):
            # Don't shuffle the adjacent list underneath the previous call's iterator, use a copy each time.
            shuffled = self._adj[:]
            random.shuffle(shuffled)
            return iter(shuffled)

        #
        # Get the index of the empty square
        #
        for r, c in itertools.product(range(self.size), repeat=2):
            if self.state[r][c] == 0:
                empty_idx = (r,c)
                break

        #
        # Swap the empty square with adjacent squares
        #
        adj = []
        for direction in ((1,0), (-1,0), (0,1), (0,-1)):
            # Find adjacent square
            swap_idx = (empty_idx[0] + direction[0], empty_idx[1] + direction[1])
            # Bounds  check
            if swap_idx[0] in range(self.size) and swap_idx[1] in range(self.size):
                next_state = copy.deepcopy(self)
                next_state.swap(empty_idx, swap_idx)
                adj.append(next_state)
        self._adj = adj
        random.shuffle(self._adj)
        return iter(adj)

    def cost(self, next_):
        assert(next_ in self.adjacent())
        return 1

    def taxicab_distance(self, r, c):
        """
        Taxicab distance from r,c to goal location of value at r,c
        Taxicabs cannot move diagonally (as opposed no straight line distance)
        """
        val = self.state[r][c]
        if val == 0:
            return 0
        val_home_row = val // self.size
        val_home_col = val % self.size
        return abs(val_home_row - r) + abs(val_home_col - c)

    def heuristic_manhattan(self):
        """
        Number of moves it would take to solve the
        puzzle if peices could move through each other

        This heuristic is admissable and consistent
        """
        dist = 0
        for r, c in itertools.product(range(self.size), repeat=2):
            dist += self.taxicab_distance(r, c)
        return dist

    def heuristic_displaced(self):
        """
        Number of peices that are out of place

        This heuristic is dominated by heuristic_manhattan, meaning that 
        heuristic_manhattan is always at least as good as this one.

        Use this to test that better heuristics result in lower branching factor
        """
        val = 0
        for r, c in itertools.product(range(self.size), repeat=2):
            if self.state[r][c] != 0:       # Ignore the empty spot
                if self.state[r][c] != r*self.size + c:
                    val += 1
        return val

    def heuristic_inconsistent(self):
        """
        Non-deterministic
        """
        best_h = self.heuristic_manhattan()
        return random.randint(0, best_h)

    def goal(self):
        for r, c in itertools.product(range(self.size), repeat=2):
            if self.state[r][c] != r*self.size + c:
                return False
        return True

    def __eq__(self, other):
        assert(self.size == other.size)
        return self.state == other.state

    def __hash__(self):
        return hash(tuple(tuple(inner) for inner in self.state))

    def __str__(self):
        max_int = self.size**2 - 1
        cell_size = len(str(max_int))
        ret = ''
        for row in self.state:
            for val in row:
                if val == 0:
                    ret += ' '*cell_size
                else:
                    ret += ' '*(cell_size - len(str(val)))
                    ret += str(val)
                ret += ' '
            ret += '\n'
        return ret.rstrip()



def measure_runtime(setup, func, verify=None, iterations=20):
    """
    setup() -> data
    func(data) -> result
    verify(result)
    """
    elapsed_time = 0
    for x in range(iterations):
        data = setup()
        # Experimential, force this thread off of the CPU just before the critical section of code.
        # This *should* reset the processes timer for how long it gets the CPU before a task switch.
        # This way when the CPU starts executing the critical (read: timed) section it has a full
        # timer and is less likely to get interrupted. 
        # I believe task switching is the cause of random (low probability) failures, where the run
        # time of a single call is several orders of magnitude greater than it usually is.
        time.sleep(0)
        start_time = time.process_time()
        result = func(data)
        elapsed_time += time.process_time() - start_time
        if verify is not None:
            verify(result)
    return elapsed_time



def measure_graph_size(vertexes, adjacent):
    """
    Returns |V| + |E|
    Does NOT find hidden nodes in the graph.
    """
    size = 0
    for v in vertexes:
        size += 1 + len(list(adjacent(v)))
    return size



class DepthFirstTraversal(unittest.TestCase):

    def test_depth_first_traversal(self):
        #
        # Build a test tree
        #
        depth = 10
        root = TreeNode(depth)

        #
        # Get all of the nodes
        #
        it = dft(root, TreeNode.adjacent)
        self.assertIsInstance(it, collections.abc.Iterator)
        all_nodes = list(it)

        #
        # Check all of the nodes
        #
        assert(all(isinstance(v, TreeNode) for v in all_nodes))
        assert(len(all_nodes) == root.subtree_size())
        self.assertIs(all_nodes[0], root)                # Check first element
        self.assertIn(all_nodes[1], root.adjacent())     # Check second element is adjacent to first element.

        #
        # Test all callbacks work
        #
        for n in all_nodes:
            # Make a flag for each callback
            n.pre = False
            n.post = False
            n.revisit = False
            n.visit = False     # Set when node is yielded to caller

        # Assert the correct flags and set them on each callback
        def pre(n):
            self.assertFalse(n.pre)
            self.assertFalse(n.post)
            self.assertFalse(n.revisit)
            self.assertFalse(n.visit)
            n.pre = True
        def post(n):
            self.assertTrue(n.pre)
            self.assertFalse(n.post)
            # Do not check revisit flag, can revisit an open vertex
            self.assertTrue(n.visit)
            n.post = True
            # Check that all adjacent have been visited
            for neighbor in n.adjacent():
                self.assertTrue(neighbor.visit)
        revisit_count = 0       # Count # times revisit is called
        def revisit(n):
            self.assertTrue(n.pre)
            # Do not check post flag, can revisit an open vertex
            # Do not check revisit flag, can revisit a vertex any number of times
            self.assertTrue(n.visit)
            n.revisit = True
            nonlocal revisit_count
            revisit_count += 1

        it = dft(root, TreeNode.adjacent, 
                preorder=pre, 
                postorder=post, 
                revisit=revisit)

        # Check for correct flags durring lazy operation
        result = []
        for n in it:
            self.assertTrue(n.pre)
            self.assertFalse(n.post)
            self.assertFalse(n.revisit)
            self.assertFalse(n.visit)
            result.append(n)
            n.visit = True

        # Check for correct flags after completion
        for n in all_nodes:
            self.assertTrue(n.pre)
            self.assertTrue(n.post)
            self.assertFalse(n.revisit)       # Its a tree, no revisits
            self.assertTrue(n.visit)

        #
        # Test revisit by linking every leaf back to the root
        #
        num_revisit_edges = 0
        for n in all_nodes:
            if not n.adj:
                n.adj.append(root)      # Make the circular connections
                num_revisit_edges += 1

        # Reset the flags
        for n in all_nodes:
            n.pre = False
            n.post = False
            n.revisit = False
            n.visit = False

        it = dft(root, TreeNode.adjacent, 
                 preorder=pre, 
                 postorder=post, 
                 revisit=revisit)
        for node in it:
            node.visit = True

        self.assertEqual(revisit_count, num_revisit_edges)
        # Check root was revisited
        self.assertTrue(all_nodes[0].revisit)
        # Check none of other nodes were revisited
        assert(not any(n.revisit for n in all_nodes[1:]))   


    def test_invalid_inputs(self):
        """
        Also check (by hand) that all of these raise sane error messages
        - Short stack trace
        - Clear indication of what went wrong and which argument caused it
        """
        # Try un-hashable start_vertex
        with self.assertRaises(TypeError):
            list(dft(unhashable_tree, simple_adjacent))

        # Try un-hashable inner vertex
        with self.assertRaises(TypeError):
            list(dft(simple_tree, hidden_unhashable))

        # Try invalid adjacent functions
        with self.assertRaises(TypeError):
            list(dft(simple_tree, None))

        with self.assertRaises(TypeError):
          list(dft(simple_tree, invalid_adjacent))

        # Try invalid preorder callbacks
        with self.assertRaises(TypeError):
            list(dft(simple_tree, simple_adjacent, preorder=4))

        # Try invalid postorder callbacks
        with self.assertRaises(TypeError):
            list(dft(simple_tree, simple_adjacent, postorder=99))

        # Try invalid revisit callbacks
        with self.assertRaises(TypeError):
            list(dft(UncleNode(6), UncleNode.adjacent, revisit=0.0))


    def test_time_complexity(self):
        """Test time complexity is linear."""
        # Average several runs for a consistent result
        # Compare all of the small input runs to all of the large input runs.
        # The per-vertex processing times should be constant (assuming algorithm is linear)
        def measure_runtime_per_vertex(size):
            setup       = lambda: TreeNode(size) 
            data_size   = TreeNode(size).subtree_size()
            func        = lambda root: list(dft(root, TreeNode.adjacent))
            run_time    = measure_runtime(setup, func)
            return run_time / data_size

        small_constant = measure_runtime_per_vertex(5)
        large_constant = measure_runtime_per_vertex(10)

        self.assertLess(small_constant / large_constant, 2)
        self.assertLess(large_constant / small_constant, 2)



class DepthFirstSearch(unittest.TestCase):

    def test_depth_first_search(self):
        # Test searching a graph
        # Find the number ten
        path = dfs(simple_tree, simple_adjacent, lambda v: v==10)
        self.assertIsInstance(path, tuple)
        self.assertIs(path[0], simple_tree)
        self.assertIs(path[1], simple_tree[1])
        self.assertIs(path[2], simple_tree[1][1])
        self.assertIs(path[3], 10)

    def test_cyclic(self):
        """Test DFS on cyclic graphs."""
        # Build a cyclic graph
        complex_graph = UncleNode(5)
        # Add some links back to the root
        for _ in range(100):
            complex_graph.random_leaf().adj.append(complex_graph)

        # Goal and correct path.
        p1   = complex_graph.adj[0]
        p2   = p1.adj[0]
        p3   = p2.adj[1]
        goal = p3.adj[0]

        path = dfs(complex_graph, UncleNode.adjacent, lambda v: v is goal)
        self.assertIs(path[0], complex_graph)
        self.assertIs(path[1], p1)
        self.assertIs(path[2], p2)
        self.assertIs(path[3], p3)
        self.assertIs(path[4], goal)

    def test_start_is_goal(self):
        path = dfs(simple_tree, simple_adjacent, lambda v: v is simple_tree)
        self.assertEqual(path, (simple_tree,))

    def test_no_path(self):
        # Test no path, acyclic
        with self.assertRaises(graph_algorithms.NoPathException):
            dfs(simple_tree, simple_adjacent, lambda v: v==99)

        # Test no path, cylcic
        complex_graph = UncleNode(5)
        # Add many random links back to the root.
        for _ in range(100):
            complex_graph.random_leaf().adj.append(complex_graph)

        with self.assertRaises(graph_algorithms.NoPathException):
            dfs(complex_graph, UncleNode.adjacent, lambda v: False)

    def test_invalid_inputs(self):
        """
        Also check (by hand) that all of these raise sane error messages
        - Short stack trace
        - Clear indication of what went wrong and which argument caused it
        """
        # Try un-hashable start_vertex
        with self.assertRaises(TypeError):
            dfs(unhashable_tree, simple_adjacent, simple_goal)

        # Try un-hashable inner vertex
        with self.assertRaises(TypeError):
            dfs(simple_tree, hidden_unhashable)

        # Try invalid adjacent functions
        with self.assertRaises(TypeError):
            dfs(simple_tree, None, simple_goal)

        with self.assertRaises(TypeError):
            dfs(simple_tree, invalid_adjacent, simple_goal)

        # Try invalid goal functions
        with self.assertRaises(TypeError):
            dfs(simple_tree, invalid_adjacent, None)

    def test_time_complexity(self):
        """Test time complexity is linear."""
        # Measure the run times with small inputs and large inputs. Find the average per-vertex
        # processing times for each group and compare them.  In theory they should be the same,
        # in practive they should be within an order of magnitude of each other.
        def measure_runtime_per_vertex(max_depth):
            def setup():
                root = UncleNode(max_depth)
                goal = root.random_leaf()
                return root, goal
            def func(args):
                root, goal = args
                return dfs(root, UncleNode.adjacent, lambda v: v is goal)
            run_time  = measure_runtime(setup, func)
            data_size = UncleNode(max_depth).subtree_size()
            return run_time / data_size

        small_constant = measure_runtime_per_vertex(5)
        large_constant = measure_runtime_per_vertex(10)
        
        self.assertLess(small_constant / large_constant, 2)
        self.assertLess(large_constant / small_constant, 2)



class IterativeDeepening(unittest.TestCase):

    def test_path(self):
        # Find the number ten
        path = iddfs(simple_tree, simple_adjacent, lambda v: v==10)
        self.assertEqual(len(path), 4)
        self.assertIsInstance(path, tuple)
        self.assertIs(path[0], simple_tree)
        self.assertIs(path[1], simple_tree[1])
        self.assertIs(path[2], simple_tree[1][1])
        self.assertIs(path[3], 10)


    def test_max_depth(self):
        # Check max depth
        path = iddfs(simple_tree, simple_adjacent, lambda v: v==10, max_depth=4)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 4)
        with self.assertRaises(graph_algorithms.NoPathException):
            iddfs(simple_tree, simple_adjacent, lambda v: v==10, max_depth=3)


    def test_find_start_vertex(self):
        # Check finding start-vertex
        short_path = iddfs(simple_tree, simple_adjacent, lambda v: v is simple_tree)
        self.assertEqual(short_path, [simple_tree])


    def test_unreachable(self):
        # Check search when the goal is unreachable
        with self.assertRaises(graph_algorithms.NoPathException):
            iddfs(simple_tree, simple_adjacent, lambda v: False)

        # Test it again with a more complicated graph
        with self.assertRaises(graph_algorithms.NoPathException):
            iddfs(UncleNode(3), UncleNode.adjacent, lambda v: False, max_depth=10)


    def test_optimal(self):
        # Check finding the sortest path
        path_length = 10
        root = UncleNode(path_length)
        for test in range(5):
            goal = root.random_leaf()
            path = iddfs(root, UncleNode.adjacent, lambda v: v is goal)
            self.assertEqual(len(path), path_length)


    def test_invalid_inputs(self):
        # Try invalid adjacent
        with self.assertRaises(TypeError):
            iddfs(simple_tree, None, simple_goal)

        with self.assertRaises(TypeError):
            iddfs(simple_tree, invalid_adjacent, simple_goal)

        # Try invalid goal
        with self.assertRaises(TypeError):
            iddfs(simple_tree, simple_adjacent, None)

        # Try invalid max_depth
        with self.assertRaises(TypeError):
            iddfs(simple_tree, simple_adjacent, simple_goal, max_depth = -1)

        with self.assertRaises(TypeError):
            iddfs(simple_tree, simple_adjacent, simple_goal, max_depth = 'not an integer')


    def test_time_complexity(self):
        """
        Test time complexity

        Measure the run times with small inputs and large inputs.

        Execution time = k * b^p
            where b = branching factor
                  p = path length
                  k = some constant

        Calculate the 'k' that makes this formula work.
        Verify that k is constant with respect to input size.
        """
        def measure_runtime_per_vertex(max_depth):
            def setup():
                root = TreeNode(max_depth)
                goal = root.random_leaf()
                return root, goal
            def func(args):
                root, goal = args
                return iddfs(root, type(root).adjacent, lambda v: v is goal)
            run_time  = measure_runtime(setup, func)
            data_size = TreeNode(max_depth).subtree_size()
            return run_time / data_size

        small_constant = measure_runtime_per_vertex(5)
        large_constant = measure_runtime_per_vertex(10)
        
        self.assertLess(small_constant / large_constant, 2)
        self.assertLess(large_constant / small_constant, 2)



def assertNoWarnings(test_case):
    def wrapped_method(self):
        # Capture all logged warnings & errors
        buff    = io.StringIO()
        handler = logging.StreamHandler(buff)
        handler.setLevel('WARNING')
        logger = logging.getLogger(graph_algorithms.__name__)
        logger.addHandler(handler)

        try:
            test_case(self)
        finally:
            # Get logged messages
            handler.flush()            # Push all messages into the buffer
            log = buff.getvalue()
            # remove test apparatus
            buff.close()
            logger.removeHandler(handler)

        # Raise warnings on all handlers
        self.assertFalse(log)

    return wrapped_method



class AStar(unittest.TestCase):

    random_iterations = 10

    def assertPuzzlePath(self, start, path):
        '''Test that the given path is valid
        Assumes that the path is a sliding puzzle'''
        self.assertIsNotNone(path)
        self.assertIsInstance(path, tuple)
        self.assertIs(path[0], start)
        self.assertTrue(path[-1].goal())
        for vert, next_vert in zip(path, path[1:]):
            self.assertIn(vert, next_vert.adjacent())

    @assertNoWarnings
    def test_complete(self):
        # Test that A* always finds a path, eventually
        for random_test in range(AStar.random_iterations):
            start = SlidingPuzzle(3)
            path = a_star(start, 
                          SlidingPuzzle.adjacent, 
                          SlidingPuzzle.cost, 
                          SlidingPuzzle.heuristic_manhattan,
                          SlidingPuzzle.goal,)
            self.assertPuzzlePath(start, path)


    @assertNoWarnings
    def test_optimal_fast(self):
        for random_test in range(AStar.random_iterations):
            depth = 6
            start = UncleNode(depth)
            goal  = start.random_leaf()
            path  = a_star(start,
                           UncleNode.adjacent,
                           lambda v1, v2: 1,
                           lambda v: v.depth-1,     # Not a great heuristic
                           lambda v: v is goal,)
            self.assertIsNotNone(path)
            self.assertEqual(len(path), depth)
            self.assertIs(path[0], start)
            self.assertIs(path[-1], goal)
            for vert, next_vert in zip(path, path[1:]):
                self.assertIn(next_vert, vert.adjacent())


    @assertNoWarnings
    def test_optimal_slow(self):
        for random_test in range(AStar.random_iterations):
            puzzle = SlidingPuzzle(3)
            uniform_cost_path = a_star(puzzle, 
                                       SlidingPuzzle.adjacent, 
                                       SlidingPuzzle.cost, 
                                       lambda v: 0,
                                       SlidingPuzzle.goal,)
            self.assertPuzzlePath(puzzle, uniform_cost_path)

            a_star_path = a_star(puzzle, 
                                 SlidingPuzzle.adjacent, 
                                 SlidingPuzzle.cost, 
                                 SlidingPuzzle.heuristic_manhattan,
                                 SlidingPuzzle.goal,)
            self.assertPuzzlePath(puzzle, a_star_path)

            self.assertEqual(len(uniform_cost_path), len(a_star_path))


    @assertNoWarnings
    def test_find_start_vertex(self):
        solved = SlidingPuzzle(3)
        solved.force_solve()
        trivial_path = a_star(solved, 
                              SlidingPuzzle.adjacent, 
                              SlidingPuzzle.cost, 
                              SlidingPuzzle.heuristic_manhattan,
                              SlidingPuzzle.goal,)
        self.assertEqual(trivial_path, tuple([solved]))


    @assertNoWarnings
    def test_unreachable_finite(self):
        # Test what happens when a finite graph is exausted (all visited, no goal)
        with self.assertRaises(graph_algorithms.NoPathException):
            a_star(UncleNode(6),
                   UncleNode.adjacent,
                   lambda v1, v2: 1,
                   lambda v: v.depth-1,
                   lambda v: False)


    @assertNoWarnings
    def test_unreachable_infinite(self):
        '''Infinite graph which isn't solvable
        Average shortest path is for sliding 8 puzzle is 22 moves
        '''
        unsolvable = SlidingPuzzle(3, solvable=False)
        with self.assertRaises(graph_algorithms.NoPathException):
            a_star(unsolvable, 
                   SlidingPuzzle.adjacent, 
                   SlidingPuzzle.cost, 
                   SlidingPuzzle.heuristic_manhattan,
                   SlidingPuzzle.goal,
                   max_depth = 22*5)


    @assertNoWarnings
    def test_max_depth(self):
        puzzle = SlidingPuzzle(3)
        path = a_star(puzzle, 
                      SlidingPuzzle.adjacent, 
                      SlidingPuzzle.cost, 
                      SlidingPuzzle.heuristic_manhattan,
                      SlidingPuzzle.goal,)
        self.assertPuzzlePath(puzzle, path)

        with self.assertRaises(graph_algorithms.NoPathException):
            a_star(puzzle, 
                   SlidingPuzzle.adjacent, 
                   SlidingPuzzle.cost, 
                   SlidingPuzzle.heuristic_manhattan,
                   SlidingPuzzle.goal,
                   max_depth=len(path)-1)


    def test_inadmissible_heuristic(self):
        for random_test in range(AStar.random_iterations):
            for factor in (1.1, 1.2, 1.5, 2, 3, 4, 5, 10, 25, 50, 100):
                start = SlidingPuzzle(3)
                with self.assertLogs('graph_algorithms', 'WARNING') as log:
                    path = a_star(start, 
                                  SlidingPuzzle.adjacent, 
                                  SlidingPuzzle.cost, 
                                  lambda v: factor * v.heuristic_manhattan(),
                                  SlidingPuzzle.goal,)

                # Test logged messages
                # This will also warn of inconsistent heuristic, which is OK
                warning = "WARNING:graph_algorithms:Detected inadmissible heuristic"
                self.assertIn(warning, log.output)

                # Test complete with inadmissible heuristic
                self.assertPuzzlePath(start, path)

                # Test the upper bound on sub-optimality
                optimal = a_star(start, 
                                 SlidingPuzzle.adjacent, 
                                 SlidingPuzzle.cost, 
                                 SlidingPuzzle.heuristic_manhattan,
                                 SlidingPuzzle.goal,)

                self.assertLessEqual(len(path), len(optimal) * factor)


    def test_inconsistent_heuristic(self):
        for random_test in range(AStar.random_iterations):
            start = SlidingPuzzle(3)
            with self.assertLogs('graph_algorithms', 'WARNING') as log:
                path = a_star(start, 
                              SlidingPuzzle.adjacent, 
                              SlidingPuzzle.cost, 
                              SlidingPuzzle.heuristic_inconsistent,
                              SlidingPuzzle.goal,)

            # Test complete with inconsistent heuristic
            self.assertPuzzlePath(start, path)

            # Test logged messages
            warning = "WARNING:graph_algorithms:Detected inconsistent heuristic"
            self.assertEqual(log.output, [warning])


    @assertNoWarnings
    def test_branching_factor(self):
        # Test two heuristics and verify their relationship with the branching factor
        for random_test in range(AStar.random_iterations):
            start = SlidingPuzzle(3)
            with self.assertLogs('graph_algorithms', 'DEBUG') as log:
                path = a_star(start,
                              SlidingPuzzle.adjacent, 
                              SlidingPuzzle.cost, 
                              SlidingPuzzle.heuristic_displaced,
                              SlidingPuzzle.goal,)

            # Test logged messages
            self.assertEqual(len(log.output), 1)
            ebf_msg = log.output[0]
            self.assertRegex(ebf_msg, r'Effective branching factor \d+')
            disp_ebf = float(ebf_msg.split()[-1])

            with self.assertLogs('graph_algorithms', 'DEBUG') as log:
                path = a_star(start,
                              SlidingPuzzle.adjacent, 
                              SlidingPuzzle.cost, 
                              SlidingPuzzle.heuristic_manhattan,
                              SlidingPuzzle.goal,)

            # Test logged messages
            self.assertEqual(len(log.output), 1)
            ebf_msg = log.output[0]
            self.assertRegex(ebf_msg, r'Effective branching factor \d+')
            manhat_ebf = float(ebf_msg.split()[-1])

            # Test better heuristics yield lower branching factor
            self.assertLessEqual(manhat_ebf, disp_ebf)
            # print("disp ebf:", disp_ebf, "manhat ebf:", manhat_ebf)


    @assertNoWarnings
    def test_no_numpy_branching_factor(self):
        '''Test A* works even if numpy is not installed'''
        try:
            np = graph_algorithms.numpy
            graph_algorithms.numpy = None
            start = SlidingPuzzle(3)
            path = a_star(start, 
                          SlidingPuzzle.adjacent, 
                          SlidingPuzzle.cost, 
                          SlidingPuzzle.heuristic_manhattan,
                          SlidingPuzzle.goal,)
            self.assertPuzzlePath(start, path)
        finally:
            graph_algorithms.numpy = np


    @assertNoWarnings
    def test_invalid_inputs(self):
        """
        Check (by hand) that all of these raise sane error messages
        - Short stack trace
        - Clear indication of what went wrong and which argument caused it
        """
        # Try unhashable start_vertex
        with self.assertRaises(TypeError):
            a_star(unhashable_tree, 
                 simple_adjacent, 
                 simple_cost, 
                 simple_heuristic,
                 simple_goal,)

        # Try unhashable inner vertex
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 hidden_unhashable, 
                 simple_cost, 
                 simple_heuristic,
                 simple_goal,)

        # Try un-callable adjacent
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 'uncallable-adjacent-function', 
                 simple_cost, 
                 simple_heuristic,
                 simple_goal,)

        # Try adjacent returning not an iterable
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 invalid_adjacent, 
                 simple_cost, 
                 simple_heuristic,
                 simple_goal,)

        # Try un-callable cost
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 'uncallable-cost-function', 
                 simple_heuristic,
                 simple_goal,)

        # Try cost returning not a value
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 lambda v1, v2: None,
                 simple_heuristic,
                 simple_goal,)

        # Try un-callable heuristic
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 simple_cost,
                 'uncallable-heuristic-function',
                 simple_goal,)

        # Try heuristic returning not a value
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 simple_cost,
                 lambda v: None,
                 simple_goal,)

        # Try un-callable goal
        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 simple_cost,
                 simple_heuristic,
                 'uncallable-goal-function',)

        # Try invalid max_depth
        with self.assertRaises(ValueError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 simple_cost,
                 simple_heuristic,
                 simple_goal,
                 max_depth = 0)

        with self.assertRaises(ValueError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 simple_cost,
                 simple_heuristic,
                 simple_goal,
                 max_depth = -99)

        with self.assertRaises(TypeError):
            a_star(simple_tree, 
                 simple_adjacent, 
                 simple_cost,
                 simple_heuristic,
                 simple_goal,
                 max_depth = 'nan')


    def test_negative_cost_cycles(self):
        # Force negative cycles in a circle graph.
        circle = ringGraph(1000)
        with self.assertRaises(CyclicGraphException) as exception:
            a_star(
                circle[0],          # Start vertex
                Node.adjacent,
                lambda x, y: -1,    # Cost
                lambda x: 0,        # Heuristic
                lambda x: False)    # Goal
        self.assertRegex(str(exception.exception), r".*[Nn]egative.*")
        self.assertRegex(str(exception.exception), r".*[Cc]ost.*")
        self.assertRegex(str(exception.exception), r".*[Cc]ycle.*")

        # Force negative cycles in a sliding puzzle.
        # Link the goal back to the start with a large negative cost.
        puzzle = SlidingPuzzle(3)
        def adj(x):
            if x.goal():
                return [puzzle]
            return x.adjacent()
        def cost(x, y):
            if x.goal() and y is puzzle:
                return -99999
            return x.cost(y)
        with self.assertRaises(CyclicGraphException) as exception:
            a_star(puzzle, 
                   adj, 
                   cost, 
                   SlidingPuzzle.heuristic_manhattan,
                   lambda x: False)  # Goal
        self.assertRegex(str(exception.exception), r".*[Nn]egative.*")
        self.assertRegex(str(exception.exception), r".*[Cc]ost.*")
        self.assertRegex(str(exception.exception), r".*[Cc]ycle.*")


    @unittest.skip('Unimplemented test case')
    @assertNoWarnings
    def test_time_complexity(self):
        #
        # Calculate the time complexity formula
        # Compare experimential results to the formula
        #
        pass



class TopologicalSort(unittest.TestCase):
    random_iterations = 20

    def assertTopological(self, topo_sort_result, adjacent):
        for i, v in enumerate(topo_sort_result):
            for adj in adjacent(v):
                self.assertNotIn(adj, topo_sort_result[:i+1])

    def verify(self, topo_sort_result, input_graph, adjacent):
        self.assertIsInstance(topo_sort_result, tuple)
        self.assertTopological(topo_sort_result, adjacent)
        # Check that result contains all of input graph.
        self.assertSetEqual(set(topo_sort_result), set(input_graph))


    def test_constant_acyclic_graphs(self):
        result = topological_sort([simple_tree], simple_adjacent)
        self.verify(result, dft(simple_tree, simple_adjacent), simple_adjacent)

        graph = TreeNode(5)
        result = topological_sort([graph], TreeNode.adjacent)
        self.verify(result, dft(graph, TreeNode.adjacent), TreeNode.adjacent)

        graph = TreeNode(12)
        result = topological_sort([graph], TreeNode.adjacent)
        self.verify(result, dft(graph, TreeNode.adjacent), TreeNode.adjacent)


    def test_random_acyclic_graphs(self):
        # Also test that the graph input can be an iterator.
        for random_test in range(self.random_iterations):
            graph = randomAcyclicGraph(1000)  # This returns a list of all nodes in the graph.
            result = topological_sort(iter(graph), Node.adjacent)
            self.verify(result, graph, Node.adjacent)


    def test_cyclic_graphs(self):
        with self.assertRaises(graph_algorithms.CyclicGraphException):
            graph = UncleNode(4)
            topological_sort([graph], UncleNode.adjacent)

        for random_test in range(self.random_iterations):
            with self.assertRaises(graph_algorithms.CyclicGraphException):
                graph = SlidingPuzzle(3)
                topological_sort([graph], SlidingPuzzle.adjacent)


    def test_invalid_inputs(self):
        graph = randomAcyclicGraph(1000)

        # Test invalid graph (not an iterable)
        with self.assertRaises(TypeError):
            topological_sort(graph[0], Node.adjacent)

        # Test invalid adjacent (not callable)
        with self.assertRaises(TypeError):
            topological_sort(graph, 'invalid-adjacent')

        # Test invalid adjacent (returns not an iterable)
        with self.assertRaises(TypeError):
            topological_sort(graph, invalid_adjacent)

        # Test invalid vertexes (not hashable)
        with self.assertRaises(TypeError):
            topological_sort(unhashable_tree, simple_adjacent)

        # Test invalid vertexes (inner vertex not hashable)
        with self.assertRaises(TypeError):
            topological_sort(hidden_unhashable, simple_adjacent)


    def test_time_complexity(self):
        """Topological sort should be linear."""
        def measure_runtime_per_vertex(data_size):
            graph = None
            def setup():
                nonlocal graph
                graph = randomAcyclicGraph(data_size)
                return graph
            func       = lambda graph: (topological_sort(graph, Node.adjacent), graph)
            run_time   = measure_runtime(setup, func)
            graph_size = measure_graph_size(graph, Node.adjacent)
            return run_time / graph_size

        small_constant = measure_runtime_per_vertex(50)
        large_constant = measure_runtime_per_vertex(500)

        self.assertLess(small_constant / large_constant, 2)
        self.assertLess(large_constant / small_constant, 2)



class StronglyConnectedComponents(unittest.TestCase):
    random_iterations = 20

    def assert_connected_components(self, connected_components, adjacent):
        """Asserts the results are connected components"""
        # Check that every vertex can reach every other vertex in its CC.
        for cc in connected_components:
            for start, end in itertools.combinations(cc, 2):
                # Raises exception if there is no path.
                dfs(start, adjacent, lambda v: v is end)
                dfs(end, adjacent, lambda v: v is start)

        # Assert no bidirectional paths between connected components.
        for a_cc, b_cc in itertools.combinations(connected_components, 2):
            a = next(iter(a_cc))
            b = next(iter(b_cc))
            with self.assertRaises(NoPathException):
                dfs(a, adjacent, lambda v: v is b)
                dfs(b, adjacent, lambda v: v is a)


    def run_and_check(self, graph, adj, check_for_hidden=False):
        # Check that algorithm works with single use iterator of input graph.
        result = scc(iter(graph), adj)
        
        # Check return type
        self.assertIsInstance(result, frozenset)
        for cc in result:
            self.assertIsInstance(cc, frozenset)

        if check_for_hidden:
            # Find all internal nodes.
            all_nodes = set()
            for node in graph:
                all_nodes.update(dft(node, adj))
        else:
            all_nodes = set(graph)

        # Check that the result is the same as the input graph.
        self.assertSetEqual(all_nodes, set(itertools.chain.from_iterable(result)))

        # Check the result is correct.
        self.assert_connected_components(result, adj)
        return result


    def test_forest(self):
        """Not all reachable nodes included in input graph"""
        forest = []
        for _ in range(3):
            forest.append(TreeNode(3))
            forest.append(UncleNode(3))
        self.run_and_check(forest, UncleNode.adjacent, check_for_hidden=True)

        forest = []
        for _ in range(5):
            forest.append(TreeNode(5))
            forest.append(UncleNode(5))
        self.run_and_check(forest, UncleNode.adjacent, check_for_hidden=True)


    def test_random_connected_components(self):
        for random_test in range(self.random_iterations):
            graph = []
            for make_a_tree in range(3):
                graph.extend(randomAcyclicGraph(10))
                graph.extend(randomGraph(10, .2))
            self.run_and_check(graph, Node.adjacent)


    def test_no_connected_components(self):
        """All nodes unconnected"""
        graph = TreeNode(7)
        self.run_and_check([graph], TreeNode.adjacent, check_for_hidden=True)
        
        for iteration in range(self.random_iterations):
            graph = randomAcyclicGraph(100)
            self.run_and_check(graph, Node.adjacent)
        
        for iteration in range(self.random_iterations):
            graph = randomGraph(100, 0)
            self.run_and_check(graph, Node.adjacent)


    def test_one_connected_component(self):
        """All nodes connected, all to all"""
        # Not connected all to all, root is its own component.
        graph = UncleNode(7)
        self.run_and_check([graph], UncleNode.adjacent, check_for_hidden=True)

        # This grpah has one strongly connected component.
        graph = TreeNode(7)
        leaves = []
        for node in dft(graph, TreeNode.adjacent):   # Find all leaves on this tree.
            if len(node.adj) == 0:
                leaves.append(node)
        for node in leaves:         # Connect every leaf to the root.
            node.adj.append(graph)
        result = self.run_and_check([graph], TreeNode.adjacent, check_for_hidden=True)
        self.assertEqual(len(result), 1)
        scc = next(iter(result))
        self.assertEqual(len(scc), graph.subtree_size())

        # Not connected all to all, instead is a dense graph.
        for iteration in range(self.random_iterations):
            graph = randomGraph(100, 1.0)
            self.run_and_check(graph, Node.adjacent)


    def test_connected_path(self):
        """All nodes connected in a circular chain."""
        graph  = ringGraph(10)
        result = self.run_and_check(graph, Node.adjacent)
        self.assertEqual(len(result), 1)

        graph  = ringGraph(100)
        result = self.run_and_check(graph, Node.adjacent)
        self.assertEqual(len(result), 1)


    def test_invalid_inputs(self):
        graph = randomGraph(1000, .3)

        # Test invalid graph (not an iterable)
        with self.assertRaises(TypeError):
            scc(graph[0], Node.adjacent)

        # Test invalid adjacent (not callable)
        with self.assertRaises(TypeError):
            scc(graph, 'invalid-adjacent')

        # Test invalid adjacent (returns not an iterable)
        with self.assertRaises(TypeError):
            scc(graph, invalid_adjacent)

        # Test invalid vertexes (not hashable)
        with self.assertRaises(TypeError):
            scc(unhashable_tree, simple_adjacent)

        # Test invalid vertexes (inner vertex not hashable)
        with self.assertRaises(TypeError):
            scc(hidden_unhashable, simple_adjacent)


    def test_time_complexity(self):
        """Strongly connected components should be linear."""
        def measure_runtime_per_vertex(data_size):
            graph = None
            def setup():
                nonlocal graph
                graph = randomGraph(data_size, 0.3)
                return graph
            func      = lambda graph: scc(graph, Node.adjacent)
            run_time  = measure_runtime(setup, func)
            real_size = measure_graph_size(graph, Node.adjacent)
            return run_time / real_size

        small_constant = measure_runtime_per_vertex(50)
        large_constant = measure_runtime_per_vertex(500)

        self.assertLess(small_constant / large_constant, 2)
        self.assertLess(large_constant / small_constant, 2)


@unittest.skip("Unimplemented Function")
class MinimumSpanningTree(unittest.TestCase):
    """
    Test Strategy:
    numpy has Kruskals algorithm, which I can use as a reference.
    
    I'm going to need new data sets to test this with...
    - Undirected
    - Randomly Generated

    Notes on MSTs:
    - If all edge weights are unique then the MST is also unique.
    - The value of the minimum spanning tree is the minimum even if the trees are not unique.
    
    """

    def run_and_check(self, vertexes, adjacent, cost):
        result = minimum_spanning_tree(vertexes, adjacent, cost)



    # FIXME: THIS CAN NOT BE CORRECT! IT DOES NOT USE THE RESULT....
    def value(self, vertexes, result):
        """Returns the value of the given minimum spanning tree."""
        value = 0
        for v in vertexes:
            for a in n.adjacent():
                value += v.cost(a)
        return value / 2  # Divide by two because all edges are seen twice, once from each end.

    def minimum(self, graph):
        pass
        # Convert this graph into the format used by the refence implementation.
        # Check that 


    def test_test(self):
        g, f = WeightedNode.random_graph(100)
        #print(self.value(graph)
        #for n in g:
        #    print("Node: "+str(id(n)))
        #    print("COSTS: \n\t"+'\n\t'.join(str(id(a))+":"+str(f(n,a)) for a in n.adjacent()))

    def test_circle(self):
        assert(False)

    def test_tree(self):
        assert(False)

    def test_random_graphs(self):
        assert(False)

    def test_invalid_inputs(self):
        assert(False)

    def test_time_complexity(self):
        assert(False)



if __name__ == '__main__':
    unittest.main()
