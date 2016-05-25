#!/usr/bin/python3
'''
depth_first_traversal.py
'''
from collections.abc import Hashable


def depth_first_traversal(
        start_vertex,
        adjacent,
        preorder  = lambda vertex:None,
        postorder = lambda vertex:None,
        revisit   = lambda vertex:None,):
    """
    Depth first traversal of a graph
    Returns an iterator over traversed objects.
    Returned iterator is evaluated lazily.

    adjacent(vertex) -> iterable-of-vertexes
    Where a vertex can be any hashable python object.
    The adjacent list is evaluated lazily.

    A vertex is 'visited' when it is yielded to the caller as part of the result.

    There are three optional callbacks:
    preorder(vertex)
        Called before visiting every vertex.
    postorder(vertex)
        Called after all of adjacent(vertex) have been visited.
    revisit(vertex)
        Called when when two branches of the search tree touch.
        Called when vertex is seen and no other callbacks are called.

    Complexity:
    time  = O(v+e)
    space = O(v)
        where v = number of vertexes
              e = number of edges
    """
    if not callable(preorder):  raise ValueError('Argument "preorder" is not callable')
    if not callable(postorder): raise ValueError('Argument "postorder" is not callable')
    if not callable(revisit):   raise ValueError('Argument "revisit" is not callable')
    if not isinstance(start_vertex, Hashable):
        raise ValueError('Argument "start_vertex" is not hashable, all vertexes must be hashable')

    preorder(start_vertex)
    yield start_vertex
    # frontier = [(v0, adjacent(v0)), (v1, adjacent(v1)), ... (vN, adjacent(vN))]
    # FILO/Stack
    frontier = [(start_vertex, iter(adjacent(start_vertex)))]
    # Seen does not include the frontier's neighbors
    seen = set([start_vertex])

    while frontier:
        parent, adj_iter = frontier[-1]
        try:
            vertex = next(adj_iter)
        except StopIteration:
            postorder(parent)
            frontier.pop()
        else:
            if vertex in seen:
                revisit(vertex)
            else:
                preorder(vertex)
                seen.add(vertex)
                yield vertex
                frontier.append((vertex, iter(adjacent(vertex))))



def unit_tests():
    import collections.abc
    import time

    #
    # Test some invalid inputs
    #
    unhashable = [[[1,2], [3,4]], [[5,6], [7,8]]]
    try:
        list(depth_first_traversal(unhashable, lambda s: s))
    except ValueError as e:
        pass
    else:
        assert(False)

    hashable = (((1,2), (3,4)), ((5,6), (7,8)))
    list(depth_first_traversal(hashable, lambda s: s if type(s)==tuple else []))
    try:
        list(depth_first_traversal(hashable, lambda s: s, 99))
    except ValueError as e:
        pass
    else:
        assert(False)

    #
    # Build a tree of depth 'n' using depth_first_traversals
    #
    n = 6
    class TreeNode():
        def __init__(self):
            self.depth = 1
        def adjacent(self):
            if hasattr(self, 'left'):
                return self.left, self.right
            return []

    def preorder_builder(node):     # Builds the tree (recursively, from the callback)
        if node.depth < n:
            node.left = TreeNode()
            node.left.depth = node.depth + 1
            node.right = TreeNode()
            node.right.depth = node.depth + 1

    it = depth_first_traversal(TreeNode(), TreeNode.adjacent, preorder=preorder_builder)

    # Check the result
    assert(isinstance(it, collections.abc.Iterator))
    all_nodes = list(it)

    # Check the tree
    assert(all(isinstance(n, TreeNode) for n in all_nodes))
    assert(len(all_nodes) == 2**n - 1)

    # Test all callbacks work
    for n in all_nodes:     # Make a flag for each callback
        n.pre = False
        n.post = False
        n.revisit = False
    def pre(n):             # Assert the correct flags and set them on each callback
        assert(not n.pre)
        assert(not n.post)
        assert(not n.revisit)
        n.pre = True
    def post(n):
        assert(n.pre)
        assert(not n.post)
        # Do not check revisit flag, can revisit an open vertex
        n.post = True
    revisit_count = 0       # Count # times revisit is called
    def revisit(n):
        assert(n.pre)
        # Do not check post flag, can revisit an open vertex
        # Do not check revisit flag, can revisit a vertex any number of times
        n.revisit = True
        nonlocal revisit_count
        revisit_count += 1

    it = depth_first_traversal(all_nodes[0], TreeNode.adjacent, 
                                            preorder=pre, 
                                            postorder=post, 
                                            revisit=revisit)
    result = []
    for n in it:
        assert(n.pre)           # assert the correct flags durring lazy operation
        assert(not n.post)
        assert(not n.revisit)
        result.append(n)

    # verify all_nodes[0] was the tree root
    assert(result == all_nodes)

    for n in all_nodes:
        assert(n.pre)
        assert(n.post)
        assert(not n.revisit)       # Its a tree, no revisits

    #
    # Test revisit by linking every node back to the root
    #
    num_revisit_edges = 0
    for n in all_nodes:
        if not hasattr(n, 'left'):      # Make the circular connections
            n.left = all_nodes[0]
            n.right = all_nodes[0]
            num_revisit_edges += 2
    for n in all_nodes:     # Reset the flags
        n.pre = False
        n.post = False
        n.revisit = False

    list(depth_first_traversal(
                all_nodes[0], 
                TreeNode.adjacent, 
                preorder=pre, 
                postorder=post, 
                revisit=revisit))
    assert(revisit_count == num_revisit_edges)
    assert(all_nodes[0].revisit)                        # Check first node was revisited
    assert(not any(n.revisit for n in all_nodes[1:]))   # Check none of other nodes were revisited

    #
    # Test time complexity
    # Average several runs for a consistent result
    # Compare all of the small input runs to all of the large input runs.
    # The per-vertex processing times should be constant (if algorithm is linear)
    #
    small_times  = 0
    small_inputs = 0
    for n in range(4, 9):
        # Build a tree of size 'n'
        tree = list(depth_first_traversal(TreeNode(), TreeNode.adjacent, preorder=preorder_builder))

        # Traverse the whole tree
        start_time = time.time()
        result = list(depth_first_traversal(tree[0], TreeNode.adjacent))
        small_times  += time.time() - start_time
        small_inputs += len(result)
    small_constant = small_times / small_inputs

    large_times  = 0
    large_inputs = 0
    for n in range(12, 17):
        # Build a tree of size 'n'
        tree = list(depth_first_traversal(TreeNode(), TreeNode.adjacent, preorder=preorder_builder))

        # Traverse the whole tree
        start_time = time.time()
        result = list(depth_first_traversal(tree[0], TreeNode.adjacent))
        large_times  += time.time() - start_time
        large_inputs += len(result)
    large_constant = large_times / large_inputs

    assert(small_constant / large_constant < 10)
    assert(large_constant / small_constant < 10)



if __name__ == '__main__':
    unit_tests()
