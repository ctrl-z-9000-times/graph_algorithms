#!/usr/bin/python3
'''
Common Graph Algorithms Library

This is a library of common graph algorithms with a functional API so that it 
can directly work with arbitrary python data structures.

Currently Implemented:
- depth_first_traversal()       A lazy depth first traversal
- depth_first_search()          A depth first search
'''
__version__ = "0.0.0",


if __name__ == '__main__':
    print('Start unit tests (silent pass)')

    import depth_first_traversal
    depth_first_traversal.unit_tests()

    import depth_first_search
    depth_first_search.unit_tests()

else:
    from common_algorithms.depth_first_traversal import depth_first_traversal
    from common_algorithms.depth_first_search import depth_first_search
