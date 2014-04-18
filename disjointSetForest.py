""" Implementation of a disjoint set forest data structure for union-find purposes.
"""
import array

class DisjointSetForest:
    def __init__(self, nbElements):
        """ Initialize a disjoint set forest for a given number of elements.
        Args:
            nbElements (int): the number of elements the forest contains.
        """
        # initialize the various arrays we'll need.
        self.forest = array.array('i', range(0,nbElements))
        self.rank = array.array('i', [0] * nbElements)
        self.nbComponents = nbElements
        self.compSize = array.array('i', [1] * nbElements)

    def find(self, i):
        """ Finds the representant of the set containing i.
        Args:
            i (int): element you want to find the set representant of.
        Returns:
            The set representant of i.
        """
        if self.forest[i] != i:
            self.forest[i] = self.find(self.forest[i])
        return self.forest[i]

    def union(self, i, j):
        """ Fuses the sets containing i and j into one.
        Args:
           i (int): element of the first set you want to fuse.
           j (int): element of the second set you want to fuse.
        Returns:
           The root of the new fused set.
        """
        iRoot = self.find(i)
        jRoot = self.find(j)
        if iRoot == jRoot:
            return jRoot
        # if the roots are different, fusion happen and we
        # have one less component.
        self.nbComponents -= 1
        newRoot = None
        if self.rank[iRoot] < self.rank[jRoot]:
            self.forest[iRoot] = jRoot
            newRoot = jRoot
        elif self.rank[iRoot] > self.rank[jRoot]:
            self.forest[jRoot] = iRoot
            newRoot = iRoot
        else:
            self.forest[jRoot] = iRoot
            self.rank[iRoot] += 1
            newRoot = iRoot
        # update the size of the new component, keep track of the largest as well.
        self.compSize[newRoot] = self.compSize[jRoot] + self.compSize[iRoot]
        return newRoot
