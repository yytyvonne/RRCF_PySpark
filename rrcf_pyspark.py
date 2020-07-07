# importing required libraries
from __future__ import print_function
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession

from collections import deque
import numpy as np
import sys

class Branch:
    
    __slots__ = __slots__ = ['cut_dim', 'cut_val', 'left', 'right', 'parent', 'n', 'bounding_box']

    def __init__(self, cut_dim, cut_val, left=None, right=None, parent=None, n=0, bounding_box=None):
        self.left = left #pointer to left child
        self.right = right #pointer to right child 
        self.parent = parent #pointer to parent
        self.n = n #no. of leaves under branch
        self.cut_dim = cut_dim #dimension to cut over 
        self.cut_val = cut_val #value of cut        
        self.bounding_box = bounding_box #bounding box of points under branch [2*d]

    def __repr__(self):
        return "Branch(cut_dim={}, cut_val={:.2f})".format(self.cut_dim, self.cut_val)

class Leaf:
    
    __slots__ = ['index', 'depth', 'parent', 'x', 'n', 'bounding_box']

    def __init__(self, index, depth=None, parent=None, x=None, n=1):
        self.parent = parent #pointer to parent  
        self.index = index #index of leaf
        self.depth = depth #depth of leaf 
        self.x = x #data  
        self.n = n #no. of points in leaf
        self.bounding_box = x.reshape(1, -1) #data [min,max]

    def __repr__(self):
        return "Leaf({0})".format(self.index)


class RRCTree(object):

    def __init__(self, X=None, seed=1234, index=None):
        '''X: np.array([n,d]); data
           seed: random state seed
           index: label of n data points in X'''
        np.random.seed(seed)

        self.leaves = {}
        self.root = None
        if X is not None:
            X = np.around(X, decimals = 10)
            if index is None:
                index = np.arange(X.shape[0], dtype = int)
            self.index_labels = index
            unique, I, N = np.unique(X, return_inverse=True, return_counts=True,axis=0)
            if N.max() > 1:
                n, d = unique.shape
                X = unique
            else:
                n, d = X.shape
                N = np.ones(n, dtype=np.int)
                I = None  
            self.ndim = d
            self.parent = None #parent node pointer
            S = np.ones(X.shape[0], dtype=np.bool)
            self.make_tree(X, S, N, I, parent=self)
            self.root.parent = None 
            self.count_leaves(self.root)
            self.get_bounding_box(self.root)
            
    def count_leaves(self, node): 
        if isinstance(node, Branch):
            if node.left:
                self.count_leaves(node.left)
            if node.right:
                self.count_leaves(node.right)
            node.n = node.left.n + node.right.n

    def get_bounding_box(self, node): 
        if isinstance(node, Branch):
            if node.left:
                self.get_bounding_box(node.left)
            if node.right:
                self.get_bounding_box(node.right)
            bbox = np.vstack([np.minimum(node.left.bounding_box[0, :], node.right.bounding_box[0, :]),
                              np.maximum(node.left.bounding_box[-1, :], node.right.bounding_box[-1, :])])
            node.bounding_box = bbox
            
    def make_cut(self, X, S, parent=None, side='left'): 
        xmax = X[S].max(axis=0)
        xmin = X[S].min(axis=0)
        l = xmax - xmin
        l = l / l.sum()
        j = np.random.choice(self.ndim, p = l)
        p = np.random.uniform(xmin[j], xmax[j])
        S1 = (X[:, j] <= p) & (S)
        S2 = (~S1) & (S)
        child = Branch(cut_dim=j, cut_val=p, parent=parent)
        # Link child node to parent
        if parent is not None:
            setattr(parent, side, child)
        return S1, S2, child


    def make_tree(self, X, S, N, I,parent=None, side='root', depth=0):
        depth += 1
        S1, S2, branch = self.make_cut(X, S, parent=parent, side=side)
        # If S1 does not contain an isolated point 
        if S1.sum() > 1:
            self.make_tree(X, S1, N, I,parent=branch, side='left', depth=depth)
        else:
            # Create a leaf node from isolated point 
            i = np.flatnonzero(S1).item()
            leaf = Leaf(index=i, depth=depth, parent=branch, x=X[i, :], n=N[i])
            # Link leaf node to left of parent  
            branch.left = leaf
            if I is not None:
                # Add a key in the leaves dict pointing to leaf for all duplicate indices   
                J = np.flatnonzero(I == i)
                # Get index label
                J = self.index_labels[J]
                for j in J:
                    self.leaves[j] = leaf
            else:
                i = self.index_labels[i]
                self.leaves[i] = leaf
        # If S2 does not contain an isolated point 
        if S2.sum() > 1:
            self.make_tree(X, S2, N, I, parent=branch, side='right', depth=depth)
        else:
            # Create a leaf node from isolated point 
            i = np.flatnonzero(S2).item()
            leaf = Leaf(index=i, depth=depth, parent=branch, x=X[i, :], n=N[i])
            # Link leaf node to right of parent
            branch.right = leaf
            if I is not None:
                # Add a key in the leaves dict pointing to leaf for all duplicate indices 
                J = np.flatnonzero(I == i)
                # Get index label 
                J = self.index_labels[J]
                for j in J:
                    self.leaves[j] = leaf
            else:
                i = self.index_labels[i]
                self.leaves[i] = leaf
        # return original depth 
        depth -= 1

    def add_depth(self, node, incre = 1):
        if isinstance(node, Branch):
            if node.left:
                self.add_depth(node.left, incre=(incre))
            if node.right:
                self.add_depth(node.right, incre=(incre))
        else:
            node.depth += (incre)

    def forget_point(self, index):
        leaf = self.leaves[index]
        #[FIND LEAF P IN T CORR. TO POINT P]
        #case 1: duplicated leaves
        if leaf.n > 1:
            while leaf:
                leaf.n -= 1
                leaf = leaf.parent
            return self.leaves.pop(index)
        #case 2: leaf is the root
        if leaf is self.root:
            self.root = None
            self.ndim = None
            return self.leaves.pop(index)
        parent = leaf.parent
        if leaf is parent.left:
            sibling = parent.right
        else:
            sibling = parent.left
        #parent is the root
        if parent is self.root:
            del parent
            sibling.parent = None
            self.root = sibling
            if isinstance(sibling, Leaf):
                sibling.depth = 0
            else:
                self.add_depth(sibling,incre= -1)
            return self.leaves.pop(index)
        grandparent = parent.parent
        sibling.parent = grandparent
        if parent is grandparent.left:
            grandparent.left = sibling
        else:
            grandparent.right = sibling
        parent = grandparent
        self.add_depth(sibling, incre = -1)
        while parent:
            parent.n -= 1
            parent = parent.parent
        # Update bounding boxes
        point = leaf.x 
        while parent:
            bbox = np.vstack([np.minimum(parent.left.bounding_box[0, :], parent.right.bounding_box[0, :]),
                          np.maximum(parent.left.bounding_box[-1, :], parent.right.bounding_box[-1, :])])
            if not ((parent.bounding_box[0, :] == point) | (parent.bounding_box[-1, :] == point)).any():
                break
            parent.bounding_box[0, :] = bbox[0, :]
            parent.bounding_box[-1, :] = bbox[-1, :]
            parent = parent.parent
        return self.leaves.pop(index)

    def insert_point(self, point, index):
        point = np.asarray(point).ravel()
        #case 1: no points in tree
        if self.root is None:
            leaf = Leaf(x=point, index=index, depth=0)
            self.root = leaf
            self.ndim = point.size
            self.leaves[index] = leaf
            return leaf
        node = self.root
        parent = node.parent
        maxdepth = max([leaf.depth for leaf in self.leaves.values()])
        depth = 0
        branch = None
        for _ in range(maxdepth + 1): #including root
            bbox = node.bounding_box
            bbox_hat = np.empty(bbox.shape)
            bbox_hat[0, :] = np.minimum(bbox[0, :], point)
            bbox_hat[-1, :] = np.maximum(bbox[-1, :], point)
            span = bbox_hat[-1, :] - bbox_hat[0, :]
            rg = span.sum()
            r = np.random.uniform(0, rg)
            span_sum = np.cumsum(span)
            cut_dimension = np.inf
            for j in range(len(span_sum)):
                if span_sum[j] >= r:
                    cut_dimension = j
                    break
            cut = bbox_hat[0, cut_dimension] + span_sum[cut_dimension] - r
            if cut <= bbox[0, cut_dimension]:
                leaf = Leaf(x=point, index=index, depth=depth)
                branch = Branch(cut_dim=cut_dimension, cut_val=cut, left=leaf, right=node,
                                n=(leaf.n + node.n))
                break
            elif cut >= bbox[-1, cut_dimension]:
                leaf = Leaf(x=point, index=index, depth=depth)
                branch = Branch(cut_dim=cut_dimension, cut_val=cut, left=node, right=leaf,
                                n=(leaf.n + node.n))
                break
            else:
                depth += 1
                if point[node.cut_dim] <= node.cut_val:
                    parent = node
                    node = node.left
                    side = 'left'
                else:
                    parent = node
                    node = node.right
                    side = 'right'
        try:
            assert branch is not None
        except:
            raise AssertionError('a cut was not found.')                    
        node.parent = branch
        leaf.parent = branch
        branch.parent = parent
        if parent is not None:
            setattr(parent, side, branch)
        else:
            self.root = branch
        self.add_depth(branch, incre = 1)
        while parent:
            parent.n += 1
            parent = parent.parent
        # Update bounding boxes of all nodes above p   
        bbox0 = np.vstack([np.minimum(branch.left.bounding_box[0, :], branch.right.bounding_box[0, :]),
                          np.maximum(branch.left.bounding_box[-1, :], branch.right.bounding_box[-1, :])])
        branch.bounding_box = bbox0
        branch = branch.parent
        while branch:
            lt = (bbox0[0, :] < branch.bounding_box[0, :])
            gt = (bbox0[-1, :] > branch.bounding_box[-1, :])
            lt_any = lt.any()
            gt_any = gt.any()
            if lt_any or gt_any:
                if lt_any:
                    branch.bounding_box[0, :][lt] = bbox0[0, :][lt]
                if gt_any:
                    branch.bounding_box[-1, :][gt] = bbox0[-1, :][gt]
            else:
                break
            branch = branch.parent
        self.leaves[index] = leaf
        return leaf
    
    def displacement(self, leaf):
        leaf = self.leaves[leaf]
        #case: leaf is root
        if leaf is self.root:
            return 0
        parent = leaf.parent
        # Find sibling
        if leaf is parent.left:
            sibling = parent.right
        else:
            sibling = parent.left
        # Count number of nodes in sibling subtree 
        displacement = sibling.n
        return displacement

    def codisplacement(self, leaf):
        leaf = self.leaves[leaf]
        #case: leaf is root 
        if leaf is self.root:
            return 0
        results = []
        for _ in range(leaf.depth):
            parent = leaf.parent
            if parent is None:
                break
            if leaf is parent.left:
                sibling = parent.right
            else:
                sibling = parent.left
            num_deleted = leaf.n
            displacement = sibling.n
            result = (displacement / num_deleted)
            results.append(result)
            leaf = parent
        co_displacement = max(results)
        return co_displacement      
        
# define the function to generate rolling window for streaming data   
def sliding(rdd, n):
    def gen_window(xi, n):
        x, i = xi
        return [(i - offset, (i, x)) for offset in range(n)]

    return (
        rdd.
        zipWithIndex(). # Add index
        flatMap(lambda xi: gen_window(xi, n)). 
        groupByKey(). 
        # Sort values to ensure order inside window and drop indices 
        mapValues(lambda vals: [x for (i, x) in sorted(vals)]). 
        sortByKey(). # Sort to makes sure we keep original order
        values(). # Get values
        filter(lambda x: len(x) == n)) # Drop beginning and end


class RRCForest(object):
    def __init__(self, n, t):
        self.make_forest(num_trees)
        self.num_trees = n
        self.tree_size = t
        self.avg_codisp = {}
        self.idx = 0
        
    def make_forest(self, n):
        self.forest = []
        for _ in range(n):
            tree = RRCTree()
            self.forest.append(tree)
            
    def get_codisp(self, points):
        for _, point in points.toLocalIterator():
            index = self.idx            #points.toLocalIterator():
            for tree in self.forest:
                if len(tree.leaves) > self.tree_size:
                    tree.forget_point(index - self.tree_size)
                tree.insert_point(point, index=index)
                new_codisp = tree.codisplacement(index)
                if not index in self.avg_codisp:
                    self.avg_codisp[index] = 0
                self.avg_codisp[index] += new_codisp / self.num_trees
            self.idx += 1
        return self.avg_codisp
    
    def update(self, point):
        index = self.idx   #points.toLocalIterator():
        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(index - self.tree_size)
            tree.insert_point(point, index=index)
            new_codisp = tree.codisplacement(index)
            if not index in self.avg_codisp:
                self.avg_codisp[index] = 0
            self.avg_codisp[index] += new_codisp / self.num_trees
        self.idx += 1
        return (index, self.avg_codisp[index])
        


if __name__ == "__main__":

    sc = SparkContext()
    spark = SparkSession(sc)
    
    # reading the data set  
    print('\n\nReading the training dataset\n')
    training_data = sc.textFile('gs://yytyvonne-bucket/aig.csv')
    print('\n\nTraining...\n')
    n_dim = 6
    shingle_size = 5
    data = training_data.map(lambda line : line.split(',')).map(lambda x: (int(x[0]), [float(x[i]) for i in range(1,n_dim+1)]))
    points = sliding(data, n=shingle_size).map(lambda window: (window[0][0], [window[i][1] for i in range(5)]))

    num_trees = 40
    tree_size = 260

    # Create a forest of empty trees
    rrcf = RRCForest(num_trees,tree_size)
    train_res = rrcf.get_codisp(training_points)
    train = sc.parallelize(train_res.items()).filter(lambda x: x[1] > 10)
    train.collect()
    print(train.collect())
    
    print('\n\nTraining done!\n')


    print('\n\nModel trained....awaiting testing data\n')

    testing_data = sc.textFile('gs://yytyvonne-bucket/aig_pred.csv')
    data1 = testing_data.map(lambda line : line.split(',')).map(lambda x: (int(x[0]), [float(x[i]) for i in range(1,n_dim+1)]))
    testing_points = sliding(data1, n=shingle_size).map(lambda window: (window[0][0], [window[i][1] for i in range(5)]))
    test_res = rrcf.get_codisp(testing_points)
    test = sc.parallelize(test_res.items()).filter(lambda x: x[1] > 10)
    test.collect()    
    print(test.collect())
    
    print('\n\nTesting finished. Ready for use!\n')
    # ssc = StreamingContext(sc, 1) # 5 second interval
    # ssc.checkpoint("checkpoint")
    # data = ssc.socketTextStream("localhost", 9999)


    # ssc.start()
    # ssc.awaitTermination()

    
