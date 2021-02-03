# author: Anna Klezovich
# the code was tested in jupyter notebook python 3.7

import pandas as pd


# Task 1

class Tree:
    '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.
    Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
    '''

    def leaf(data):
        '''Create a leaf tree
        '''
        return Tree(data=data)

    # pretty-print trees
    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right)

            # all arguments after `*` are *keyword-only*!

    def __init__(self, *, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        '''Check if this tree is a leaf tree
        '''
        return self.left == None and self.right == None

    def children(self):
        '''List of child subtrees
        '''
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        '''Compute the depth of a tree
        A leaf is depth-1, and a child is one deeper than the parent.
        '''
        return max([x.depth() for x in self.children()], default=0) + 1

l1 = Tree.leaf("like")
l2 = Tree.leaf("TakeOtherSys?")
l3 = Tree.leaf("morning?")
l4 = Tree.leaf("likedOtherSys?")

l5 = Tree.leaf("like")
l6 = Tree.leaf("nah")

tree3 = Tree(data=l3, left=l5, right=l6)
tree4 = Tree(data=l4, left=l6, right=l5)
tree2 = Tree(data=l2, left=tree3, right=tree4)
tree = Tree(data="isSystem?", left=l1, right=tree2)

print(tree)

# task 2
# add a boolean "ok" column, where "True" means the rating is non-negative
# and "False" means the rating is negative.

data = pd.read_csv('data_class1.csv')
#data.head()
data['ok'] = list(data['rating'] >= 0)

# Write a function which takes a feature and computes the performance
# of the corresponding single-feature classifier:

def single_feature_score(data, goal, feature):
    corr = data[goal] == data[feature]
    score_pos = round(float(list(corr).count(True)/len(data)) * 100)
    corr_neg = ~corr
    score_neg = round(float(list(corr_neg).count(True)/len(data)) * 100)
    return max(score_neg, score_pos)

# test that single_feature_score works
print(single_feature_score(data, 'ok', 'morning'))

def best_feature(data, goal, features):
    # optional: avoid the lambda using `functools.partial`
    return max(features, key=lambda f: single_feature_score(data, goal, f))

# Which feature is best? Which feature is worst?

print("Which feature is best? ", best_feature(data, 'ok', ['easy', 'ai', 'systems', 'theory', 'morning']))

def worst_feature(data, goal, features):
    # optional: avoid the lambda using `functools.partial`
    return min(features, key=lambda f: single_feature_score(data, goal, f))

print("Which feature is worst?", worst_feature(data, 'ok', ['easy', 'ai', 'systems', 'theory', 'morning']))

# Task 4

#Implement the DecisionTreeTrain and DecisionTreeTest algorithms from Daumé,
# returning Trees. (Note: our dataset and his are different; we won't get the same tree.)
#How does the performance compare to the single-feature classifiers?

def decision_tree_train(data, remaining_features):
    # guess ← most frequent answer in data |= True or False
    guess = data['ok'].value_counts().idxmax()
    #print(guess)
    if data['ok'].nunique() == 1:
        return Tree.leaf(guess)
    elif remaining_features == []:
        return Tree.leaf(guess)
    else:
        f = best_feature(data, 'ok', remaining_features)
        # NO ← the subset of data on which f=no
        no = data[data[f] == False]
        # YES ← the subset of data on which f=yes
        yes = data[data[f] == True]
        remaining_features.remove(f)
        if no.empty:
            return Tree.leaf(guess)
        else:
            left = decision_tree_train(no, remaining_features)
        if yes.empty:
            return Tree.leaf(guess)
        else:
            right = decision_tree_train(yes, remaining_features)
        return Tree(data=f, left=left, right=right)


# learn tree
my_tree = decision_tree_train(data, ['easy', 'ai', 'systems', 'theory', 'morning'])


def decision_tree_test(tree, test_point):
    if tree.is_leaf():
        return tree.data
    else:
        if test_point[tree.data] == False:
            return decision_tree_test(tree.children()[0], test_point)
        else:
            return decision_tree_test(tree.children()[1], test_point)

predicted = []
for _, row in data.iterrows():
    test = decision_tree_test(my_tree, row)
    predicted.append(test)

def performance(predicted, data):
    correct = 0
    for el in predicted == data['ok']:
        if el is True:
            correct += 1
    accuracy = correct/len(predicted)
    return accuracy

print("Accuracy of decision tree ", performance(predicted, data))

# Add an optional maxdepth parameter to DecisionTreeTrain,
# which limits the depth of the tree produced.
# Plot performance against maxdepth.

# with maxdepth
def decision_tree_train(data, remaining_features, maxdepth):
    # guess ← most frequent answer in data |= True or False
    guess = data['ok'].value_counts().idxmax()
    # print(guess)
    if data['ok'].nunique() == 1:
        return Tree.leaf(guess)
    elif remaining_features == []:
        return Tree.leaf(guess)
    elif Tree.leaf(guess).depth() == maxdepth:
        return Tree.leaf(guess)
    else:
        f = best_feature(data, 'ok', remaining_features)
        # NO ← the subset of data on which f=no
        no = data[data[f] == False]
        # YES ← the subset of data on which f=yes
        yes = data[data[f] == True]
        remaining_features.remove(f)
        if no.empty:
            return Tree.leaf(guess)
        else:
            left = decision_tree_train(no, remaining_features, maxdepth)
        if yes.empty:
            return Tree.leaf(guess)
        else:
            right = decision_tree_train(yes, remaining_features, maxdepth)
        return Tree(data=f, left=left, right=right)

performances = []
for maxdepth in range(2,10):
    my_tree = decision_tree_train(data, ['easy', 'ai', 'systems', 'theory', 'morning'], maxdepth)
    predicted = []
    for _, row in data.iterrows():
        test = decision_tree_test(my_tree, row)
        predicted.append(test)
    performances.append(performance(predicted, data))

import matplotlib.pyplot as plt

plt.plot([maxdepth for maxdepth in range(2,10)], performances)
