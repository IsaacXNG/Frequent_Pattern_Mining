import pandas as pd
import numpy as np
import itertools
import operator
import sys
import time

'''
@Author Xiaoning Guo
CSC 240: Data Mining Miniproject 1
Python Version 3.6
Anaconda Version 4.3.30

Uses the Apriori Algorithm and FP-Growth Algorithm
Described in Data Mining: Principles and Techniques 3rd Edition
'''

filename = "adult.data"
min_sup = 20000
showtree = False
fp_only = False

try:
    if len(sys.argv) != 1:
        filename = sys.argv[1]
        min_sup = float(sys.argv[2])
    if len(sys.argv) >= 4:
        for i in np.arange(0, len(sys.argv)):
            if sys.argv[i] == 'showtree':
                showtree = True
            if sys.argv[i] == 'fp-only':
                fp_only = True

except:
    print('Command-line arguments are not correct. Falling back to default.')

data = pd.read_csv(filename, header=None, sep=', ', engine='python')
data = data.replace('?', np.nan)

if filename == "adult.data":
    data[0] = data[0].apply(lambda x: 'Young' if x < 30 else ('Middle-aged' if x < 55 else 'Old'))
    data[10] = data[10].apply(lambda x: 'gain_zero' if x==0 else 'gain_positive')
    data[11] = data[11].apply(lambda x: 'loss_zero' if x==0 else 'loss_positive')
    data[12] = data[12].apply(lambda x: 'full-time' if x>=40 else 'part-time')
    data[2] = data[2].apply(lambda x: 'fnlwgt_low' if x < 50000 else ('fngwgt_med' if x < 100000 else 'fnlwgt_high'))
    data.drop(4, axis=1, inplace = True) #column 4 is just column 3 in numerical form, thus unnecessary

'''
Preparation
'''
frequent_list = [] #For apriori
frequent_table = {} #For fp-growth
if min_sup < 1:
    min_sup = min_sup*len(data)
    
'''
Mining Algorithms Begin
'''
def apriori(D):
    L1 = find_frequent_1item(D)
    frequent_list.append(L1)
    k = 2

    #Keep pruning and finding frequent patterns until no more supersets can be formed
    while(bool(frequent_list[k-2])):
        candidate_list = apriori_gen(frequent_list[k-2])
        candidate_list = find_occurrences(candidate_list)
        min_prune(candidate_list)
        frequent_list.append(candidate_list)
        k += 1
    return frequent_list

def apriori_gen(dict_k1):
    candidate_list = []
    list_k1 = list(dict_k1.keys())

    #Finds all combinations formed from List_k-1
    for i in np.arange(0, len(list_k1)-1):
        for j in np.arange(i+1, len(list_k1)):
            candidate = []

            #dump all possible unseen items into a candidate itemset
            for item in list_k1[i]:
                if item not in candidate:
                    candidate.append(item)
            for item2 in list_k1[j]:
                if item2 not in candidate:
                    candidate.append(item2)

            #Finds all combinations of size k itemsets from the candidate itemset
            candidate = list(itertools.combinations(candidate, len(list_k1[0])+ 1))

            #Check each combination to see if the subsets are infrequent
            for combo in candidate:
                #Sorting insures that tuple permutations are treated the same as any other
                combo = list(combo)
                combo.sort()
                combo = tuple(combo)
                if(combo not in candidate_list and not has_infrequent_subset(combo, list_k1)):
                    candidate_list.append(combo)

    return candidate_list

def has_infrequent_subset(candidate, list_k1):
    for item in candidate:
        subset = list(candidate)
        subset.remove(item)
        if(tuple(subset) not in list_k1):
            return True
    return False

def find_frequent_1item(D):
    item1_list = {}
    for row in D.itertuples():
        for item in row[1:]:
            if item is np.nan:
                continue
            elif tuple([item]) in item1_list:
                item1_list[tuple([item])] = item1_list[tuple([item])] + 1
            else:
                item1_list[tuple([item])] = 1

    min_prune(item1_list)
    return item1_list

'''
Returns a dictionary with the occurrences in the database
'''
def find_occurrences(tuple_list):
    list_k = {}

    for itemset in tuple_list:

        for row in data.itertuples():
            contains = True

            for item in itemset:
                if item not in row[1:]:
                    contains = False
                    break

            if contains:
                if itemset in list_k:
                    list_k[itemset] = list_k[itemset] + 1
                else:
                    list_k[itemset] = 1

    return list_k

'''
Prunes items that do not meet the minimum support
'''
def min_prune(dict_k):
    bad_key = []

    for k, v in dict_k.items():
        if v < min_sup:
            bad_key.append(k)

    for item in bad_key:
        del dict_k[item]

'''
Construction of the fp-tree given the initial data. 
This step does the initial 1-itemset frequency prune
@Returns the fp-tree 
'''
def fp_construct(D):
    L = find_frequent_1item(D)
    L_desc = dict(sorted(L.items(), key=operator.itemgetter(1), reverse=True))
    ordered_keys = L_desc.keys()
    ordered_itemsets = [[],[]]

    #Create ordered itemsets
    for row in D.itertuples():
        record = []
        for item in ordered_keys:
            if item[0] in row[1:]:
                record.append(item)
        ordered_itemsets[0].append(record)
        ordered_itemsets[1].append(1)

    return generate_tree(L_desc, ordered_itemsets)

'''
Generates an FP-tree given the ordered itemsets and items in descending frequency
@Returns the fp-tree
'''
def generate_tree(L_desc, ordered_itemsets):
    header_table = {}
    # Link each atomic item to a linkedlist
    for item in L_desc:
        linkedlist = LinkedList()
        header_table[item] = linkedlist

    newTree = FP_Tree(header_table)

    #ordered_itemsets[0] is the item. ordered_itemsets[1] is the counts
    for i in np.arange(0, len(ordered_itemsets[0])):
        newTree.insert_tree(ordered_itemsets[0][i], header_table, ordered_itemsets[1][i])

    newTree.root.header = header_table
    return newTree

'''
Performs the actual mining of the data and spits it out into the global frequency itemset list
'''
def fp_growth(tree, alpha):
    single_path = False
    current = tree.root

    while len(current.next) == 1:
        current = current.next[0]
        if current.next is None:
            single_path = True

    if single_path:
        current = tree.root
        combo_choices = {}

        while current.next[0] is not None:
            if current.next[0].count < min_sup:
                current.next[0] = None
            else:
                current = current.next[0]
                combo_choices[current.item] = current.count

        combo_list = []
        for i in np.arange(1, len(combo_choices) + 1):
            combo_list += list(itertools.combinations(combo_choices, i))

        for combo in combo_list:
            count = combo_choices[combo[-1]] #Frequency is the count of the last node count
            itemset = alpha + list(combo)
            frequent_table[tuple(itemset)] = count

    else:
        for k, v in tree.header.items():
            if v.sumcounts() >= min_sup:
                beta = alpha + list(k)
                frequent_table[tuple(beta)] = v.sumcounts()
                ordered_itemsets = [[], []]
                L_desc = []

                current = v.head
                while current is not None:
                    ordered_itemsets[0].append(current.data.get_pattern())
                    ordered_itemsets[1].append(current.data.count)
                    current = current.next

                for key in tree.header.keys():
                    L_desc.append(key)
                L_desc.remove(k)

                newTree = generate_tree(L_desc, ordered_itemsets)
                if tree is not None:
                    fp_growth(newTree, beta)

'''
LinkedLists for the purpose of the header tables
'''
class Node:
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def __repr__(self):
        return 'Node-LinkedList'

    def append(self, data):
        if self.head is None:
            self.head = Node(data, self.head)
            self.tail = self.head
        else:
            newNode = Node(data)
            self.tail.next = newNode
            self.tail = newNode

    def sumcounts(self):
        total = 0
        current = self.head
        while (current is not None):
            total += current.data.count
            current = current.next
        return total

    def printall(self):
        current = self.head
        while (current is not None):
            print(current.data.item, current.data.count)
            current = current.next


'''
FP-growth tree datastructures that contain the atomic item and counts
'''
class TreeNode:
    def __init__(self, item = None, count=1):
        self.item = item
        self.count = count
        self.parent = None
        self.header = None
        self.next = []

    def get_pattern(self):
        collection = []
        current = self

        while current.parent is not None:
            collection.insert(0, current.parent.item)
            current = current.parent

        return collection

class FP_Tree:
    def __init__(self, header_table):
        self.root = TreeNode()
        self.header = header_table

    def insert_tree(self, itemset, header_table, count=1):
        current = self.root

        for item in itemset:
            found_match = False

            for index in np.arange(0, len(current.next)):
                if item == current.next[index].item:
                    current = current.next[index]
                    current.count += count
                    found_match = True
                    break

            if item is None:
                continue
            if not found_match:
                newNode = TreeNode(item, count)
                newNode.parent = current
                header_table[item].append(newNode)
                current.next.append(newNode)
                current = newNode

    def print_tree(self):
        self.print_tree_wrapper(self.root, 0)

    def print_tree_wrapper(self, node, shift):
        if node is None:
            return
        for node in node.next:
            print(shift*"\t", node.item, node.count)
            self.print_tree_wrapper(node, shift+1)

'''
Command-line Arguments
'''
if not fp_only:
    start = time.time()
    apriori(data)
    count = 0
    for l_k in frequent_list:
        for k, v in l_k.items():
            count+=1
            print(k, v)
    end = time.time()
    runtime = str(round(end - start, 4))
    print("Number of frequent patterns (Apriori): ", count)
    print("Completion time: ", runtime, "seconds\n")

start = time.time()
tree = fp_construct(data)
fp_growth(tree, [])
frequent_table = sorted(frequent_table.items(), key=lambda x: len(x[0]))
for k, v in frequent_table:
    print(k, v)
end = time.time()
runtime = str(round(end - start, 4))
print("Number of frequent patterns (FP-Growth): ",len(frequent_table))
print("Completion time: ", runtime, "seconds\n")

if(showtree):
    tree.print_tree()