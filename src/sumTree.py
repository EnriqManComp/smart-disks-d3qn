import numpy as np

class SumTree():
    def __init__(self, capacity):
        # Pointer to the data in the tree
        self.data_pointer = 0
        # Number of leaves nodes
        self.capacity = capacity                
        # Number of nodes in the tree and fill them with zeros
        self.tree = np.zeros(2 * capacity - 1, dtype=float)
        # Setting size of data = number of leaves nodes and fill them with zeros
        self.data = np.zeros(capacity, dtype=int)        
        
        
    
    def add(self, priority, data_idx):
        """
            Adding the data and priority to the sumtree leaf
            Args:                
    
        """
        # Setting the index where represent the data in the tree, filling the tree from left to right
        tree_index = self.data_pointer + self.capacity - 1        
        # Inserting data        
        self.data[self.data_pointer] = data_idx        
        # Updating the leaf
        self.update(tree_index, priority)
        # Adding 1 to the pointer
        self.data_pointer += 1        
        # Checking if the pointer exceed the capacity
        if self.data_pointer >= self.capacity:
            # Restart in the first leaf
            self.data_pointer = 0
        

    def update(self, tree_index, priority):
        """
            Update all the tree with the new priority

        """
        
        # Calculating the changes
        parent_change = priority - self.tree[tree_index]
        # Updating leaf
        self.tree[tree_index] = priority        
        # Updating parents
        while tree_index != 0:
            # One step up
            tree_index = (tree_index - 1) // 2
            # Updating parent
            self.tree[tree_index] += parent_change            

    def get_leaf(self, value):
        """
            Get the leaf with priority closely to value
        """
        
        parent_index = 0
        while True:
            # Establish child pointer
            left_child_pointer = 2 * parent_index + 1
            right_child_pointer = left_child_pointer + 1

            # Stopping condition, bottom of the tree reached 
            if left_child_pointer >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_pointer]:
                    
                    parent_index = left_child_pointer
                else:
                    value -= self.tree[left_child_pointer]
                    
                    parent_index = right_child_pointer

        # Getting the data index that correspond with the leaf index
        data_p = leaf_index - self.capacity + 1    
                        
        return int(self.data[data_p]), self.tree[leaf_index], leaf_index
    
    @property
    def total_priority(self):
        """
            Return the root node
        """
        return self.tree[0]
    
    def load(self):
        path = './records' 
        self.data = np.loadtxt(path+'/data_idx.csv', dtype=int)  
        self.tree = np.loadtxt(path+'/tree_idx.csv')  
    
    
        
        


        
        