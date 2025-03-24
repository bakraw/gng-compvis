"""

"""


################################## IMPORTS ##################################


import torch
import torchvision


################################### MODEL ##################################


class Gng(torch.nn.Module):
    def __init__(self, e_b, e_n, a_max, l, a, d, passes, input_dim, max_nodes):   
        """
        Constructor for the GNG class.

        :param e_b: Learning rate for the best matching unit
        :param e_n: Learning rate for the neighbors of the BMU
        :param a_max: Maximum age of a node before it gets removed
        :param l: Number of inputs before a new node is created
        :param a: Error reduction factor applied to the new unit's neighbors
        :param passes: Number of passes over the training set (AKA epochs)
        :param input_dim: Dimensions of the input data (i.e. dimensions of the images)
        :param max_nodes: Maximum number of nodes in the network.
        """
        super().__init__()

        self._e_b = e_b
        self._e_n = e_n
        self._a_max = a_max
        self._l = l
        self._a = a
        self._d = d
        self._passes = passes
        self._input_dim = input_dim
        self._max_nodes = max_nodes

        # Determine the device to run on (i.e. CUDA if available, else CPU).
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\033[92m✓ CUDA available.\033[0m" if self.device.type == 'cuda' 
                                                 else "\033[91m⚠ CUDA unavailable. Running on CPU.\033[0m")
        
        # Disable gradient calculation, which is not needed for a GNG.
        torch.set_grad_enabled(False)
        
        # Initialize the first two nodes with random values.
        # We register the attributes as buffers, so that they are saved when we save the model.
        # We can access them like any other attribute (i.e. self._nodes).
        self.register_buffer('_nodes', torch.randn(2, input_dim, device=self.device))
        self.register_buffer('_local_error', torch.zeros(2, device=self.device))

        # Edges are represented as an adjacency matrix, where each entry is the age of the edge (0 if no edge).
        # It is initialized at maximum size, which is a bit wasteful but avoids having to dynamically 
        # "resize" it later on (i.e. recreating the entire matrix).
        self.register_buffer('_edges', torch.zeros(max_nodes, max_nodes, device=self.device))

        # Move all tensors to the device.
        self._nodes = self._nodes.to(self.device)
        self._edges = self._edges.to(self.device)

        # Create an edge between the first two nodes.
        self._connect_nodes(0, 1)

        self._inputs_count = 1


    #----------- PUBLIC METHODS -----------#


    def train(self, data_loader):
        """
        Begin training the network on the given data.

        :param data_loader: A PyTorch DataLoader object.
        """
        for epoch in range(self._passes):
            print(f"\033[94mEpoch {epoch + 1} / {self._passes}\033[0m")
            for i, (images, labels) in enumerate(data_loader):
                self.forward((images, labels))


    #----------- PRIVATE METHODS -----------#


    def forward(self, data):
        """
        Update the network's nodes and edges according to the input.
        Basically steps 1-9 of the GNG algorithm described in Fritzke's paper.

        Technically not a "forward pass" since this is a GNG, but I have to
        name it like that due to inheriting from torch.nn.Module.
        As such, despite it not being named "_forward()", I still placed it in the private section
        as **it is not meant to be called externally**.
        Use the train() method instead.

        :param data: A tuple containing an image batch and a label batch.
        """
        
        images, labels = data

        # Move the data to the device.
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Core algorithm
        # Step 1: Generate an input signal
        for image in images:
            #if self._inputs_count > 10: break
            # Step 2: Find the two nodes closest to the input signal
            bmu, second_closest = self._get_closest(image)
            # Step 3: Increment the age of all edges connected to the BMU
            self._increment_ages(bmu)
            # Step 4: Increase the BMU's local error
            self._squared_error(bmu, torch.norm(self._nodes[bmu] - image.view(-1)))
            # Step 5: Update the position of the BMU and its neighbors
            self._move_nodes(bmu, image)
            # Step 6: Reset / create the edge between the BMU and the second closest node
            self._connect_nodes(bmu, second_closest)
            # Step 7: Remove old edges and single nodes
            self._prune()
            # Step 8: Create a new node if we have reached the insertion threshold
            if self._inputs_count % self._l == 0 and self._nodes.shape[0] < self._max_nodes:
                self._insert_node()
            # Step 9: Increment error of all nodes
            self._increment_errors()

            self._inputs_count += 1
            print(self._inputs_count)



    def _get_closest(self, image):
        """
        Find the two nodes closest to the input image.
        Step 2 of the GNG algorithm.

        :param image: A PyTorch tensor representing an image.
        :return: A tensor containing the indices of the two closest nodes, first one being the BMU.
        """

        # Here torch.norm() is used to calculate the euclidean distance between the input and each node.
        # We flatten the image tensor to get the difference between it and each node.
        # We then take the norm of this difference to get the distance.
        distances = torch.norm(self._nodes - image.view(-1), dim=1)

        print("Distances: ", distances)
        print("Closest nodes: ", torch.topk(distances, 2, largest=False)[1])

        # torch.topk() returns a tuple of two tensors (values, indices).
        # We only need the indices, so we can just take the second element of the tuple.
        return torch.topk(distances, 2, largest=False)[1]


    def _increment_ages(self, node):
        """
        Increment the age of all edges connected to the given node (most likely the BMU).
        Step 3 of the GNG algorithm.

        :param node: The index of the node whose edges' ages should be incremented.
        """

        # Get the indices of all edges connected to the given node.
        connected_edges = torch.nonzero(self._edges[node], as_tuple=True)[0]

        #print("Connected edges: ", connected_edges)

        # Increment the age of all edges connected to the given node.
        # We have to do this for both directions since the adjacency matrix is symmetric.
        # PyTorch's "advanced indexing" allows us to use whole arrays as indices.
        self._edges[node, connected_edges] += 1
        self._edges[connected_edges, node] += 1

        print("Ages: ", self._edges)


    def _squared_error(self, node, distance):
        """
        Add the square of the given distance to the given node's local error.
        Step 4 of the GNG algorithm.

        :param node: The index of the node whose local error should be incremented.
        :param distance: The distance between the given node and the input signal.
        """
        self._local_error[node] += distance ** 2

        #print("Local errors: ", self._local_error)


    def _move_nodes(self, node, image):
        """
        Move the given node and its neighbors towards the given input signal.
        Nodes are moved by a fraction of the distance to the input signal
        (e_b for the node and e_n for its neighbors).
        Step 5 of the GNG algorithm.

        :param node: The index of the node to move.
        :param image: The input signal.
        """

        # Get the indices of all edges connected to the given node.
        connected_edges = torch.nonzero(self._edges[node], as_tuple=True)
        print("Nodes: ", self._nodes)
        print("Connected edges: ", connected_edges)

        # Calculate the difference between the input signal and the node.
        difference = image.view(-1) - self._nodes[node]

        # Move the node and its neighbors towards the input signal.
        self._nodes[node] += self._e_b * difference
        self._nodes[connected_edges] += self._e_n * difference


    def _connect_nodes(self, node_a, node_b):
        """
        Connect two nodes with an edge (most likely the BMU and the second closest node).
        Alternatively, reset its age if it already exists.
        Step 6 of the GNG algorithm.

        :param node_a: The index of the first node.
        :param node_b: The index of the second node.
        """

        self._edges[node_a, node_b] = 1
        self._edges[node_b, node_a] = 1


    def _prune(self):
        """
        Remove edges older than the maximum age
        as well as nodes with no connected neighbors.
        Step 7 of the GNG algorithm.
        """
    
        # Get the indices of all edges older than the maximum age and remove them.
        #print("Edges before prune: ", self._edges)
        old_edges = torch.nonzero(input=self._edges > self._a_max, as_tuple=True)
        #print("Old edges: ", old_edges)
        self._edges[old_edges] = 0
        print("Edges after prune: ", self._edges)

        # Get the indices of all nodes with no connected neighbors and remove them. 
        # We first get the indices of all nodes with at least one connected neighbor by summing the adjacency matrix.
        connected_nodes = torch.nonzero(torch.sum(self._edges, dim=1), as_tuple=True)
        print("Connected nodes: ", connected_nodes)

        single_nodes = torch.nonzero(torch.sum(input=self._edges, dim=1) == 0, as_tuple=True)[0]
        single_nodes = torch.nonzero(torch.sum(input=single_nodes) <= self._nodes.shape[0], as_tuple=True)[0]
        print("-----", single_nodes)
        single_nodes = single_nodes[:self._nodes.shape[0]]
        print("Single nodes: ", single_nodes)
        
        # Remove the nodes with no connected neighbors.
        # We also need to update the local error tensor and the edges tensor.
        #self._nodes = self._nodes[connected_nodes]
        if len(single_nodes) > 0:
            self._nodes[single_nodes] = 999
        self._local_error = self._local_error[connected_nodes]

        print("Nodes after prune: ", self._nodes)


    def _insert_node(self):
        """
        Create a new node between the node with the largest error and its neighbor with the largest error,
        update the edges and reduce the error of the two nodes according to alpha.
        Step 8 of the GNG algorithm.
        """

        # Get the index of the node with the largest error.
        largest_error_node = torch.argmax(self._local_error)
        print("Largest error node: ", largest_error_node)

        # Get the node's connected neighbors.
        connected_nodes = torch.nonzero(self._edges[largest_error_node], as_tuple=True)[0]
        print("Connected nodes: ", connected_nodes)

        # Get the index of the neighbor with the largest error.
        largest_error_neighbor = torch.argmax(self._local_error[connected_nodes])

        # Create a new node between them
        # We first have to create an empty tensor with the correct size, then fill it with the values we want.
        self._nodes = torch.cat((self._nodes, torch.zeros(1, self._input_dim)))
        self._nodes[self._nodes.shape[0] - 1] = (self._nodes[largest_error_node] + self._nodes[connected_nodes[largest_error_neighbor]]) / 2

        # Same process for the error tensor.
        self._local_error = torch.cat((self._local_error, torch.zeros(1)))
        self._local_error[self._local_error.shape[0] - 1] = self._local_error[largest_error_node]

        print("Nodes: ", self._nodes)
        print("Local errors: ", self._local_error)


    def _increment_errors(self):
        """
        Mutliply local error of all nodes by a factor of d.
        Step 9 of the GNG algorithm.
        """
        
        self._local_error *= self._d
