"""

"""


################################## IMPORTS ##################################


import torch
import torchvision


################################### MODEL ##################################


class Gng(torch.nn.Module):
    def __init__(self, e_b, e_n, a_max, l, a, passes, input_dim, max_nodes):   
        """
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
        self._passes = passes
        self._input_dim = input_dim
        self._max_nodes = max_nodes

        # Determine the device to run on (i.e. CUDA if available, else CPU).
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\033[92m✓ CUDA available.\033[0m" if self.device.type == 'cuda' 
                                                 else "\033[91m⚠ CUDA unavailable. Running on CPU.\033[0m")
        
        # Initialize the first two nodes with random values.
        # We register the attributes as buffers, so that they are saved when we save the model.
        # We can access them like any other attribute (i.e. self._nodes).
        self.register_buffer('_nodes', torch.randn(3, input_dim, device=self.device))

        # Edges are represented as an adjacency matrix, where each entry is the age of the edge (0 if no edge).
        # It is initialized at maximum size, which is a bit wasteful but avoids having to dynamically 
        # "resize" it later on (i.e. recreating the entire matrix).
        self.register_buffer('_edges', torch.zeros(max_nodes, max_nodes, device=self.device))

        # Move all tensors to the device.
        self._nodes = self._nodes.to(self.device)
        self._edges = self._edges.to(self.device)

        # Create an edge between the first two nodes.
        self._connect_nodes(0, 1)
        self._connect_nodes(0,2)
        self._connect_nodes(1,2)


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
            # Step 2: Find the two nodes closest to the input signal
            bmu, second_closest = self._get_closest(image)
            # Step 3: Increment the age of all edges connected to the BMU
            self._increment_ages(bmu)
            # Step 4: Increase the BMU's local error



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

        #print("Distances: ", distances)
        #print("Closest nodes: ", torch.topk(distances, 2, largest=False)[1])

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

        #print("Ages: ", self._edges)


    #def _squared_error(self, )


    def _connect_nodes(self, node_a, node_b):
        """
        Connect two nodes with an edge (most likely the BMU and the second closest node).
        Step 6 of the GNG algorithm.

        :param node_a: The index of the first node.
        :param node_b: The index of the second node.
        """

        self._edges[node_a, node_b] = 1
        self._edges[node_b, node_a] = 1