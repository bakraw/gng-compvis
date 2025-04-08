################################## IMPORTS ##################################


import torch


################################### MODEL ##################################


class Gng(torch.nn.Module):
    def __init__(self, input_dim=1, e_b=0.02, e_n=0.06, a_max=50, l=40, a=0.5, d=0.995, passes=2, max_nodes=5000, device="cpu"):   
        """
        Constructor for the GNG class.

        :param input_dim: Dimensions of the input data (i.e. dimensions of the images)
        :param e_b: Learning rate for the best matching unit
        :param e_n: Learning rate for the neighbors of the BMU
        :param a_max: Maximum age of a node before it gets removed
        :param l: Number of inputs before a new node is created
        :param a: Error reduction factor applied to the new unit's neighbors
        :param d: Error reduction factor applied to all nodes
        :param passes: Number of passes over the training set (AKA epochs)
        :param max_nodes: Maximum number of nodes in the network
        :param device: Device to use for training 
                        ("cpu" or "cuda" ; I recommend CPU for < 1000 nodes, CUDA for > 1000 nodes)
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
        self._device = device

        print(f"\033[93m🛈 Initializing GNG on the CPU.\033[0m" if self._device=="cpu" 
            else f"\033[93m🛈 Initializing GNG on the GPU.\033[0m")
        
        # Disable gradient calculation, which is not needed for a GNG.
        torch.set_grad_enabled(False)
        
        # Initialize the first two nodes with random values.
        # We register the attributes as buffers, so that they are saved when we save the model.
        # We can access them like any other attribute (i.e. self._nodes).
        self.register_buffer('_nodes', torch.randn(2, input_dim, device=self._device))
        self.register_buffer('_local_error', torch.zeros(2, device=self._device))

        # Edges are represented as an adjacency matrix, where each entry is the age of the edge (0 if no edge).
        self.register_buffer('_edges', torch.zeros(2, 2, device=self._device))

        # Create an edge between the first two nodes.
        self._connect_nodes(0, 1)

        self._inputs_count = 1


    #----------- PUBLIC METHODS -----------#


    def train(self, data_loader, class_count):
        """
        Begin training the network on the given data.

        :param data_loader: A PyTorch DataLoader object.
        :param class_count: The number of classes in the dataset.
        """

        print("\033[94m⏲ Training...\033[0m")
        for epoch in range(self._passes):
            print(f"\033[94mEpoch {epoch + 1} / {self._passes}\033[0m")
            for i, (images, labels) in enumerate(data_loader):
                self.forward((images, labels))
        
        print("\033[92m✓ Training complete.\033[0m")

        # The labels tensor is a 2D tensor (nodes x classes) which counts how many
        # times a node has been the BMU for a given class.
        self.register_buffer('_labels',
                             torch.zeros(self._nodes.shape[0], class_count, device=self._device))

        # Iterate through the data loader a second time to assign labels to the nodes.
        print("\033[94m⏲ Assigning labels to nodes...\033[0m")
        for i, (images, labels) in enumerate(data_loader):
            self._assign_labels((images, labels))
        print(f"\033[92m✓ Labels assigned to {self._nodes.shape[0]} nodes.\033[0m")


    def test(self, data_loader):
        """
        Test the network on the given data.

        Flat accuracy is a simple correct / total accuracy.
        Weighted accuracy takes certainty into account. Since incorrect predictions
        usually have lower certainty, weighted accuracy tends to be higher than flat accuracy.

        :param data_loader: A PyTorch DataLoader object.
        :return: The flat accuracy of the network as a float between 0 and 1.
        """

        total = 0
        weighted_total = 0
        total_correct = 0
        weighted_total_correct = 0
        total_certainty = 0

        print("\033[94m⏲ Testing...\033[0m")
        for _, (images, labels) in enumerate(data_loader):
            # Move the data to the device.
            images = images.to(self._device)
            labels = labels.to(self._device)

            for i, image in enumerate(images):
                total += 1

                closest_node = self._get_closest(image, 1)
                closest_labels = self._labels[closest_node]
                #print(closest_labels)

                # Prediction (i.e. label that has activated the closest node the most).
                predicted_label = closest_labels.argmax().item()
                certainty =  closest_labels.max().item() / (closest_labels.sum().item() 
                                                            if closest_labels.sum().item() != 0 else 9999)
                weighted_total += certainty
                total_certainty += certainty

                actual_label = labels[i].item()
                if predicted_label == actual_label:
                    total_correct += 1
                    weighted_total_correct += certainty
                #     print(f"\033[92m✓ Correct: {predicted_label} ({certainty * 100:.2f}%)\033[0m")
                # else: print(f"\033[91m✗ Incorrect: {predicted_label} ({certainty * 100:.2f}%)\033[0m")

                if total % 1000 == 0:
                    print(f"Tested {total} images | "
                          f"Accuracy: {total_correct / total * 100:.2f}%")
        
        print(f"\033[92m✓ Test complete.\n-> "
              f"Flat accuracy: {total_correct / total * 100:.2f}% | "
              f"Weighted accuracy: {weighted_total_correct / weighted_total * 100:.2f}% | "
              f"Average certainty: {total_certainty / total * 100:.2f}%\033[0m")
        
        return total_correct / total


    def load(self, path):
        """
        Load a pre-trained GNG from a ```.pth``` file.

        :param path: The path to the file to load.
        """

        print("\033[94m⏲ Loading model from {path}...\033[0m")

        # Load the state dictionary.
        loaded_model = torch.load(path, map_location=self._device)

        # Get the size of the loaded tensors.
        nodes_size = loaded_model['_nodes'].size()
        edges_size = loaded_model['_edges'].size()
        local_error_size = loaded_model['_local_error'].size()
        labels_size = loaded_model['_labels'].size()
        
        # Resize the tensors to match the loaded sizes.
        # self._labels has to be registered as a buffer as it doesn't exist before training.
        self._nodes = torch.zeros(nodes_size, device=self._device)
        self._edges = torch.zeros(edges_size, device=self._device)
        self._local_error = torch.zeros(local_error_size, device=self._device)
        self.register_buffer('_labels', 
                             torch.zeros(labels_size, device=self._device))

        # Load the state dictionary.
        self.load_state_dict(loaded_model)

        print("\033[92m✓ Model loaded.\033[0m")


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
        images = images.to(self._device)
        labels = labels.to(self._device)

        # Core algorithm
        # Step 1: Generate an input signal
        for i, image in enumerate(images):
            if self._inputs_count % 10000 == 0:
                print(f"Processed {self._inputs_count} inputs | {self._nodes.shape[0]} nodes")
            # Step 2: Find the two nodes closest to the input signal (& increment label count for the node)
            bmu, second_closest = self._get_closest(image, 2)
            # Step 3: Increment the age of all edges connected to the BMU
            self._increment_ages(bmu)
            # Step 4: Increase the BMU's local error by the squared distance between the BMU and the input signal
            self._local_error[bmu] += torch.norm(self._nodes[bmu] - image.view(-1)) ** 2
            # Step 5: Update the position of the BMU and its neighbors
            self._move_nodes(bmu, image)
            # Step 6: Reset / create the edge between the BMU and the second closest node
            self._connect_nodes(bmu, second_closest)
            # Step 7: Remove old edges and single nodes
            self._prune()
            # Step 8: Create a new node if we have reached the insertion threshold
            if self._inputs_count % self._l == 0 and self._nodes.shape[0] < self._max_nodes:
                self._insert_node()
            # Step 9: Decrease error of all nodes by a factor of d
            self._local_error *= self._d

            self._inputs_count += 1


    def _get_closest(self, image, k):
        """
        Find the k nodes closest to the input image.
        Step 2 of the GNG algorithm.

        :param image: A PyTorch tensor representing an image.
        :param k: The number of nodes to return.
        :return: A tensor containing the indices of the k closest nodes (first is closest).
        """

        # Here torch.norm() is used to calculate the euclidean distance between the input and each node.
        # We flatten the image tensor to get the difference between it and each node.
        # We then take the norm of this difference to get the distance.
        distances = torch.norm(self._nodes - image.view(-1), dim=1)

        # torch.topk() returns a tuple of two tensors (values, indices).
        # We only need the indices, so we can just take the second element of the tuple.
        return torch.topk(distances, k, largest=False)[1]


    def _increment_ages(self, node):
        """
        Increment the age of all edges connected to the given node.
        Step 3 of the GNG algorithm.

        :param node: The index of the node whose edges' ages should be incremented.
        """

        # Get the indices of all edges connected to the given node.
        connected_edges = torch.nonzero(self._edges[node], as_tuple=True)[0]

        # Increment the age of all edges connected to the given node.
        # We have to do this for both directions since the adjacency matrix is symmetric.
        # PyTorch's "advanced indexing" allows us to use whole arrays as indices.
        self._edges[node, connected_edges] += 1
        self._edges[connected_edges, node] += 1


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

        # Calculate the difference between the input signal and the node.
        difference = image.view(-1) - self._nodes[node]

        # Move the node and its neighbors towards the input signal.
        self._nodes[node] += self._e_b * difference
        self._nodes[connected_edges] += self._e_n * difference


    def _connect_nodes(self, node_a, node_b):
        """
        Connect two nodes with an edge.
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
        old_edges = torch.nonzero(input=self._edges > self._a_max, as_tuple=True)
        self._edges[old_edges] = 0
 
        # Get the indices of all nodes with at least one connected neighbor by summing the adjacency matrix.
        connected_nodes = torch.nonzero(torch.sum(self._edges, dim=1), as_tuple=True)[0]

        # Remove the nodes with no connected neighbors.
        # We also need to update the local error tensor and the labels tensor.
        self._nodes = self._nodes[connected_nodes]
        self._local_error = self._local_error[connected_nodes]

        # Update the adjacency matrix.
        self._edges = self._edges[: , connected_nodes]
        self._edges = self._edges[connected_nodes, :]


    def _insert_node(self):
        """
        Create a new node between the node with the largest error and its neighbor with the largest error,
        update the edges and reduce the error of the two nodes according to alpha.
        Step 8 of the GNG algorithm.
        """

        # Get the index of the node with the largest error.
        largest_error_node = torch.argmax(self._local_error)

        # Get the node's connected neighbors.
        connected_nodes = torch.nonzero(self._edges[largest_error_node], as_tuple=True)[0]

        # Get the index of the neighbor with the largest error.
        largest_error_neighbor = torch.argmax(self._local_error[connected_nodes])

        # Create a new node between them
        # We first have to create an empty tensor with the correct size, then fill it with the values we want.
        self._nodes = torch.cat((self._nodes, torch.zeros(1, self._input_dim, device=self._device)))
        self._nodes[self._nodes.shape[0] - 1] = (self._nodes[largest_error_node] 
                                                 + self._nodes[connected_nodes[largest_error_neighbor]]) / 2

        # Same process for the error tensor.
        self._local_error = torch.cat((self._local_error, torch.zeros(1, device=self._device)))
        self._local_error[self._local_error.shape[0] - 1] = self._local_error[largest_error_node]

         # Multiply their errors by alpha.
        self._local_error[largest_error_node] *= self._a
        self._local_error[connected_nodes[largest_error_neighbor]] *= self._a

        # Add row and column to the adjacency matrix.
        self._edges = torch.cat((self._edges, torch.zeros(self._edges.shape[0], 1, device=self._device)), dim=1)
        self._edges = torch.cat((self._edges, torch.zeros(1, self._edges.shape[1], device=self._device)), dim=0)

        # Connect the new node to the two nodes with the largest error.
        new_node = self._nodes.shape[0] - 1
        self._connect_nodes(largest_error_node, new_node)
        self._connect_nodes(largest_error_neighbor, new_node)

        # Remove the edge between the two nodes with the largest error.
        self._edges[largest_error_node, largest_error_neighbor] = 0
        self._edges[largest_error_neighbor, largest_error_node] = 0

    
    def _assign_labels(self, data):
        """
        Assign labels to the nodes based on the labels of the given images.

        Currently inefficient as hell but I'm too lazy to implement some kind of batch processing.

        :param data: A tuple containing an image batch and a label batch.
        """

        images, labels = data

        images = images.to(self._device)
        labels = labels.to(self._device)

        for i, image in enumerate(images):
            closest_nodes = self._get_closest(image, 4)
            self._labels[closest_nodes[0], labels[i]] += 3
            self._labels[closest_nodes[0:], labels[i]] += 1