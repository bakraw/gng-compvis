# sklearn doesn't offer GPU-acceleration by default,
# but Nvidia's cuML allows it without changing the code, so :
# - if you're running this locally, you'll have to install cuML manually ;
# - if you're on Colab, uncomment the line below.
# %load_ext cuml.accel

import gng
import umap
import torch
import sklearn.manifold
import sklearn.decomposition
import matplotlib.pyplot
import matplotlib.collections
import mpl_toolkits.mplot3d.art3d


class Visualization:
    def __init__(self, gng:gng.Gng, device=None, node_size=2, edge_width=0.1, edge_alpha=0.1):
        """
        Constructor for the Visualization class.

        :param gng: A GNG object to visualize.
        :param device: Device to use for computation
                    (```"cpu"``` or ```"cuda"``` ; if left empty,
                    will use the same device as the GNG)
        :param node_size: Size of the nodes to be plotted (default: 2).
        :param edge_width: Width of the edges to be plotted (default: 0.1).
        :param edge_alpha: Opacity of the edges to be plotted (default: 0.1).
        """

        if device is None: self._device = gng._device
        else: self._device = device
        print(f"\033[93m🛈 Initializing visualization on the CPU.\033[0m" if self._device=="cpu"
              else f"\033[93m🛈 Initializing visualization on the GPU.\033[0m")

        self._nodes = gng._nodes.to(self._device)
        self._edges = gng._edges.to(self._device)
        self._labels = gng._labels.to(self._device)
        self._node_size = node_size
        self._edge_width = edge_width
        self._edge_alpha = edge_alpha

        self._colors = [self._labels[i].argmax().item() for i in range(self._nodes.shape[0])]


    #----------- PUBLIC METHODS -----------#


    def pca(self, save_path=None, third_dim=True):
        """
        Perform PCA to project the nodes onto a 2D plane or 3D space.

        Here we don't use sklearn's implementation but PyTorch's so we can work 
        directly with tensors (-> GPU-acceleration OOTB).

        :param save_path: Path to save the plot to. If None, the plot will only be displayed.
        :param third_dim: If left to True, the points will be projected onto a 3D space
                        (for PCA 2D is just a "top-down view" of the 3D projection,
                        since we simply discard the 3rd principal component, i.e. the Z-axis).
        """
        print("\033[94m⏲ Performing PCA (3D)...\033[0m" if third_dim
              else "\033[94m⏲ Performing PCA (2D)...\033[0m")

        # Perform PCA
        # U -> principal components, S -> singular values, V -> projection matrix
        # We only need to use the first two or three columns of the projection matrix on the nodes.
        (U, S, V) = torch.pca_lowrank(self._nodes)
        projected_nodes = torch.matmul(self._nodes,
                                      V[:, :3] if third_dim else V[:, :2]) * 2

        self._plot(save_path, third_dim, projected_nodes,
                    ("1st principal component", "2nd principal component", "3rd principal component") if third_dim 
                    else ("1st principal component", "2nd principal component"), "PCA")
        

    def mds(self, save_path=None, third_dim=True, precision=0.2):
        """
        Perform MDS to project the nodes onto a 2D plane or 3D space.

        :param save_path: Path to save the plot to. If None, the plot will only be displayed.
        :param third_dim: If set to True, the points will be projected onto a 3D space.
        """

        print("\033[94m⏲ Performing MDS (3D)...\033[0m" if third_dim
              else "\033[94m⏲ Performing MDS (2D)...\033[0m")
        
        # Pre-compute the distance matrix
        distance_matrix = torch.cdist(self._nodes, self._nodes)

        # Landmark selection
        if precision < 1:
            num_landmarks = int(self._nodes.shape[0] * precision)
            landmarks = torch.zeros(num_landmarks, dtype=torch.long)

            # Select the point that is farthest from all other points.
            landmarks[0] =  torch.argmax(distance_matrix.sum(axis=1))

            for i in range(1, num_landmarks):
                # Compute minimum distance from all landmarks
                min_distances = torch.min(distance_matrix[landmarks[:i], :], dim=0)[0]
                # Add the point furthest from all landmarks.
                landmarks[i] = torch.argmax(min_distances)

                reduced_matrix = distance_matrix[landmarks, :]
                reduced_matrix = reduced_matrix[:, landmarks]

                reduced_matrix = reduced_matrix.detach().cpu().numpy()

                
        
        distance_matrix = distance_matrix.detach().cpu().numpy()

        # Perform MDS
        projected_nodes = torch.Tensor(sklearn.manifold.MDS(n_components=2 if not third_dim else 3,
                                                            metric=True,
                                                            dissimilarity="precomputed").fit_transform(reduced_matrix if precision < 1 else distance_matrix))

        # Plot the results.
        self._plot(save_path, third_dim, projected_nodes,
                    ("1st principal component", "2nd principal component", "3rd principal component") if third_dim
                    else ("1st principal component", "2nd principal component"), "MDS")
        

    def tsne(self, save_path=None, third_dim=False):
        """
        Perform t-SNE to project the nodes onto a 2D plane or 3D space.

        :param save_path: Path to save the plot to. If None, the plot will only be displayed.
        :param third_dim: If set to True, the points will be projected onto a 3D space.
        """
        print("\033[94m⏲ Performing t-SNE (3D)...\033[0m" if third_dim
              else "\033[94m⏲ Performing t-SNE (2D)...\033[0m")
        
        # Perform t-SNE.
        nodes = self._nodes.detach().cpu().numpy()
        projected_nodes = torch.Tensor(sklearn.manifold.TSNE(n_components=2 if not third_dim else 3).fit_transform(nodes))

        # Plot the results.
        self._plot(save_path, third_dim, projected_nodes,
                   ("t-SNE 1st component", "t-SNE 2nd component", "t-SNE 3rd component") if third_dim
                   else ("t-SNE 1st component", "t-SNE 2nd component"), "t-SNE")
        

    def umap(self, save_path=None, third_dim=False):
        """
        Perform UMAP to project the nodes onto a 2D plane or 3D space.

        :param save_path: Path to save the plot to. If None, the plot will only be displayed.
        :param third_dim: If set to True, the points will be projected onto a 3D space.

        :return: The coordinates of the projected points.
        """
        print("\033[94m⏲ Performing UMAP (3D)...\033[0m" if third_dim
              else "\033[94m⏲ Performing UMAP (2D)...\033[0m")
        
        # Perform UMAP.
        # Metric can be changed if necessary.
        nodes = self._nodes.detach().cpu().numpy()
        projected_nodes = torch.Tensor(umap.UMAP(n_components=2 if not third_dim else 3,
                                                 metric="euclidean").fit_transform(nodes))

        # Plot the results.
        self._plot(save_path, third_dim, projected_nodes,
                   ("UMAP 1st component", "UMAP 2nd component", "UMAP 3rd component") if third_dim
                   else ("UMAP 1st component", "UMAP 2nd component"), "UMAP")


    #----------- PRIVATE METHODS -----------#


    def _plot(self, save_path, third_dim, projected_nodes, labels, method):
        """
        Plot the nodes and edges of the graph.

        :param save_path: Path to save the plot to. If None, the plot will only be displayed.
        :param third_dim: If set to True, the points will be projected onto a 3D space.
        :param projected_nodes: The coordinates of the projected points to plot.
        :param labels: A tuple of labels for each axis (z is only needed if third_dim is True).
        :param method: The projection method used (shown in the title of the plot).
        """

        # Canvas setup
        matplotlib.pyplot.figure(figsize=(12, 8))

        # Plot the nodes
        if not third_dim:
            matplotlib.pyplot.scatter(projected_nodes.cpu()[:, 0], projected_nodes.cpu()[:, 1],
                                      s=self._node_size, c=self._colors, cmap='tab10')
        else:
            ax = matplotlib.pyplot.axes(projection='3d')
            ax.scatter3D(projected_nodes.cpu()[:, 0], projected_nodes.cpu()[:, 1], projected_nodes.cpu()[:, 2],
                         s=self._node_size, c=self._colors, cmap='tab10')

        # Get indices where edges exist
        edge_indices = torch.nonzero(self._edges, as_tuple=True)

        # Create edge pairs.
        edges = torch.stack([projected_nodes[edge_indices[0]], projected_nodes[edge_indices[1]]])

        # Convert to numpy array for matplotlib.
        # We have to transpose from (2, num_edges, 2) to (num_edges, 2, 2).
        edges = edges.detach().cpu().numpy().transpose(1, 0, 2)

        # Using a line collection for batch drawing (drawing the edges one by one is atrociously slow).
        if third_dim:
            line_collection = mpl_toolkits.mplot3d.art3d.Line3DCollection(edges,
                                                                          linewidths=self._edge_width,
                                                                          color=("gray", self._edge_alpha))
            ax.add_collection(line_collection)
        else:
            line_collection = matplotlib.collections.LineCollection(edges,
                                                                    linewidths=self._edge_width,
                                                                    color=("gray", self._edge_alpha))
            matplotlib.pyplot.gca().add_collection(line_collection)

        # Title and labels
        matplotlib.pyplot.title(f"GNG 3D Visualization - {method} - {self._nodes.shape[0]} nodes" if third_dim
                                else f"GNG 2D Visualization - {method} - {self._nodes.shape[0]} nodes")
        matplotlib.pyplot.xlabel(labels[0])
        matplotlib.pyplot.ylabel(labels[1])
        if third_dim: ax.set_zlabel(labels[2])

        # Legend
        unique_labels = list(set(self._colors))
        legend_elements = [matplotlib.pyplot.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor=matplotlib.pyplot.cm.tab10(i/10),
                                                    label=label, markersize=10)
                                                    for i, label in enumerate(unique_labels)]
        if third_dim: ax.legend(handles=legend_elements, title='Classes',
                                loc='center left', bbox_to_anchor=(1.1, 0.5))
        else: matplotlib.pyplot.legend(handles=legend_elements, title='Classes',
                                loc='center left', bbox_to_anchor=(1, 0.5))

        # Save and show the plot
        print("\033[92m✓ Visualization generated.\033[0m")
        if save_path is not None:
            print("\033[94m⏲ Saving...\033[0m")
            matplotlib.pyplot.savefig(save_path)
            print(f"\033[92m✓ Visualization saved to {save_path}.\033[0m")
        matplotlib.pyplot.show()