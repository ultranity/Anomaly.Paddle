"""CoreSet Sampling Methods for PatchCore.
ported from anomalib

- k Center Greedy Method
    Returns points that minimizes the maximum distance of any point to a center.
    . https://arxiv.org/abs/1708.00489
"""

from typing import List, Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor

from dim_reduction import SparseRandomProjection
from tqdm import tqdm
pairwise_distance = paddle.nn.PairwiseDistance(p=2)
class KCenterGreedy:
    """Implements k-center-greedy method.

    Args:
        embedding (Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        paddle.Size([219520, 1536])
        >>> sampler = KCenterGreedy(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        paddle.Size([219, 1536])
    """

    def __init__(self, embedding: Tensor, sampling_ratio: float) -> None:
        assert sampling_ratio<1
        assert sampling_ratio>0
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: Tensor
        self.min_distances: Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = None

    def update_distances(self, cluster_centers: List[int]) -> None:
        """Update min distances given cluster centers.

        Args:
            cluster_centers (List[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]
            distance = pairwise_distance(self.features, centers).reshape((-1, 1))

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = paddle.minimum(self.min_distances, distance)

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []
        #w = torch.load('../anomalib/random_matrix.pth',map_location=torch.device('cpu')).numpy()
        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape((self.embedding.shape[0], -1))
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs = []
        idx = paddle.randint(high=self.n_observations, shape=(1,))#.item()
        for _ in tqdm(range(self.coreset_size)):
            self.update_distances(cluster_centers=[idx])
            idx = paddle.argmax(self.min_distances)
            #if idx in selected_idxs:
            #    raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return paddle.concat(selected_coreset_idxs)

    def sample_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            paddle.Size([219520, 1536])
            >>> sampler = KCenterGreedy(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            paddle.Size([219, 1536])
        """

        idxs = self.select_coreset_idxs(selected_idxs)
        coreset = self.embedding[idxs]

        return coreset
