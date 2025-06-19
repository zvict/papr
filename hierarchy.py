import torch
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict, Tuple, List, Any  # For type hinting

# --- Dependency Check & Placeholders ---

try:
    # Attempt to import for rotation conversion
    from pytorch3d.transforms import axis_angle_to_matrix as p3d_axis_angle_to_matrix

    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False
    print(
        "Warning: pytorch3d not found. Using basic Rodrigues implementation for axis-angle conversion."
    )
    print(
        "Consider installing pytorch3d for potentially better performance and features: `pip install pytorch3d`"
    )


# Placeholder for your closest point finding function
# It should take parent points and child points (both torch tensors N,3)
# and return the *positions* of 'k' points from the child set deemed 'joint points'.
def find_child_joint_points(
    parent_points: torch.Tensor, child_points: torch.Tensor, k: int = 1
) -> torch.Tensor:
    """
    Placeholder: Finds k points from child_points considered joints relative to parent_points.

    Args:
        parent_points: Tensor (P, 3)
        child_points: Tensor (C, 3)
        k: Number of joint points to find.

    Returns:
        Tensor (3, ): Positions of the joint points from the child set.
    """
    distances = torch.norm(
        child_points.unsqueeze(1) - parent_points.unsqueeze(0), dim=2
    )  # (N, M)

    # 2) For each wing point, find its minimal distance to any body point.
    #    This effectively measures how close that wing point is to the body.
    #    min_dist_to_body will be (N,) containing the minimal distance for each of the N wing points.
    min_dist_to_body, _ = torch.min(distances, dim=1)  # shape (N,)

    # 3) Identify the 'k' wing points that are closest to the body.
    #    We sort the distances to find the top k.
    #    The result of argsort is a (N,) array of indices.
    #    Then we take the first k to get the k closest.
    closest_indices = torch.argsort(min_dist_to_body)[:k]

    # The boundary region is the set of these k closest wing points.
    boundary_region = child_points[closest_indices]

    # 4) The anchor point can be the centroid (average) of these boundary points.
    anchor_point = boundary_region.mean(dim=0, keepdim=True)  # shape (3,)

    return anchor_point

# --- Transformation Helpers ---


def average_transforms_6d(transforms_6d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Averages a batch of 6D transforms (axis-angle + translation) to get a single rotation and translation.

    Args:
        transforms_6d: Tensor (N, 6) - axis-angle and translation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (R, t) where R is the average rotation matrix (3, 3)
                                            and t is the average translation vector (3,).
    """
    # Split into axis-angle and translation
    axis_angle = transforms_6d[:, :3]  # (N, 3)
    translation = transforms_6d[:, 3:]  # (N, 3)

    # Convert axis-angle to rotation matrices
    R = axis_angle_to_matrix(axis_angle)  # (N, 3, 3)

    # Average the rotation matrices
    avg_R = torch.mean(R, dim=0)  # (3, 3)

    # Average the translations
    avg_t = torch.mean(translation, dim=0)  # (3,)

    return avg_R, avg_t


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Converts axis-angle vectors to rotation matrices."""
    if not _PYTORCH3D_AVAILABLE:
        # Basic Rodrigues' Formula Implementation (Batch-compatible)
        batch_size = axis_angle.shape[0]
        angle = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # Shape (B, 1)
        small_angle_mask = angle < 1e-6
        # Avoid division by zero for small angles
        axis = F.normalize(axis_angle, p=2, dim=-1)  # Shape (B, 3)
        axis[small_angle_mask.squeeze(-1)] = torch.tensor(
            [1.0, 0.0, 0.0], device=axis.device
        )  # Default axis for zero angle

        cos_a = torch.cos(angle)  # (B, 1)
        sin_a = torch.sin(angle)  # (B, 1)

        # Skew-symmetric matrix K
        zeros = torch.zeros_like(angle)  # (B, 1)
        K = torch.stack(
            [
                zeros,
                -axis[..., 2:3],
                axis[..., 1:2],
                axis[..., 2:3],
                zeros,
                -axis[..., 0:1],
                -axis[..., 1:2],
                axis[..., 0:1],
                zeros,
            ],
            dim=-1,
        ).reshape(
            batch_size, 3, 3
        )  # (B, 3, 3)

        I = (
            torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(batch_size, 3, 3)
        )  # (B, 3, 3)

        # Rodrigues' formula: R = I + sin(a)K + (1 - cos(a))K^2
        R = I + sin_a.unsqueeze(-1) * K + (1.0 - cos_a.unsqueeze(-1)) * torch.bmm(K, K)

        # Handle the small angle case (approximation R = I + sin(a)K ~ I + aK)
        # Alternatively, just use Identity for zero angle
        R[small_angle_mask.squeeze(-1)] = torch.eye(3, device=R.device)
        return R  # (B, 3, 3)

    else:
        # Use pytorch3d if available
        return p3d_axis_angle_to_matrix(axis_angle)  # Handles batching automatically


# --- Node Class (Modified) ---
class Node:
    """Represents a part node in the non-rigid hierarchy."""

    def __init__(
        self,
        name: str,
        part_index: int,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a Tensor of shape (N, 3)")
        if normals is not None and (
            normals.ndim != 2
            or normals.shape[1] != 3
            or normals.shape[0] != points.shape[0]
        ):
            raise ValueError(
                "Normals must be a Tensor of shape (N, 3) matching the number of points"
            )

        self.name = name
        self.part_index = part_index  # Unique integer ID
        self.device = device

        # --- Core Part Data ---
        self.points: torch.Tensor = points.to(device)  # (N, 3) Original local points
        self.normals: Optional[torch.Tensor] = (
            normals.to(device) if normals is not None else None
        )  # Optional (N, 3)

        # --- Hierarchy Links ---
        self.parent: Optional[Node] = None
        self.children: List[Node] = []

        # --- Joint Data (Stored in Parent) ---
        # Stores the *local positions* of joint points identified from children,
        # keyed by the child's part_index. These points are treated as part
        # of the parent for model input generation.
        self.child_joint_positions: Dict[int, torch.Tensor] = (
            {}
        )  # child_idx -> Tensor(K, 3)

        # --- Model Output Storage ---
        # Stores the model's predicted 6D transform (axis-angle, translation) for each point *of this node*.
        self.point_transforms: Optional[torch.Tensor] = None  # Tensor (N, 6)
        # Stores the model's predicted 6D transform for the joint points associated with children.
        # Keyed by child's part_index. These are slices of the parent's total predicted transforms.
        self.joint_transforms_for_child: Dict[int, torch.Tensor] = (
            {}
        )  # child_idx -> Tensor (K, 6)

        # --- World State ---
        # Stores the final calculated world positions for this node's points
        self.world_points: Optional[torch.Tensor] = None  # Tensor (N, 3)
        # Stores the calculated world transformation (R, t) for the joint connecting to each child.
        # This is what children will query. Keyed by child's part_index.
        self.world_joint_transforms: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = (
            {}
        )  # child_idx -> (Tensor(3,3), Tensor(3))

    def add_child(self, child_node: "Node"):
        """Adds a child node and establishes parent link."""
        if not isinstance(child_node, Node):
            raise TypeError("Child must be an instance of Node")
        if child_node.parent is not None:
            raise ValueError(f"Node {child_node.name} already has a parent.")
        child_node.parent = self
        self.children.append(child_node)

    def store_child_joints(self, child_node: "Node", joint_positions: torch.Tensor):
        """Stores the identified joint positions for a specific child."""
        if joint_positions.ndim != 2 or joint_positions.shape[1] != 3:
            raise ValueError("Joint positions must be a Tensor of shape (K, 3)")
        self.child_joint_positions[child_node.part_index] = joint_positions.to(
            self.device
        )

    @property
    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node."""
        return not bool(self.children)

    @property
    def num_points(self) -> int:
        """Number of original points in this part."""
        return self.points.shape[0]

    @property
    def num_total_stored_joints(self) -> int:
        """Total number of joint points stored in this node (from all children)."""
        return sum(j.shape[0] for j in self.child_joint_positions.values())

    def __repr__(self):
        """String representation of the node."""
        parent_name = self.parent.name if self.parent else "None"
        info = (
            f"Node(name='{self.name}', index={self.part_index}, pts={self.num_points}"
        )
        if self.child_joint_positions:
            info += f", joints_stored={self.num_total_stored_joints}"
        info += f", parent='{parent_name}')"
        return info


# --- Hierarchy Class (Modified) ---
class Hierarchy:
    """Manages the non-rigid node hierarchy and related operations."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.nodes: Dict[int, Node] = {}  # Map part_index to Node
        self.nodes_by_name: Dict[str, Node] = {}  # Map name to Node
        self.root: Optional[Node] = None
        self._part_counter: int = 0
        self.device: torch.device = device
        # Store info needed to map model input/output efficiently
        self._model_input_info: Dict[str, Any] = {}

    def add_node(
        self,
        name: str,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        parent_name: Optional[str] = None,
        num_joint_points: int = 1,
    ) -> Node:
        """
        Adds a new node (part) to the hierarchy.

        If a parent is specified, it identifies joint points using
        `find_child_joint_points` and stores their *local positions* in the parent node.

        Args:
            name: Unique name for the node.
            points: Tensor (N, 3) of local point positions for this part.
            normals: Optional Tensor (N, 3) of local normals.
            parent_name: Name of the parent node. If None, this node becomes the root.
            num_joint_points (k): Number of joint points to identify between child and parent.

        Returns:
            The newly created Node object.
        """
        if name in self.nodes_by_name:
            raise ValueError(f"Node with name '{name}' already exists.")
        if num_joint_points < 0:
            raise ValueError("num_joint_points cannot be negative.")

        part_index = self._part_counter
        self._part_counter += 1

        points_dev = points.to(self.device)
        normals_dev = normals.to(self.device) if normals is not None else None

        new_node = Node(name, part_index, points_dev, normals_dev, device=self.device)
        self.nodes[part_index] = new_node
        self.nodes_by_name[name] = new_node

        parent_node: Optional[Node] = None
        if parent_name:
            if parent_name not in self.nodes_by_name:
                # Clean up incomplete node addition
                del self.nodes[part_index]
                del self.nodes_by_name[name]
                self._part_counter -= 1
                raise ValueError(f"Parent node '{parent_name}' not found.")
            parent_node = self.nodes_by_name[parent_name]
            parent_node.add_child(new_node)

            # --- Identify and Store Joint Points ---
            if (
                num_joint_points > 0
                and parent_node.num_points > 0
                and new_node.num_points > 0
            ):
                # Call the external function to find joint positions *from the child's perspective*
                child_joint_positions = find_child_joint_points(
                    parent_node.points,  # Parent local points
                    new_node.points,  # Child local points
                    k=num_joint_points,
                )
                if child_joint_positions.shape[0] > 0:
                    # Store these child-local positions *in the parent node*
                    parent_node.store_child_joints(new_node, child_joint_positions)
                else:
                    print(
                        f"Warning: No joint points found or returned between {parent_name} and {name}."
                    )

            elif num_joint_points > 0:
                print(
                    f"Warning: Cannot find joints for {name} as parent or child has no points."
                )

        elif self.root is None:
            self.root = new_node  # First node added without parent is the root
        else:
            # Clean up if trying to add another root
            del self.nodes[part_index]
            del self.nodes_by_name[name]
            self._part_counter -= 1
            raise ValueError(
                "Hierarchy already has a root. Specify a parent_name for subsequent nodes."
            )

        return new_node

    def generate_model_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates concatenated points and a part mask for model input.
        Includes original points and stored joint points (treated as part of parent).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - concatenated_points: Tensor (TotalPoints, 3)
                - mask: Tensor (TotalPoints,) with part_index for each point.
                        Joint points get the part_index of the *parent* node they are stored in.
        """
        if not self.root:
            return torch.empty((0, 3), device=self.device), torch.empty(
                (0,), dtype=torch.long, device=self.device
            )

        all_points: List[torch.Tensor] = []
        all_indices: List[torch.Tensor] = []
        node_order: List[int] = []  # Store the order nodes are processed
        point_slice_info: Dict[int, Tuple[int, int]] = (
            {}
        )  # node_idx -> (start, end) for original points
        joint_slice_info: Dict[int, Dict[int, Tuple[int, int]]] = (
            {}
        )  # parent_idx -> {child_idx: (start, end)} for joints stored in parent

        current_pos = 0

        # Use BFS order for potentially better memory layout / consistency
        queue = deque([self.root])
        visited_indices = set()

        while queue:
            node = queue.popleft()
            if node.part_index in visited_indices:
                continue
            visited_indices.add(node.part_index)
            node_order.append(node.part_index)
            joint_slice_info[node.part_index] = {}

            # 1. Add original points for this node
            num_node_points = node.num_points
            if num_node_points > 0:
                all_points.append(node.points)
                indices = torch.full(
                    (num_node_points,),
                    node.part_index,
                    dtype=torch.long,
                    device=self.device,
                )
                all_indices.append(indices)
                point_slice_info[node.part_index] = (
                    current_pos,
                    current_pos + num_node_points,
                )
                current_pos += num_node_points
            else:
                point_slice_info[node.part_index] = (
                    current_pos,
                    current_pos,
                )  # Empty slice

            # 2. Add joint points stored in this node (from its children)
            for child_idx, joint_pos_tensor in node.child_joint_positions.items():
                num_joint_points = joint_pos_tensor.shape[0]
                if num_joint_points > 0:
                    all_points.append(joint_pos_tensor)
                    # IMPORTANT: Mask joints with the PARENT's index
                    indices = torch.full(
                        (num_joint_points,),
                        node.part_index,
                        dtype=torch.long,
                        device=self.device,
                    )
                    all_indices.append(indices)
                    joint_slice_info[node.part_index][child_idx] = (
                        current_pos,
                        current_pos + num_joint_points,
                    )
                    current_pos += num_joint_points
                else:
                    joint_slice_info[node.part_index][child_idx] = (
                        current_pos,
                        current_pos,
                    )

            for child in node.children:
                if child.part_index not in visited_indices:
                    queue.append(child)

        if not all_points:
            # Handle empty hierarchy case
            self._model_input_info = {
                "total_points": 0,
                "node_order": [],
                "point_slices": {},
                "joint_slices": {},
            }
            return torch.empty((0, 3), device=self.device), torch.empty(
                (0,), dtype=torch.long, device=self.device
            )

        concatenated_points = torch.cat(all_points, dim=0)
        mask = torch.cat(all_indices, dim=0)

        # Store info needed for output mapping
        self._model_input_info = {
            "total_points": concatenated_points.shape[0],
            "node_order": node_order,  # Order parts were processed
            "point_slices": point_slice_info,  # Slices for original points
            "joint_slices": joint_slice_info,  # Slices for joint points stored in parent
        }

        return concatenated_points, mask.unsqueeze(-1)

    def store_model_output(self, all_transforms_6d: torch.Tensor):
        """
        Distributes the flat tensor of transformations back to the individual nodes,
        separating transforms for original points and joint points.
        Assumes all_transforms_6d corresponds to the points generated by
        the most recent call to generate_model_input().

        Args:
            all_transforms_6d: Tensor (TotalPoints, 6) - axis-angle and translation.
        """
        if not self._model_input_info or self._model_input_info["total_points"] == 0:
            print(
                "Warning: Cannot store model output. Call generate_model_input() first or hierarchy is empty."
            )
            return
        if all_transforms_6d.shape[0] != self._model_input_info["total_points"]:
            raise ValueError(
                f"Input transform tensor has wrong number of points "
                f"(expected {self._model_input_info['total_points']}, got {all_transforms_6d.shape[0]})"
            )
        if all_transforms_6d.shape[1] != 6:
            raise ValueError(
                "Input transform tensor must have 6 dimensions (axis-angle, translation)"
            )

        all_transforms_6d = all_transforms_6d.to(self.device)  # Ensure device match

        point_slices = self._model_input_info["point_slices"]
        joint_slices = self._model_input_info["joint_slices"]

        for node_idx in self._model_input_info["node_order"]:
            node = self.nodes[node_idx]

            # Store transforms for original points
            start, end = point_slices[node_idx]
            if end > start:
                node.point_transforms = all_transforms_6d[start:end]
            else:
                node.point_transforms = torch.empty(
                    (0, 6), device=self.device
                )  # Handle no points case

            # Store transforms for joint points (stored in this parent node)
            node.joint_transforms_for_child = {}  # Clear previous
            if node_idx in joint_slices:
                for child_idx, (j_start, j_end) in joint_slices[node_idx].items():
                    if j_end > j_start:
                        node.joint_transforms_for_child[child_idx] = all_transforms_6d[
                            j_start:j_end
                        ]
                    else:
                        node.joint_transforms_for_child[child_idx] = torch.empty(
                            (0, 6), device=self.device
                        )

    def generate_model_input_dict(self) -> Dict[str, torch.Tensor]:
        """
        Generates a dictionary of points for model input, keyed by node name.
        Each value tensor contains the node's points concatenated with the
        single joint point positions for each of its children (if any).

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping node name to its input points tensor (N_total, 3).
        """
        output_dict: Dict[str, torch.Tensor] = {}
        if not self.root:
            return output_dict

        for name, node in self.nodes_by_name.items():
            node_input_points: List[torch.Tensor] = []

            # 1. Add original points
            if node.num_points > 0:
                node_input_points.append(node.points)

            # 2. Add stored joint points (for children), ensure consistent order
            # Use sorted items for deterministic joint order
            for child_idx, joint_pos_tensor in sorted(
                node.child_joint_positions.items()
            ):
                if joint_pos_tensor.shape[0] > 0:  # Should be shape (1, 3)
                    node_input_points.append(joint_pos_tensor)

            # 3. Concatenate if list is not empty
            if node_input_points:
                output_dict[name] = torch.cat(node_input_points, dim=0)
            else:
                # Handle nodes with no points and no stored joints
                output_dict[name] = torch.empty(
                    (0, 3), dtype=torch.float32, device=self.device
                )  # Match default dtype

        return output_dict

    def store_model_output_dict(self, transforms_dict: Dict[str, torch.Tensor]):
        """
        Distributes transformations from a dictionary (keyed by node name)
        to the corresponding nodes' point_transforms and joint_transforms_for_child fields.

        Args:
            transforms_dict: Dict mapping node name to its 6D transform tensor (N_total, 6).
                             The tensor shape must match the points generated by generate_model_input_dict.
        """
        if not transforms_dict:
            print("Warning: Input transforms_dict is empty. No data stored.")
            return

        for name, transforms_6d in transforms_dict.items():
            node = self.nodes_by_name.get(name)
            if node is None:
                print(
                    f"Warning: Node '{name}' not found in hierarchy. Skipping its transforms."
                )
                continue

            if transforms_6d.ndim != 2 or transforms_6d.shape[1] != 6:
                print(
                    f"Warning: Invalid shape {transforms_6d.shape} for transforms of node '{name}'. Expected (*, 6). Skipping."
                )
                continue

            transforms_6d = transforms_6d.to(self.device)  # Ensure device match

            num_orig_points = node.num_points
            num_stored_joints = node.num_total_stored_joints
            total_expected_points = num_orig_points + num_stored_joints

            if transforms_6d.shape[0] != total_expected_points:
                print(
                    f"Warning: Shape mismatch for node '{name}'. Expected {total_expected_points} points, got {transforms_6d.shape[0]}. Skipping."
                )
                continue

            # Slice transforms for original points
            if num_orig_points > 0:
                node.point_transforms = transforms_6d[:num_orig_points]
            else:
                node.point_transforms = torch.empty((0, 6), device=self.device)

            # Slice transforms for joint points
            node.joint_transforms_for_child = {}  # Clear previous
            current_joint_idx_start = num_orig_points
            # Use sorted items for deterministic slicing, consistent with generation
            for child_idx, joint_pos_tensor in sorted(
                node.child_joint_positions.items()
            ):
                num_joints_for_child = joint_pos_tensor.shape[0]  # Should be 1 or 0
                if num_joints_for_child > 0:
                    current_joint_idx_end = (
                        current_joint_idx_start + num_joints_for_child
                    )
                    # Slice the single transform for this joint
                    joint_tfm = transforms_6d[
                        current_joint_idx_start:current_joint_idx_end
                    ]  # Shape (1, 6)
                    node.joint_transforms_for_child[child_idx] = joint_tfm
                    current_joint_idx_start = current_joint_idx_end
                else:
                    # Store empty tensor even if no position was stored, for consistency
                    node.joint_transforms_for_child[child_idx] = torch.empty(
                        (0, 6), device=self.device
                    )

    def update_world_transforms(self):
        """
        Calculates world positions and joint transforms for all nodes (simplified for single joint point).
        Requires model output to be stored first via store_model_output().
        """
        if not self.root:
            print("Hierarchy is empty.")
            return

        # Basic check if transforms seem to be stored
        # if not self._model_input_info or self._model_input_info["total_points"] == 0:
        #     print(
        #         "Warning: Model input info not found. Call generate_model_input() first."
        #     )
        # Cannot reliably check if transforms are stored without input info
        # else:
        #     # More robust check might iterate through nodes expected to have transforms
        #     pass

        queue = deque([self.root])
        processed_indices = set()

        while queue:
            current_node = queue.popleft()
            if current_node.part_index in processed_indices:
                continue
            processed_indices.add(current_node.part_index)

            # Clear previous world joint transforms calculated by this node
            current_node.world_joint_transforms = {}

            has_points = current_node.num_points > 0
            local_transforms_6d = (
                current_node.point_transforms
            )  # (N, 6) or None or (0, 6)

            # --- I. Calculate World Points for Current Node ---
            if not has_points:
                current_node.world_points = torch.empty((0, 3), device=self.device)
            elif local_transforms_6d is None or local_transforms_6d.shape[0] == 0:
                print(
                    f"Error: Missing or empty point transforms for node {current_node.name} which has points. Setting world points to local."
                )
                current_node.world_points = current_node.points.clone()  # Fallback
            else:
                # Valid transforms exist for node points
                axis_angle = local_transforms_6d[:, :3]  # (N, 3)
                translation = local_transforms_6d[:, 3:]  # (N, 3)
                local_R = axis_angle_to_matrix(axis_angle)  # (N, 3, 3)

                if current_node.parent is None:  # Root Node
                    rotated_points = torch.bmm(
                        local_R, current_node.points.unsqueeze(-1)
                    ).squeeze(-1)
                    current_node.world_points = rotated_points + translation
                else:  # Non-Root Node
                    parent_node = current_node.parent
                    child_idx = current_node.part_index

                    # --- Get Parent's World Transform at the Joint ---
                    parent_joint_transform = parent_node.world_joint_transforms.get(
                        child_idx
                    )

                    if parent_joint_transform is None:
                        print(
                            f"Error: Parent {parent_node.name} did not have pre-calculated world_joint_transform for child {current_node.name}. Using Identity."
                        )
                        parent_world_R_at_joint = torch.eye(3, device=self.device)
                        parent_world_t_at_joint = torch.zeros(3, device=self.device)
                    else:
                        parent_world_R_at_joint, parent_world_t_at_joint = (
                            parent_joint_transform
                        )

                    # --- Get Child's Local Joint Position ---
                    joint_pos_local_in_child_tensor = (
                        parent_node.child_joint_positions.get(child_idx)
                    )  # Shape (1, 3)
                    if (
                        joint_pos_local_in_child_tensor is None
                        or joint_pos_local_in_child_tensor.shape[0] == 0
                    ):
                        print(
                            f"Warning: Could not find local joint position in child {current_node.name} (relative to parent {parent_node.name}). Using child origin [0,0,0]."
                        )
                        joint_pos_local_in_child = torch.zeros(
                            3, device=self.device
                        )  # Shape (3,)
                    else:
                        joint_pos_local_in_child = (
                            joint_pos_local_in_child_tensor.squeeze(0)
                        )  # Shape (3,)

                    # --- Apply Snippet Logic ---
                    num_pts = current_node.num_points
                    src_centered = (
                        current_node.points - joint_pos_local_in_child
                    )  # (N, 3)
                    rotated_relative_to_joint = torch.bmm(
                        local_R, src_centered.unsqueeze(-1)
                    ).squeeze(
                        -1
                    )  # (N, 3)
                    temp_point_local_frame = (
                        rotated_relative_to_joint + joint_pos_local_in_child
                    )  # (N, 3)

                    parent_R_expanded = parent_world_R_at_joint.unsqueeze(0).expand(
                        num_pts, 3, 3
                    )
                    parent_t_expanded = parent_world_t_at_joint.unsqueeze(0).expand(
                        num_pts, 3
                    )

                    current_node.world_points = (
                        torch.bmm(
                            parent_R_expanded, temp_point_local_frame.unsqueeze(-1)
                        ).squeeze(-1)
                        + parent_t_expanded
                    )

            # --- II. Calculate World Joint Transforms for Children of Current Node ---
            if not current_node.is_leaf:
                # Determine Base Transform for the current node
                if current_node.parent is None:  # Current node is Root
                    base_R = torch.eye(3, device=self.device)
                    base_t = torch.zeros(3, device=self.device)
                else:
                    # Current node is Non-Root, get its connection transform from its parent
                    parent_connection_transform = (
                        current_node.parent.world_joint_transforms.get(
                            current_node.part_index
                        )
                    )
                    if parent_connection_transform is None:
                        print(
                            f"Error: Could not get parent's world joint transform for node {current_node.name} when calculating its children's joints. Using Identity as base."
                        )
                        base_R = torch.eye(3, device=self.device)
                        base_t = torch.zeros(3, device=self.device)
                    else:
                        base_R, base_t = (
                            parent_connection_transform  # This is the world transform where current_node connects
                        )

                # Iterate through joints stored in current_node (for its children)
                for (
                    child_joint_idx,
                    joint_local_tfm_6d,
                ) in current_node.joint_transforms_for_child.items():
                    # joint_local_tfm_6d has shape (1, 6) because k=1

                    if joint_local_tfm_6d is None or joint_local_tfm_6d.shape[0] == 0:
                        print(
                            f"Warning: Missing or empty local transform for joint connecting to child {child_joint_idx} from node {current_node.name}. Cannot calculate world joint transform."
                        )
                        continue

                    # joint_local_pos_tensor = current_node.child_joint_positions.get(
                    #     child_joint_idx
                    # )  # Shape (1, 3)
                    # if (
                    #     joint_local_pos_tensor is None
                    #     or joint_local_pos_tensor.shape[0] == 0
                    # ):
                    #     print(
                    #         f"Warning: Missing or empty local position for joint connecting to child {child_joint_idx} from node {current_node.name}. Using origin [0,0,0]."
                    #     )
                    #     joint_local_pos = torch.zeros(
                    #         3, device=self.device
                    #     )  # Shape (3,)
                    # else:
                    #     joint_local_pos = joint_local_pos_tensor.squeeze(
                    #         0
                    #     )  # Shape (3,)

                    # Get the local R, t for this single joint point
                    joint_axis_angle = joint_local_tfm_6d[:, :3]  # (1, 3)
                    joint_translation = joint_local_tfm_6d[:, 3:]  # (1, 3)
                    # axis_angle_to_matrix expects (N, 3), returns (N, 3, 3)
                    joint_local_R = axis_angle_to_matrix(joint_axis_angle).squeeze(
                        0
                    )  # (3, 3)
                    joint_local_t = joint_translation.squeeze(0)  # (3,)

                    # # Apply unified logic: final = base * local
                    # # 1. Calculate joint position after its local transform (relative to base origin)
                    # transformed_joint_pos_relative_to_base = (
                    #     joint_local_R @ joint_local_pos + joint_local_t
                    # )  # (3,)
                    # 2. Combine rotations and apply base transform
                    if current_node.parent is None:  # Current node is Root
                        final_world_joint_R = joint_local_R
                        final_world_joint_t = joint_local_t
                    else:
                        final_world_joint_R = base_R @ joint_local_R  # (3, 3)
                        final_world_joint_t = (
                            base_t
                            + (base_R - final_world_joint_R) @ joint_pos_local_in_child
                        )
                    # final_world_joint_t = (
                    #     base_R @ transformed_joint_pos_relative_to_base + base_t
                    # )  # (3,)

                    # Store the result for the child to use
                    current_node.world_joint_transforms[child_joint_idx] = (
                        final_world_joint_R,
                        final_world_joint_t,
                    )

            # --- III. Enqueue Children ---
            for child in current_node.children:
                if child.part_index not in processed_indices:
                    queue.append(child)

    def get_joint_positions_stored_in_parent(
        self, child_node_name: str
    ) -> Optional[torch.Tensor]:
        """
        Given the name of a child node, retrieves the joint positions
        that its parent node has stored for it.

        These joint positions are in the local coordinate system of the parent node.

        Args:
            child_node_name: The name of the child node.

        Returns:
            A Tensor (K, 3) of joint positions if found, otherwise None.
            K is the number of joint points defined during hierarchy setup.
        """
        child_node = self.nodes_by_name.get(child_node_name)
        if not child_node:
            print(f"Error: Child node '{child_node_name}' not found in hierarchy.")
            return None

        parent_node = child_node.parent
        if not parent_node:
            # print(f"Info: Node '{child_node_name}' is the root and has no parent.")
            return None

        # The joint positions are stored in the parent, keyed by the child's part_index
        joint_positions = parent_node.child_joint_positions.get(child_node.part_index)

        if joint_positions is None:
            # This might happen if num_joint_points was 0 or joint finding failed for this child
            # print(f"Info: Parent node '{parent_node.name}' does not have stored joint positions for child '{child_node_name}'.")
            return None

        if joint_positions.shape[0] == 0:
            # print(f"Info: Parent node '{parent_node.name}' has empty stored joint positions for child '{child_node_name}'.")
            return torch.empty((0, 3), device=self.device)

        return joint_positions

    # --- Helper Methods ---
    def get_world_points(self, node_name: str) -> Optional[torch.Tensor]:
        """Returns the computed world points for a node by name."""
        if node_name not in self.nodes_by_name:
            print(f"Node '{node_name}' not found.")
            return None
        node = self.nodes_by_name[node_name]
        if node.world_points is None:
            print(f"World points for node '{node_name}' not computed yet.")
        return node.world_points

    def print_hierarchy(self, node: Optional[Node] = None, level: int = 0):
        """Prints the node hierarchy structure."""
        if node is None:
            node = self.root
            if node is None:
                print("Hierarchy is empty.")
                return

        indent = "  " * level
        print(f"{indent}- {node}")  # Uses node's __repr__
        # Optionally print stored joint info
        if node.child_joint_positions:
            joint_info = ", ".join(
                [
                    f"Child {idx}: {pts.shape[0]} pts"
                    for idx, pts in node.child_joint_positions.items()
                ]
            )
            print(f"{indent}  Stored Joints: [{joint_info}]")

        for child in node.children:
            self.print_hierarchy(child, level + 1)


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    USE_GPU = True
    NUM_JOINT_POINTS = 1  # k value for joint finding

    # --- Device Setup ---
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    elif USE_GPU:
        print("Warning: CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"pytorch3d available: {_PYTORCH3D_AVAILABLE}")
    print("-" * 30)

    # 1. Setup Hierarchy
    print("1. Setting up Hierarchy...")
    hierarchy = Hierarchy(device=device)

    # Create some dummy point clouds (replace with actual data)
    root_pts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
        dtype=torch.float32,
    )
    upper_arm_pts = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=torch.float32
    )  # Connects near [1,0,0] of root
    lower_arm_pts = torch.tensor(
        [[1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [1.0, 2.0, 2.0]], dtype=torch.float32
    )  # Connects near [1,0,2] of upper_arm
    hand_pts = torch.tensor(
        [[1.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32
    )  # Connects near [1,2,2] of lower_arm

    hierarchy.add_node("root", points=root_pts, num_joint_points=NUM_JOINT_POINTS)
    hierarchy.add_node(
        "upper_arm",
        points=upper_arm_pts,
        parent_name="root",
        num_joint_points=NUM_JOINT_POINTS,
    )
    hierarchy.add_node(
        "lower_arm",
        points=lower_arm_pts,
        parent_name="upper_arm",
        num_joint_points=NUM_JOINT_POINTS,
    )
    hierarchy.add_node(
        "hand",
        points=hand_pts,
        parent_name="lower_arm",
        num_joint_points=NUM_JOINT_POINTS,
    )

    print("\nHierarchy Structure:")
    hierarchy.print_hierarchy()
    print("-" * 30)

    # 2. Generate Input for the Model
    print("2. Generating Model Input...")
    concatenated_points, mask = hierarchy.generate_model_input()
    print(f"   Model Input Shape: {concatenated_points.shape}")
    print(f"   Mask Shape: {mask.shape}")
    # Example: Verify mask content for joints
    # print("   Mask Details:")
    # for i, node_idx in enumerate(hierarchy._model_input_info['node_order']):
    #     node = hierarchy.nodes[node_idx]
    #     p_start, p_end = hierarchy._model_input_info['point_slices'][node_idx]
    #     print(f"   Node {node.name}({node_idx}): Points slice [{p_start}:{p_end}]")
    #     if node_idx in hierarchy._model_input_info['joint_slices']:
    #         for child_idx, (j_start, j_end) in hierarchy._model_input_info['joint_slices'][node_idx].items():
    #              print(f"     Joints for Child {child_idx}: Slice [{j_start}:{j_end}], Mask value should be {node_idx}")
    #              if j_end > j_start: print(f"       Mask values: {mask[j_start:j_end]}")

    print("-" * 30)

    # 3. Simulate Model Output (Replace with actual model inference)
    print("3. Simulating Model Output...")
    num_total_points = concatenated_points.shape[0]
    # Simple simulation: Small identity-like transform + small unique translation per part
    simulated_transforms = torch.zeros(
        (num_total_points, 6), device=device, dtype=torch.float32
    )
    base_translation = torch.tensor([0.01, 0.02, 0.03], device=device)
    for i, node_idx in enumerate(hierarchy._model_input_info["node_order"]):
        part_mask = mask == node_idx
        # Small rotation (e.g., around Z axis)
        simulated_transforms[part_mask, 2] = 0.05 * (i + 1)  # Axis-angle Z component
        # Unique translation offset
        simulated_transforms[part_mask, 3:] = base_translation * (i + 1)

    print(f"   Simulated Output Shape: {simulated_transforms.shape}")
    print("-" * 30)

    # 4. Store Model Output in Hierarchy
    print("4. Storing Model Output...")
    hierarchy.store_model_output(simulated_transforms)
    # Example check:
    # print("   Stored Root Point Transforms:", hierarchy.nodes[0].point_transforms)
    # print("   Stored Root Joint Transforms:", hierarchy.nodes[0].joint_transforms_for_child)
    print("-" * 30)

    # 5. Update World Transforms
    print("5. Updating World Transforms...")
    try:
        hierarchy.update_world_transforms()
        print("   Update Complete.")
    except Exception as e:
        print(f"\n   An error occurred during world transform update: {e}")
        import traceback

        traceback.print_exc()
    print("-" * 30)

    # 6. Access Results
    print("6. Accessing Results...")
    torch.set_printoptions(precision=3, sci_mode=False)

    for node_idx in hierarchy._model_input_info["node_order"]:
        node = hierarchy.nodes[node_idx]
        print(f"\n--- {node.name} (Index: {node.part_index}) ---")
        print("Local Points (Original):")
        print(node.points)
        if node.world_points is not None:
            print("World Points (Calculated):")
            print(node.world_points)
        else:
            print("World Points: Not Computed")
        # Print computed world joint transforms *for its children*
        if node.world_joint_transforms:
            print("Computed World Joint Transforms (for children):")
            for c_idx, (R, t) in node.world_joint_transforms.items():
                print(f"  Child {c_idx}: R=\n{R}\n    t={t}")

    print("\nExample: Get hand world points")
    hand_wp = hierarchy.get_world_points("hand")
    if hand_wp is not None:
        print(hand_wp)
