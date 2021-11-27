pub type Index = i32;
pub type Real = f32;

#[allow(dead_code)]
pub enum ObjectiveType {
    CUT,
    VOLUME,
    NODE,
}

#[allow(dead_code)]
pub enum RefinementType {
    FM,
    GREEDY,
    SEP2SIDED,
    SEP1SIDED,
}

#[allow(dead_code)]
pub enum InitialPartitioningType {
    GROW,
    RANDOM,
    EDGE,
    NODE,
    METISRB,
}

#[allow(dead_code)]
pub enum CoarseningType {
    RM,
    SHEM,
}

pub struct Config {
    pub objective_type: ObjectiveType,
    pub refinement_type: RefinementType,
    pub initial_partitioning_type: InitialPartitioningType,
    pub n_separators: Index,
    pub n_iterations: Index,
    pub user_load_imbalance_factor: Index,
    pub compress_before_ordering: bool,
    pub connected_components_order: bool,
    pub pruning_factor: Real,
    pub coarsen_to: Index,

    pub coarsening_type: CoarseningType,
    pub two_hop_matching: bool,
    pub on_disk: bool,
    pub seed: Index,
    pub debug_level: Index,
    pub num_flag: Index,
    pub drop_edges: bool,

    pub n_balancing_constraints: Index,
    pub n_parts: Index,
    pub max_vertex_weight: Vec<Index>,

    pub target_partition_weights: Vec<Real>,

    pub ub_factors: Vec<Real>,
}

impl Config {
    pub fn single_separator_threshold_node_bisection_multiple(&self) -> usize {
        if self.compress_before_ordering {
            1000
        } else {
            2000
        }
    }

    pub fn single_separator_threshold_node_bisection_l2(&self) -> usize {
        5000
    }

    pub fn init_n_i_parts(&self) -> usize {
        7
    }

    pub fn coarsen_fraction(&self) -> f32 {
        0.85
    }

    pub fn unmatched_for_two_hop(&self) -> f32 {
        0.1
    }
}

pub fn default_config() -> Config {
    const N_BALANCING_CONSTRAINTS: Index = 1;
    const N_PARTS: Index = 3;

    fn i2rubfactor(user_load_imbalance_factor: Index) -> Real {
        1.0 + 0.001 * (user_load_imbalance_factor as Real)
    }

    let user_load_imbalance_factor = 200;

    Config {
        objective_type: ObjectiveType::NODE,
        refinement_type: RefinementType::SEP1SIDED,
        initial_partitioning_type: InitialPartitioningType::EDGE,
        n_separators: 1,
        n_iterations: 10,
        user_load_imbalance_factor: user_load_imbalance_factor,
        compress_before_ordering: true,
        connected_components_order: false,
        pruning_factor: 0.0,
        coarsen_to: 100,
        coarsening_type: CoarseningType::SHEM,
        two_hop_matching: true,
        on_disk: false,
        seed: -1,
        debug_level: 0,
        num_flag: 0,
        drop_edges: false,
        n_balancing_constraints: N_BALANCING_CONSTRAINTS,
        n_parts: N_PARTS,
        max_vertex_weight: vec![0; N_BALANCING_CONSTRAINTS as usize],
        target_partition_weights: vec![0.5, 0.5],
        ub_factors: vec![
            i2rubfactor(user_load_imbalance_factor) + 0.0000499;
            N_BALANCING_CONSTRAINTS as usize
        ],
    }
}
