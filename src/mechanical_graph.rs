//! Mechanically oriented skeleton-graph descriptors.
//!
//! These quantities are graph/path/transport surrogates, not finite-element
//! stiffness. Branch identities and physical lengths come from the exact
//! Fiji/AnalyzeSkeleton-compatible state machine exposed by
//! [`crate::skeleton::compressed_graph_fiji`].

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use anyhow::{bail, Result};

#[cfg(test)]
use crate::skeleton::CompressedGraphEdge;
use crate::skeleton::{compressed_graph_fiji, voxel_xyz, FijiCompressedGraph};
use crate::volume::BinaryVolume;

/// Production values plus QC diagnostics.
/// Some QC fields are intentionally retained even when not emitted by the CSV preset.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct MechanicalGraphStats {
    pub axial_shortest_path_tortuosity: f64,
    pub axial_edge_connectivity: f64,
    pub cz_path_length_fraction: f64,
    pub axial_normalized_conductance: f64,
    pub axial_dissipation_gini: f64,
    pub axial_span_um: f64,
    pub top_terminal_nodes: usize,
    pub bottom_terminal_nodes: usize,
    pub electrical_energy_identity_relative_error: f64,
}

/// Compute the five-variable mechanical-graph descriptor block.
///
/// Terminal vertices are compressed graph nodes touching a one-voxel cap at
/// the minimum or maximum occupied skeleton z coordinate. Parallel branches
/// retain their multiplicity for edge connectivity and conductance. Self loops
/// contribute to total graph length and dissipation Gini exactly as in the
/// reference implementation, but they cannot participate in a top-bottom path.
pub fn mechanical_graph_stats(
    skeleton: &BinaryVolume,
    spacing: [f64; 3],
) -> Result<MechanicalGraphStats> {
    if spacing.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        bail!("mechanical graph requires finite positive spacing");
    }
    let tol = 1e-9 * spacing[0].abs().max(1.0);
    if (spacing[1] - spacing[0]).abs() > tol || (spacing[2] - spacing[0]).abs() > tol {
        bail!("mechanical-graph descriptors require isotropic spacing");
    }

    let graph = compressed_graph_fiji(skeleton, spacing);
    compute_metrics(&graph, skeleton, spacing[2])
}

fn compute_metrics(
    graph: &FijiCompressedGraph,
    volume: &BinaryVolume,
    h_z: f64,
) -> Result<MechanicalGraphStats> {
    let n = graph.node_voxels.len();
    if graph.skeleton_voxels.is_empty() || n == 0 {
        bail!("mechanical graph requires a nonempty compressed skeleton graph");
    }

    let mut z_min = usize::MAX;
    let mut z_max = 0usize;
    for &index in &graph.skeleton_voxels {
        let (_, _, z) = voxel_xyz(volume, index);
        z_min = z_min.min(z);
        z_max = z_max.max(z);
    }

    // Terminal definition: any constituent node voxel within
    // one voxel of the occupied z extreme belongs to the corresponding cap.
    let top: HashSet<usize> = graph
        .node_voxels
        .iter()
        .enumerate()
        .filter_map(|(node, voxels)| {
            let node_min_z = voxels
                .iter()
                .map(|&index| voxel_xyz(volume, index).2)
                .min()
                .expect("compressed graph node is nonempty");
            (node_min_z <= z_min + 1).then_some(node)
        })
        .collect();
    let bottom: HashSet<usize> = graph
        .node_voxels
        .iter()
        .enumerate()
        .filter_map(|(node, voxels)| {
            let node_max_z = voxels
                .iter()
                .map(|&index| voxel_xyz(volume, index).2)
                .max()
                .expect("compressed graph node is nonempty");
            (node_max_z >= z_max.saturating_sub(1)).then_some(node)
        })
        .collect();

    if top.is_empty() || bottom.is_empty() {
        bail!("missing axial graph terminals");
    }
    if top.iter().any(|node| bottom.contains(node)) {
        bail!("top and bottom terminal sets overlap");
    }

    let axial_span_um = (z_max - z_min) as f64 * h_z;

    // Collapse to a simple graph only where the mathematical definition calls
    // for it. Preserve branch multiplicity and parallel conductance separately.
    let mut multiplicity = HashMap::<(usize, usize), usize>::new();
    let mut min_length = HashMap::<(usize, usize), f64>::new();
    let mut conductance = HashMap::<(usize, usize), f64>::new();
    let mut simple_adj = vec![Vec::<usize>::new(); n];
    let mut seen_pairs = HashSet::new();

    for edge in &graph.edges {
        if edge.u == edge.v {
            continue;
        }
        let p = pair(edge.u, edge.v);
        *multiplicity.entry(p).or_insert(0) += 1;
        min_length
            .entry(p)
            .and_modify(|value| *value = value.min(edge.length_um))
            .or_insert(edge.length_um);
        if edge.length_um > 0.0 {
            *conductance.entry(p).or_insert(0.0) += 1.0 / edge.length_um;
        }
        if seen_pairs.insert(p) {
            simple_adj[p.0].push(p.1);
            simple_adj[p.1].push(p.0);
        }
    }

    if !any_terminal_path(&simple_adj, &top, &bottom) {
        bail!("no top-bottom path in compressed graph");
    }

    let shortest = shortest_terminal_path(&simple_adj, &min_length, &top, &bottom)?;
    let shortest_tortuosity = if axial_span_um > 0.0 {
        shortest / axial_span_um
    } else {
        f64::NAN
    };
    let edge_connectivity = maxflow_edge_connectivity(n, &multiplicity, &top, &bottom) as f64;

    // C_z is the fraction of total AnalyzeSkeleton branch length that lies in
    // biconnected blocks on the augmented source-to-sink block-cut-tree path.
    let relevant_pairs = relevant_st_pairs(n, &simple_adj, &top, &bottom)?;
    let total_length: f64 = graph.edges.iter().map(|edge| edge.length_um).sum();
    let relevant_length: f64 = graph
        .edges
        .iter()
        .filter(|edge| {
            edge.u != edge.v && relevant_pairs.contains(&pair(edge.u, edge.v))
        })
        .map(|edge| edge.length_um)
        .sum();
    let cz_path_length_fraction = if total_length > 0.0 {
        relevant_length / total_length
    } else {
        f64::NAN
    };

    // Unit-resistor transport: each physical branch has conductance 1/L.
    let potentials = solve_potentials(n, &simple_adj, &conductance, &top, &bottom)?;
    let mut dissipation = Vec::with_capacity(graph.edges.len());
    for edge in &graph.edges {
        let current = if edge.u == edge.v || edge.length_um <= 0.0 {
            0.0
        } else {
            (potentials[edge.u] - potentials[edge.v]).abs() / edge.length_um
        };
        dissipation.push(current * current * edge.length_um);
    }

    // Effective conductance is the total source current at unit potential drop.
    let mut effective_conductance = 0.0;
    for &terminal in &top {
        for &neighbor in &simple_adj[terminal] {
            if top.contains(&neighbor) {
                continue;
            }
            effective_conductance +=
                conductance[&pair(terminal, neighbor)] * (1.0 - potentials[neighbor]);
        }
    }

    let total_dissipation: f64 = dissipation.iter().sum();
    let energy_error = (total_dissipation - effective_conductance).abs()
        / effective_conductance.abs().max(1e-30);

    Ok(MechanicalGraphStats {
        axial_shortest_path_tortuosity: shortest_tortuosity,
        axial_edge_connectivity: edge_connectivity,
        cz_path_length_fraction,
        axial_normalized_conductance: effective_conductance * axial_span_um,
        axial_dissipation_gini: gini(&dissipation),
        axial_span_um,
        top_terminal_nodes: top.len(),
        bottom_terminal_nodes: bottom.len(),
        electrical_energy_identity_relative_error: energy_error,
    })
}

#[inline]
fn pair(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn any_terminal_path(
    adj: &[Vec<usize>],
    top: &HashSet<usize>,
    bottom: &HashSet<usize>,
) -> bool {
    let mut seen = vec![false; adj.len()];
    let mut queue = VecDeque::new();
    for &terminal in top {
        seen[terminal] = true;
        queue.push_back(terminal);
    }

    while let Some(u) = queue.pop_front() {
        if bottom.contains(&u) {
            return true;
        }
        for &v in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                queue.push_back(v);
            }
        }
    }
    false
}

#[derive(Clone, Copy)]
struct HeapState {
    cost: f64,
    node: usize,
}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.cost.to_bits() == other.cost.to_bits()
    }
}
impl Eq for HeapState {}
impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

fn shortest_terminal_path(
    adj: &[Vec<usize>],
    lengths: &HashMap<(usize, usize), f64>,
    top: &HashSet<usize>,
    bottom: &HashSet<usize>,
) -> Result<f64> {
    let mut distance = vec![f64::INFINITY; adj.len()];
    let mut heap = BinaryHeap::new();
    for &terminal in top {
        distance[terminal] = 0.0;
        heap.push(HeapState {
            cost: 0.0,
            node: terminal,
        });
    }

    while let Some(HeapState { cost, node: u }) = heap.pop() {
        if cost > distance[u] {
            continue;
        }
        if bottom.contains(&u) {
            return Ok(cost);
        }
        for &v in &adj[u] {
            let next = cost + lengths[&pair(u, v)];
            if next < distance[v] {
                distance[v] = next;
                heap.push(HeapState { cost: next, node: v });
            }
        }
    }
    bail!("no top-bottom shortest path")
}

/// Maximum number of edge-disjoint top-bottom paths with unit capacity per
/// AnalyzeSkeleton branch. Parallel branches therefore contribute capacity.
fn maxflow_edge_connectivity(
    n: usize,
    multiplicity: &HashMap<(usize, usize), usize>,
    top: &HashSet<usize>,
    bottom: &HashSet<usize>,
) -> usize {
    let source = n;
    let sink = n + 1;
    let node_count = n + 2;
    let terminal_capacity = multiplicity.values().sum::<usize>() + 1;
    let mut capacity = HashMap::<(usize, usize), i64>::new();
    let mut adj = vec![Vec::<usize>::new(); node_count];

    let mut add_arc = |u: usize, v: usize, c: i64| {
        if !adj[u].contains(&v) {
            adj[u].push(v);
        }
        if !adj[v].contains(&u) {
            adj[v].push(u);
        }
        *capacity.entry((u, v)).or_insert(0) += c;
        capacity.entry((v, u)).or_insert(0);
    };

    for (&(u, v), &count) in multiplicity {
        add_arc(u, v, count as i64);
        add_arc(v, u, count as i64);
    }
    for &u in top {
        add_arc(source, u, terminal_capacity as i64);
    }
    for &u in bottom {
        add_arc(u, sink, terminal_capacity as i64);
    }

    let mut flow = 0i64;
    loop {
        let mut parent = vec![usize::MAX; node_count];
        parent[source] = source;
        let mut queue = VecDeque::from([source]);

        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if parent[v] == usize::MAX && *capacity.get(&(u, v)).unwrap_or(&0) > 0 {
                    parent[v] = u;
                    queue.push_back(v);
                    if v == sink {
                        break;
                    }
                }
            }
            if parent[sink] != usize::MAX {
                break;
            }
        }
        if parent[sink] == usize::MAX {
            break;
        }

        let mut augment = i64::MAX;
        let mut v = sink;
        while v != source {
            let u = parent[v];
            augment = augment.min(capacity[&(u, v)]);
            v = u;
        }
        v = sink;
        while v != source {
            let u = parent[v];
            *capacity
                .get_mut(&(u, v))
                .expect("residual edge must exist") -= augment;
            *capacity.entry((v, u)).or_insert(0) += augment;
            v = u;
        }
        flow += augment;
    }

    flow.max(0) as usize
}

/// Biconnected edge blocks of an undirected simple graph (Tarjan algorithm).
fn biconnected_blocks(adj: &[Vec<usize>]) -> Vec<Vec<(usize, usize)>> {
    #[allow(clippy::too_many_arguments)]
    fn dfs(
        u: usize,
        parent: usize,
        adj: &[Vec<usize>],
        time: &mut usize,
        discovery: &mut [usize],
        low: &mut [usize],
        stack: &mut Vec<(usize, usize)>,
        blocks: &mut Vec<Vec<(usize, usize)>>,
    ) {
        *time += 1;
        discovery[u] = *time;
        low[u] = *time;

        for &v in &adj[u] {
            if discovery[v] == 0 {
                stack.push((u, v));
                dfs(v, u, adj, time, discovery, low, stack, blocks);
                low[u] = low[u].min(low[v]);
                if low[v] >= discovery[u] {
                    let mut block = Vec::new();
                    while let Some(edge) = stack.pop() {
                        let stop = pair(edge.0, edge.1) == pair(u, v);
                        block.push(edge);
                        if stop {
                            break;
                        }
                    }
                    if !block.is_empty() {
                        blocks.push(block);
                    }
                }
            } else if v != parent && discovery[v] < discovery[u] {
                low[u] = low[u].min(discovery[v]);
                stack.push((u, v));
            }
        }
    }

    let n = adj.len();
    let mut discovery = vec![0usize; n];
    let mut low = vec![0usize; n];
    let mut time = 0usize;
    let mut stack = Vec::new();
    let mut blocks = Vec::new();

    for u in 0..n {
        if discovery[u] == 0 {
            dfs(
                u,
                usize::MAX,
                adj,
                &mut time,
                &mut discovery,
                &mut low,
                &mut stack,
                &mut blocks,
            );
            if !stack.is_empty() {
                blocks.push(std::mem::take(&mut stack));
            }
        }
    }
    blocks
}

/// Return original simple-edge pairs that belong to a biconnected block on the
/// unique source-to-sink path of the augmented block-cut tree.
fn relevant_st_pairs(
    n: usize,
    base_adj: &[Vec<usize>],
    top: &HashSet<usize>,
    bottom: &HashSet<usize>,
) -> Result<HashSet<(usize, usize)>> {
    let source = n;
    let sink = n + 1;
    let mut adj = vec![Vec::<usize>::new(); n + 2];
    for u in 0..n {
        for &v in &base_adj[u] {
            if !adj[u].contains(&v) {
                adj[u].push(v);
            }
        }
    }

    let mut connect = |u: usize, v: usize| {
        if !adj[u].contains(&v) {
            adj[u].push(v);
        }
        if !adj[v].contains(&u) {
            adj[v].push(u);
        }
    };
    for &u in top {
        connect(source, u);
    }
    for &u in bottom {
        connect(u, sink);
    }

    let blocks = biconnected_blocks(&adj);
    let mut membership = vec![Vec::<usize>::new(); n + 2];
    let mut block_pairs = Vec::<HashSet<(usize, usize)>>::new();
    for (block_index, edges) in blocks.iter().enumerate() {
        let mut pairs = HashSet::new();
        let mut nodes = HashSet::new();
        for &(u, v) in edges {
            pairs.insert(pair(u, v));
            nodes.insert(u);
            nodes.insert(v);
        }
        for v in nodes {
            membership[v].push(block_index);
        }
        block_pairs.push(pairs);
    }

    if membership[source].is_empty() || membership[sink].is_empty() {
        bail!("block-cut decomposition cannot represent terminals");
    }

    let block_count = blocks.len();
    let articulations: HashSet<usize> = (0..n + 2)
        .filter(|&v| membership[v].len() > 1)
        .collect();

    // Block-cut-tree node ids: blocks are 0..block_count and articulation
    // vertices are block_count + original vertex id.
    let mut tree = vec![Vec::<usize>::new(); block_count + n + 2];
    for &v in &articulations {
        let articulation_node = block_count + v;
        for &block_index in &membership[v] {
            tree[articulation_node].push(block_index);
            tree[block_index].push(articulation_node);
        }
    }

    let representative = |v: usize| -> Result<usize> {
        if articulations.contains(&v) {
            Ok(block_count + v)
        } else if membership[v].len() == 1 {
            Ok(membership[v][0])
        } else {
            bail!("block-cut membership ambiguity for vertex {v}")
        }
    };

    let source_rep = representative(source)?;
    let sink_rep = representative(sink)?;
    let mut parent = vec![usize::MAX; tree.len()];
    parent[source_rep] = source_rep;
    let mut queue = VecDeque::from([source_rep]);
    while let Some(u) = queue.pop_front() {
        if u == sink_rep {
            break;
        }
        for &v in &tree[u] {
            if parent[v] == usize::MAX {
                parent[v] = u;
                queue.push_back(v);
            }
        }
    }
    if parent[sink_rep] == usize::MAX {
        bail!("no path in block-cut tree");
    }

    let mut relevant_blocks = HashSet::new();
    let mut current = sink_rep;
    loop {
        if current < block_count {
            relevant_blocks.insert(current);
        }
        if current == source_rep {
            break;
        }
        current = parent[current];
    }

    let mut output = HashSet::new();
    for block_index in relevant_blocks {
        for &(u, v) in &block_pairs[block_index] {
            if u != source && v != source && u != sink && v != sink {
                output.insert(pair(u, v));
            }
        }
    }
    Ok(output)
}

fn connected_components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; adj.len()];
    let mut output = Vec::new();
    for start in 0..adj.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some(u) = queue.pop_front() {
            component.push(u);
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    queue.push_back(v);
                }
            }
        }
        output.push(component);
    }
    output
}

fn solve_potentials(
    n: usize,
    adj: &[Vec<usize>],
    conductance: &HashMap<(usize, usize), f64>,
    top: &HashSet<usize>,
    bottom: &HashSet<usize>,
) -> Result<Vec<f64>> {
    let mut voltage = vec![0.5; n];
    for component in connected_components(adj) {
        let component_top: HashSet<_> = component
            .iter()
            .copied()
            .filter(|v| top.contains(v))
            .collect();
        let component_bottom: HashSet<_> = component
            .iter()
            .copied()
            .filter(|v| bottom.contains(v))
            .collect();

        if !component_top.is_empty() && !component_bottom.is_empty() {
            for &v in &component_top {
                voltage[v] = 1.0;
            }
            for &v in &component_bottom {
                voltage[v] = 0.0;
            }
            let free: Vec<_> = component
                .iter()
                .copied()
                .filter(|v| !component_top.contains(v) && !component_bottom.contains(v))
                .collect();
            if !free.is_empty() {
                let position: HashMap<usize, usize> = free
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (v, i))
                    .collect();
                let mut diagonal = vec![0.0; free.len()];
                let mut rhs = vec![0.0; free.len()];
                let mut neighbors = vec![Vec::<(usize, f64)>::new(); free.len()];

                for (i, &u) in free.iter().enumerate() {
                    for &v in &adj[u] {
                        let g = conductance[&pair(u, v)];
                        diagonal[i] += g;
                        if let Some(&j) = position.get(&v) {
                            neighbors[i].push((j, g));
                        } else {
                            rhs[i] += g * voltage[v];
                        }
                    }
                }

                let solution = cg_solve(&diagonal, &neighbors, &rhs)?;
                for (i, &u) in free.iter().enumerate() {
                    voltage[u] = solution[i];
                }
            }
        } else if !component_top.is_empty() {
            for &v in &component {
                voltage[v] = 1.0;
            }
        } else if !component_bottom.is_empty() {
            for &v in &component {
                voltage[v] = 0.0;
            }
        }
    }
    Ok(voltage)
}

/// Dependency-free preconditioned conjugate-gradient solve for the positive
/// definite free-node Dirichlet graph Laplacian.
fn cg_solve(diagonal: &[f64], neighbors: &[Vec<(usize, f64)>], b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    let matvec = |x: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                diagonal[i] * x[i]
                    - neighbors[i]
                        .iter()
                        .map(|&(j, g)| g * x[j])
                        .sum::<f64>()
            })
            .collect()
    };

    let mut x = vec![0.0; n];
    let mut residual = b.to_vec();
    let mut z: Vec<f64> = residual
        .iter()
        .enumerate()
        .map(|(i, &value)| value / diagonal[i].max(1e-30))
        .collect();
    let mut direction = z.clone();
    let mut rz: f64 = residual
        .iter()
        .zip(&z)
        .map(|(a, b)| a * b)
        .sum();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);

    for _ in 0..n.saturating_mul(20).max(200) {
        let a_direction = matvec(&direction);
        let p_ap: f64 = direction
            .iter()
            .zip(&a_direction)
            .map(|(a, b)| a * b)
            .sum();
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / p_ap;
        for i in 0..n {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * a_direction[i];
        }
        if residual.iter().map(|v| v * v).sum::<f64>().sqrt() <= 1e-12 * b_norm {
            return Ok(x);
        }

        z = residual
            .iter()
            .enumerate()
            .map(|(i, &value)| value / diagonal[i].max(1e-30))
            .collect();
        let rz_new: f64 = residual
            .iter()
            .zip(&z)
            .map(|(a, b)| a * b)
            .sum();
        let beta = rz_new / rz;
        for i in 0..n {
            direction[i] = z[i] + beta * direction[i];
        }
        rz = rz_new;
    }

    let final_residual = matvec(&x)
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    if final_residual > 1e-9 * b_norm {
        bail!(
            "mechanical graph resistor solve did not converge: relative residual {}",
            final_residual / b_norm
        );
    }
    Ok(x)
}

fn gini(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted: Vec<f64> = values
        .iter()
        .map(|&value| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                0.0
            }
        })
        .collect();
    let sum: f64 = sorted.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    let weighted: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, &value)| (i as f64 + 1.0) * value)
        .sum();
    2.0 * weighted / (n * sum) - (n + 1.0) / n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_graph(
        z_coordinates: &[usize],
        edges: &[(usize, usize, f64)],
    ) -> (FijiCompressedGraph, BinaryVolume) {
        let depth = z_coordinates.iter().copied().max().unwrap() + 2;
        let volume = BinaryVolume::new(vec![1; depth * 2 * 2], 2, 2, depth).unwrap();
        let node_voxels = z_coordinates
            .iter()
            .map(|&z| vec![volume.index(0, 0, z)])
            .collect();
        let graph_edges = edges
            .iter()
            .map(|&(u, v, length_um)| CompressedGraphEdge {
                u,
                v,
                length_um,
                pure_cycle: false,
            })
            .collect();
        let skeleton_voxels = z_coordinates
            .iter()
            .map(|&z| volume.index(0, 0, z))
            .collect();
        (
            FijiCompressedGraph {
                node_voxels,
                edges: graph_edges,
                skeleton_voxels,
            },
            volume,
        )
    }

    #[test]
    fn stage5_chain() {
        let (graph, volume) = fake_graph(&[0, 2, 4], &[(0, 1, 2.0), (1, 2, 2.0)]);
        let metrics = compute_metrics(&graph, &volume, 1.0).unwrap();
        assert!((metrics.cz_path_length_fraction - 1.0).abs() < 1e-12);
        assert_eq!(metrics.axial_edge_connectivity, 1.0);
        assert!((metrics.axial_shortest_path_tortuosity - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stage5_parallel_routes() {
        let length = 5.0_f64.sqrt();
        let (graph, volume) = fake_graph(
            &[0, 2, 2, 4],
            &[
                (0, 1, length),
                (1, 3, length),
                (0, 2, length),
                (2, 3, length),
            ],
        );
        let metrics = compute_metrics(&graph, &volume, 1.0).unwrap();
        assert_eq!(metrics.axial_edge_connectivity, 2.0);
        assert!((metrics.cz_path_length_fraction - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stage5_chain_with_dangling_leaf() {
        let (graph, volume) = fake_graph(
            &[0, 2, 4, 2],
            &[(0, 1, 2.0), (1, 2, 2.0), (1, 3, 1.0)],
        );
        let metrics = compute_metrics(&graph, &volume, 1.0).unwrap();
        assert!((metrics.cz_path_length_fraction - 0.8).abs() < 1e-12);
    }
}
