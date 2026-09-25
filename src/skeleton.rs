use std::collections::{HashSet, VecDeque};

use crate::volume::BinaryVolume;

const N26: [(isize, isize, isize); 26] = make_n26();
const N6: [(isize, isize, isize); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

const fn make_n26() -> [(isize, isize, isize); 26] {
    let mut out = [(0, 0, 0); 26];
    let mut k = 0usize;
    let mut dz = -1isize;
    while dz <= 1 {
        let mut dy = -1isize;
        while dy <= 1 {
            let mut dx = -1isize;
            while dx <= 1 {
                if !(dx == 0 && dy == 0 && dz == 0) {
                    out[k] = (dx, dy, dz);
                    k += 1;
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct GraphStats {
    pub skeleton_voxels: usize,
    pub endpoint_voxels: usize,
    pub slab_voxels: usize,
    pub junction_voxels: usize,
    pub junction_clusters: usize,
    pub branches: usize,
    pub graph_components: usize,
    pub cycle_rank: isize,
    pub mean_branch_length: f64,
    pub max_branch_length: f64,
    pub mean_branch_tortuosity: f64,
    pub top_bottom_spanning: bool,
}

/// Lee et al. 3-D medial-axis thinning, matching Fiji Skeletonize3D.
///
/// Reference semantics:
/// - six directional border passes in N,S,E,W,U,B order;
/// - initial endpoint preservation;
/// - Euler-invariant candidate test;
/// - 26-connected foreground simple-point test;
/// - sequential simple-point re-check before deletion.
///
/// This replaces the earlier stricter local 26-FG/6-BG component test, which
/// leaves large medial sheets on complex trabecular volumes.
pub fn skeletonize(volume: &BinaryVolume) -> BinaryVolume {
    let mut data = volume
        .data
        .iter()
        .map(|&v| u8::from(v != 0))
        .collect::<Vec<_>>();

    const LEE_BORDERS: [(isize, isize, isize); 6] = [
        (0, -1, 0),
        (0, 1, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    let euler_lut = lee_euler_lut();

    loop {
        let mut unchanged_borders = 0usize;

        for &(bdx, bdy, bdz) in &LEE_BORDERS {
            let mut candidates = Vec::<usize>::new();

            // Fiji loops z, then y, then x.
            for z in 0..volume.depth {
                for y in 0..volume.height {
                    for x in 0..volume.width {
                        let idx = volume.index(x, y, z);
                        if data[idx] == 0 {
                            continue;
                        }

                        if get_data(
                            &data,
                            volume,
                            x as isize + bdx,
                            y as isize + bdy,
                            z as isize + bdz,
                        ) != 0
                        {
                            continue;
                        }

                        // Fiji's initial scan preserves points with exactly one
                        // foreground neighbour.
                        if foreground_neighbour_count(&data, volume, x, y, z) == 1 {
                            continue;
                        }

                        let neighborhood = lee_neighborhood(&data, volume, x, y, z);
                        if !lee_is_euler_invariant(&neighborhood, &euler_lut) {
                            continue;
                        }
                        if !lee_is_simple_point(&neighborhood) {
                            continue;
                        }
                        candidates.push(idx);
                    }
                }
            }

            let mut changed_border = false;

            // Fiji intentionally re-checks only the simple-point condition here.
            for idx in candidates {
                if data[idx] == 0 {
                    continue;
                }
                let (x, y, z) = xyz(volume, idx);
                let neighborhood = lee_neighborhood(&data, volume, x, y, z);
                if lee_is_simple_point(&neighborhood) {
                    data[idx] = 0;
                    changed_border = true;
                }
            }

            if !changed_border {
                unchanged_borders += 1;
            }
        }

        if unchanged_borders == 6 {
            break;
        }
    }

    BinaryVolume::new(data, volume.width, volume.height, volume.depth)
        .expect("skeleton preserves volume dimensions")
}

fn lee_neighborhood(data: &[u8], volume: &BinaryVolume, x: usize, y: usize, z: usize) -> [u8; 27] {
    let mut n = [0u8; 27];
    let mut k = 0usize;
    for dz in -1isize..=1 {
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                n[k] = u8::from(
                    get_data(
                        data,
                        volume,
                        x as isize + dx,
                        y as isize + dy,
                        z as isize + dz,
                    ) != 0,
                );
                k += 1;
            }
        }
    }
    n
}

fn lee_is_simple_point(neighborhood: &[u8; 27]) -> bool {
    // Lee94 N(v)_labeling / Fiji isSimplePoint counts 26-connected
    // foreground components after deleting the centre.
    let mut fg = [false; 27];
    for i in 0..27 {
        fg[i] = i != 13 && neighborhood[i] != 0;
    }
    component_count_local(&fg, true) <= 1
}

fn lee_is_euler_invariant(neighborhood: &[u8; 27], lut: &[i8; 256]) -> bool {
    const OCTANTS: [[usize; 7]; 8] = [
        [24, 25, 15, 16, 21, 22, 12], // SWU
        [26, 23, 17, 14, 25, 22, 16], // SEU
        [18, 21, 9, 12, 19, 22, 10],  // NWU
        [20, 23, 19, 22, 11, 14, 10], // NEU
        [6, 15, 7, 16, 3, 12, 4],     // SWB
        [8, 7, 17, 16, 5, 4, 14],     // SEB
        [0, 9, 3, 12, 1, 10, 4],      // NWB
        [2, 1, 11, 10, 5, 4, 14],     // NEB
    ];
    const BITS: [usize; 7] = [128, 64, 32, 16, 8, 4, 2];

    let mut euler = 0i32;
    for octant in OCTANTS {
        let mut index = 1usize;
        for j in 0..7 {
            if neighborhood[octant[j]] != 0 {
                index |= BITS[j];
            }
        }
        euler += lut[index] as i32;
    }
    euler == 0
}

fn lee_euler_lut() -> [i8; 256] {
    // Exact Lee94/Fiji Skeletonize3D Euler LUT. Even entries remain zero.
    let mut lut = [0i8; 256];
    let entries: &[(usize, i8)] = &[
        (1, 1),
        (3, -1),
        (5, -1),
        (7, 1),
        (9, -3),
        (11, -1),
        (13, -1),
        (15, 1),
        (17, -1),
        (19, 1),
        (21, 1),
        (23, -1),
        (25, 3),
        (27, 1),
        (29, 1),
        (31, -1),
        (33, -3),
        (35, -1),
        (37, 3),
        (39, 1),
        (41, 1),
        (43, -1),
        (45, 3),
        (47, 1),
        (49, -1),
        (51, 1),
        (53, 1),
        (55, -1),
        (57, 3),
        (59, 1),
        (61, 1),
        (63, -1),
        (65, -3),
        (67, 3),
        (69, -1),
        (71, 1),
        (73, 1),
        (75, 3),
        (77, -1),
        (79, 1),
        (81, -1),
        (83, 1),
        (85, 1),
        (87, -1),
        (89, 3),
        (91, 1),
        (93, 1),
        (95, -1),
        (97, 1),
        (99, 3),
        (101, 3),
        (103, 1),
        (105, 5),
        (107, 3),
        (109, 3),
        (111, 1),
        (113, -1),
        (115, 1),
        (117, 1),
        (119, -1),
        (121, 3),
        (123, 1),
        (125, 1),
        (127, -1),
        (129, -7),
        (131, -1),
        (133, -1),
        (135, 1),
        (137, -3),
        (139, -1),
        (141, -1),
        (143, 1),
        (145, -1),
        (147, 1),
        (149, 1),
        (151, -1),
        (153, 3),
        (155, 1),
        (157, 1),
        (159, -1),
        (161, -3),
        (163, -1),
        (165, 3),
        (167, 1),
        (169, 1),
        (171, -1),
        (173, 3),
        (175, 1),
        (177, -1),
        (179, 1),
        (181, 1),
        (183, -1),
        (185, 3),
        (187, 1),
        (189, 1),
        (191, -1),
        (193, -3),
        (195, 3),
        (197, -1),
        (199, 1),
        (201, 1),
        (203, 3),
        (205, -1),
        (207, 1),
        (209, -1),
        (211, 1),
        (213, 1),
        (215, -1),
        (217, 3),
        (219, 1),
        (221, 1),
        (223, -1),
        (225, 1),
        (227, 3),
        (229, 3),
        (231, 1),
        (233, 5),
        (235, 3),
        (237, 3),
        (239, 1),
        (241, -1),
        (243, 1),
        (245, 1),
        (247, -1),
        (249, 3),
        (251, 1),
        (253, 1),
        (255, -1),
    ];
    for &(i, v) in entries {
        lut[i] = v;
    }
    lut
}

// Index-based traversal intentionally mirrors AnalyzeSkeleton's voxel-level
// state machine. In particular, branch lengths depend on global voxel visit
// state and on the ordered points within grouped junction vertices.
#[allow(clippy::needless_range_loop)]
pub fn graph_stats(skeleton: &BinaryVolume, spacing: [f64; 3]) -> GraphStats {
    let mut degree = vec![0u8; skeleton.len()];
    let mut skeleton_voxels = 0usize;
    let mut endpoints = 0usize;
    let mut slabs = 0usize;
    let mut junction_voxels = 0usize;
    for idx in 0..skeleton.len() {
        if skeleton.data[idx] == 0 {
            continue;
        }
        skeleton_voxels += 1;
        let (x, y, z) = xyz(skeleton, idx);
        let d = foreground_neighbour_count(&skeleton.data, skeleton, x, y, z);
        degree[idx] = d as u8;
        match d {
            0 | 1 => endpoints += 1,
            2 => slabs += 1,
            _ => junction_voxels += 1,
        }
    }

    if skeleton_voxels == 0 {
        return GraphStats {
            skeleton_voxels: 0,
            endpoint_voxels: 0,
            slab_voxels: 0,
            junction_voxels: 0,
            junction_clusters: 0,
            branches: 0,
            graph_components: 0,
            cycle_rank: 0,
            mean_branch_length: f64::NAN,
            max_branch_length: f64::NAN,
            mean_branch_tortuosity: f64::NAN,
            top_bottom_spanning: false,
        };
    }

    // AnalyzeSkeleton tags endpoints in z, x, y scan order. Each endpoint is
    // its own graph vertex.
    let mut node_of = vec![usize::MAX; skeleton.len()];
    let mut node_voxels = Vec::<Vec<usize>>::new();
    for z in 0..skeleton.depth {
        for x in 0..skeleton.width {
            for y in 0..skeleton.height {
                let idx = skeleton.index(x, y, z);
                if skeleton.data[idx] == 0 || degree[idx] > 1 {
                    continue;
                }
                let node = node_voxels.len();
                node_of[idx] = node;
                node_voxels.push(vec![idx]);
            }
        }
    }
    let endpoint_node_count = node_voxels.len();

    // Junction grouping must preserve AnalyzeSkeleton's fusionNeighborJunction
    // point order. That algorithm follows a junction chain as far as possible,
    // then revisits earlier points FIFO; ordinary BFS gives the same connected
    // groups but can change later stateful branch-length decisions.
    let junction_groups = group_junction_voxels_fiji(skeleton, &degree);
    let junction_clusters = junction_groups.len();
    for voxels in junction_groups {
        let node = node_voxels.len();
        for &idx in &voxels {
            node_of[idx] = node;
        }
        node_voxels.push(voxels);
    }

    // AnalyzeSkeleton uses one global visited flag per skeleton voxel. It visits
    // endpoint branches first and junction-started branches second. This is the
    // key distinction from an undirected "trace each voxel edge once" graph
    // traversal: an already-visited junction can change the endpoint correction
    // applied to a later branch.
    let mut visited = vec![false; skeleton.len()];
    let mut branch_lengths = Vec::<f64>::new();
    let mut tortuosities = Vec::<f64>::new();
    let mut graph_edges = Vec::<(usize, usize)>::new();

    // Endpoint-started branches.
    for node in 0..endpoint_node_count {
        let start = node_voxels[node][0];
        if visited[start] {
            continue;
        }

        let visit = visit_branch_fiji(
            skeleton,
            &degree,
            &node_of,
            &node_voxels,
            start,
            spacing,
            &mut visited,
        );
        let mut length = visit.length;

        // Fiji special case for an endpoint directly adjacent to a junction
        // that has already become visited. Note that AnalyzeSkeleton's source
        // deliberately uses the endpoint again in the representative correction
        // (rather than the adjacent junction voxel); preserve that semantics.
        if length == 0.0 {
            if let Some(adjacent_junction) =
                visited_junction_neighbor_fiji(skeleton, &degree, &node_of, &visited, start, node)
            {
                let end_node = node_of[adjacent_junction];
                length += step_length(skeleton, start, adjacent_junction, spacing);
                length += step_length(skeleton, node_voxels[end_node][0], start, spacing);
                push_graph_edge(
                    skeleton,
                    &node_voxels,
                    node,
                    end_node,
                    length,
                    spacing,
                    &mut graph_edges,
                    &mut branch_lengths,
                    &mut tortuosities,
                );
            }
            continue;
        }

        let mut end_node = visit.final_node;
        if let Some(aux) = visit.aux_point {
            // If visitBranch stopped at the final slab because all of its
            // neighbours were already visited, Fiji reconnects that slab to an
            // already-visited junction. If no other junction exists, it treats
            // the branch as an inner self-loop to the initial vertex.
            if degree[aux] == 2 {
                if let Some(adjacent_junction) =
                    visited_junction_neighbor_fiji(skeleton, &degree, &node_of, &visited, aux, node)
                {
                    let final_node = node_of[adjacent_junction];
                    end_node = Some(final_node);
                    length += step_length(skeleton, adjacent_junction, aux, spacing);
                    length += step_length(
                        skeleton,
                        node_voxels[final_node][0],
                        adjacent_junction,
                        spacing,
                    );
                } else {
                    end_node = Some(node);
                    // Java sets auxPoint = aux, so the first correction is zero;
                    // only representative-to-final-slab remains.
                    length += step_length(skeleton, node_voxels[node][0], aux, spacing);
                }
            }
        }

        if let Some(end_node) = end_node {
            push_graph_edge(
                skeleton,
                &node_voxels,
                node,
                end_node,
                length,
                spacing,
                &mut graph_edges,
                &mut branch_lengths,
                &mut tortuosities,
            );
        }
    }

    // Junction-started branches. Iterate junction vertices and the points inside
    // each vertex in the exact fusion order established above.
    for node in endpoint_node_count..node_voxels.len() {
        let representative = node_voxels[node][0];
        for &junction in &node_voxels[node] {
            visited[junction] = true;

            while let Some(next) = next_unvisited_voxel_fiji(skeleton, &visited, junction) {
                if degree[next] > 2 {
                    // Adjacent junction voxels are not graph branches; Fiji only
                    // marks them visited here.
                    visited[next] = true;
                    continue;
                }

                let mut length = step_length(skeleton, junction, next, spacing);
                let visit = visit_branch_fiji(
                    skeleton,
                    &degree,
                    &node_of,
                    &node_voxels,
                    next,
                    spacing,
                    &mut visited,
                );
                length += visit.length;

                // The initial junction-to-next step is non-zero, so Fiji always
                // enters its edge-creation block here.
                let aux = visit.aux_point.unwrap_or(next);
                let mut end_node = visit.final_node.or_else(|| {
                    let n = node_of[aux];
                    (n != usize::MAX).then_some(n)
                });

                if degree[aux] == 2 {
                    if let Some(adjacent_junction) = visited_junction_neighbor_fiji(
                        skeleton, &degree, &node_of, &visited, aux, node,
                    ) {
                        end_node = Some(node_of[adjacent_junction]);
                        // AnalyzeSkeleton does NOT add an end-representative
                        // correction in this already-visited-junction path.
                        length += step_length(skeleton, adjacent_junction, aux, spacing);
                    } else {
                        // Inner self-loop: auxPoint is reset to the final slab,
                        // so the slab-to-auxPoint correction is zero.
                        end_node = Some(node);
                    }
                }

                // Always add the correction from the actual starting junction
                // voxel to the representative first point of its vertex.
                length += step_length(skeleton, representative, junction, spacing);

                if let Some(end_node) = end_node {
                    push_graph_edge(
                        skeleton,
                        &node_voxels,
                        node,
                        end_node,
                        length,
                        spacing,
                        &mut graph_edges,
                        &mut branch_lengths,
                        &mut tortuosities,
                    );
                }
            }
        }
    }

    // Pure loop components can contain only degree-2 voxels and therefore no
    // explicit endpoint/junction vertex. AnalyzeSkeleton stores each as one
    // self-cycle branch and omits the closing step back to the starting slab.
    let mut visited_voxels = vec![false; skeleton.len()];
    let mut graph_components = 0usize;
    let mut pure_cycles = 0usize;
    for start in 0..skeleton.len() {
        if skeleton.data[start] == 0 || visited_voxels[start] {
            continue;
        }
        graph_components += 1;
        let mut has_node = false;
        let mut component = Vec::new();
        let mut queue = VecDeque::from([start]);
        visited_voxels[start] = true;
        while let Some(idx) = queue.pop_front() {
            component.push(idx);
            has_node |= node_of[idx] != usize::MAX;
            let (x, y, z) = xyz(skeleton, idx);
            for n in neighbour_indices(skeleton, x, y, z) {
                if skeleton.data[n] != 0 && !visited_voxels[n] {
                    visited_voxels[n] = true;
                    queue.push_back(n);
                }
            }
        }
        if !has_node && component.iter().all(|&i| degree[i] == 2) {
            pure_cycles += 1;
            branch_lengths.push(pure_cycle_length(skeleton, &component, spacing));
        }
    }

    let graph_vertices = node_voxels.len();
    let branches = graph_edges.len() + pure_cycles;
    // Pure degree-2 cycles contribute one implicit self-loop edge and one
    // implicit vertex, which cancel in E - V + C.
    let cycle_rank =
        graph_edges.len() as isize - graph_vertices as isize + graph_components as isize;

    GraphStats {
        skeleton_voxels,
        endpoint_voxels: endpoints,
        slab_voxels: slabs,
        junction_voxels,
        junction_clusters,
        branches,
        graph_components,
        cycle_rank: cycle_rank.max(0),
        mean_branch_length: mean_or_nan(&branch_lengths),
        max_branch_length: branch_lengths.iter().copied().fold(f64::NAN, f64::max),
        mean_branch_tortuosity: mean_or_nan(&tortuosities),
        top_bottom_spanning: top_bottom_spanning(skeleton),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FijiBranchVisit {
    length: f64,
    aux_point: Option<usize>,
    final_node: Option<usize>,
}

fn group_junction_voxels_fiji(volume: &BinaryVolume, degree: &[u8]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; volume.len()];
    let mut groups = Vec::<Vec<usize>>::new();

    for z in 0..volume.depth {
        for x in 0..volume.width {
            for y in 0..volume.height {
                let start = volume.index(x, y, z);
                if volume.data[start] == 0 || degree[start] <= 2 || visited[start] {
                    continue;
                }

                let mut group = vec![start];
                visited[start] = true;
                let mut to_revisit = VecDeque::from([start]);
                let mut next = next_unvisited_junction_voxel_fiji(volume, degree, &visited, start);

                while next.is_some() || !to_revisit.is_empty() {
                    if let Some(n) = next {
                        if !visited[n] {
                            group.push(n);
                            visited[n] = true;
                            to_revisit.push_back(n);
                            next = next_unvisited_junction_voxel_fiji(volume, degree, &visited, n);
                            continue;
                        }
                    }

                    let Some(&revisit) = to_revisit.front() else {
                        break;
                    };
                    next = next_unvisited_junction_voxel_fiji(volume, degree, &visited, revisit);
                    if next.is_none() {
                        to_revisit.pop_front();
                    }
                }

                groups.push(group);
            }
        }
    }

    groups
}

#[inline]
fn next_unvisited_voxel_fiji(
    volume: &BinaryVolume,
    visited: &[bool],
    point: usize,
) -> Option<usize> {
    let (x, y, z) = xyz(volume, point);
    neighbour_indices_fiji_visit_order(volume, x, y, z)
        .into_iter()
        .find(|&n| volume.data[n] != 0 && !visited[n])
}

#[inline]
fn next_unvisited_junction_voxel_fiji(
    volume: &BinaryVolume,
    degree: &[u8],
    visited: &[bool],
    point: usize,
) -> Option<usize> {
    let (x, y, z) = xyz(volume, point);
    neighbour_indices_fiji_visit_order(volume, x, y, z)
        .into_iter()
        .find(|&n| volume.data[n] != 0 && !visited[n] && degree[n] > 2)
}

#[inline]
fn visited_junction_neighbor_fiji(
    volume: &BinaryVolume,
    degree: &[u8],
    node_of: &[usize],
    visited: &[bool],
    point: usize,
    exclude_node: usize,
) -> Option<usize> {
    let (x, y, z) = xyz(volume, point);
    neighbour_indices_fiji_visit_order(volume, x, y, z)
        .into_iter()
        .find(|&n| volume.data[n] != 0 && visited[n] && degree[n] > 2 && node_of[n] != exclude_node)
}

fn visit_branch_fiji(
    volume: &BinaryVolume,
    degree: &[u8],
    node_of: &[usize],
    node_voxels: &[Vec<usize>],
    start: usize,
    spacing: [f64; 3],
    visited: &mut [bool],
) -> FijiBranchVisit {
    let mut out = FijiBranchVisit::default();
    visited[start] = true;

    let Some(mut next) = next_unvisited_voxel_fiji(volume, visited, start) else {
        // AnalyzeSkeleton returns immediately here, before assigning auxPoint.
        return out;
    };
    let mut previous = start;

    while degree[next] == 2 {
        out.length += step_length(volume, previous, next, spacing);
        visited[next] = true;
        previous = next;
        let Some(n) = next_unvisited_voxel_fiji(volume, visited, previous) else {
            out.aux_point = Some(previous);
            return out;
        };
        next = n;
    }

    out.length += step_length(volume, previous, next, spacing);
    visited[next] = true;
    let node = node_of[next];
    if node != usize::MAX {
        out.final_node = Some(node);
        if degree[next] > 2 {
            // Reaching an unvisited junction gets a representative correction
            // inside visitBranch itself.
            out.length += step_length(volume, node_voxels[node][0], next, spacing);
        }
    }
    out.aux_point = Some(next);
    out
}

#[allow(clippy::too_many_arguments)]
fn push_graph_edge(
    volume: &BinaryVolume,
    node_voxels: &[Vec<usize>],
    start_node: usize,
    end_node: usize,
    length: f64,
    spacing: [f64; 3],
    graph_edges: &mut Vec<(usize, usize)>,
    branch_lengths: &mut Vec<f64>,
    tortuosities: &mut Vec<f64>,
) {
    graph_edges.push((start_node, end_node));
    branch_lengths.push(length);
    let endpoint_distance = step_length(
        volume,
        node_voxels[start_node][0],
        node_voxels[end_node][0],
        spacing,
    );
    if endpoint_distance > 0.0 {
        tortuosities.push(length / endpoint_distance);
    }
}

fn pure_cycle_length(volume: &BinaryVolume, component: &[usize], spacing: [f64; 3]) -> f64 {
    // AnalyzeSkeleton marks the circular tree's starting slab visited before
    // branch traversal, so the final closing step back to that slab is omitted.
    let set = component.iter().copied().collect::<HashSet<_>>();
    let Some(&start) = component.iter().min_by_key(|&&idx| {
        let (x, y, z) = xyz(volume, idx);
        (z, x, y)
    }) else {
        return 0.0;
    };

    let mut visited_voxels = HashSet::<usize>::new();
    visited_voxels.insert(start);
    let mut current = start;
    let mut length = 0.0;

    loop {
        let (x, y, z) = xyz(volume, current);
        let Some(next) = neighbour_indices_fiji_visit_order(volume, x, y, z)
            .into_iter()
            .find(|n| set.contains(n) && !visited_voxels.contains(n))
        else {
            break;
        };
        length += step_length(volume, current, next, spacing);
        visited_voxels.insert(next);
        current = next;
    }
    length
}

fn top_bottom_spanning(volume: &BinaryVolume) -> bool {
    let mut min_z = usize::MAX;
    let mut max_z = 0usize;
    for idx in 0..volume.len() {
        if volume.data[idx] != 0 {
            let (_, _, z) = xyz(volume, idx);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
    }
    if min_z == usize::MAX || min_z == max_z {
        return false;
    }
    let mut queue = VecDeque::new();
    let mut seen = vec![false; volume.len()];
    for y in 0..volume.height {
        for x in 0..volume.width {
            let idx = volume.index(x, y, min_z);
            if volume.data[idx] != 0 {
                seen[idx] = true;
                queue.push_back(idx);
            }
        }
    }
    while let Some(idx) = queue.pop_front() {
        let (x, y, z) = xyz(volume, idx);
        if z == max_z {
            return true;
        }
        for n in neighbour_indices(volume, x, y, z) {
            if volume.data[n] != 0 && !seen[n] {
                seen[n] = true;
                queue.push_back(n);
            }
        }
    }
    false
}

fn component_count_local(mask: &[bool; 27], foreground: bool) -> usize {
    let offsets: &[(isize, isize, isize)] = if foreground { &N26 } else { &N6 };
    let mut seen = [false; 27];
    let mut count = 0usize;
    for start in 0..27 {
        if !mask[start] || seen[start] {
            continue;
        }
        count += 1;
        if count > 1 {
            return count;
        }
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(i) = queue.pop_front() {
            let lx = (i % 3) as isize;
            let ly = ((i / 3) % 3) as isize;
            let lz = (i / 9) as isize;
            for &(dx, dy, dz) in offsets {
                let nx = lx + dx;
                let ny = ly + dy;
                let nz = lz + dz;
                if nx < 0 || ny < 0 || nz < 0 || nx >= 3 || ny >= 3 || nz >= 3 {
                    continue;
                }
                let ni = (nz * 9 + ny * 3 + nx) as usize;
                if mask[ni] && !seen[ni] {
                    seen[ni] = true;
                    queue.push_back(ni);
                }
            }
        }
    }
    count
}

#[inline]
fn foreground_neighbour_count(
    data: &[u8],
    volume: &BinaryVolume,
    x: usize,
    y: usize,
    z: usize,
) -> usize {
    N26.iter()
        .filter(|&&(dx, dy, dz)| {
            get_data(
                data,
                volume,
                x as isize + dx,
                y as isize + dy,
                z as isize + dz,
            ) != 0
        })
        .count()
}

fn neighbour_indices(volume: &BinaryVolume, x: usize, y: usize, z: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(26);
    for &(dx, dy, dz) in &N26 {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        let nz = z as isize + dz;
        if nx >= 0
            && ny >= 0
            && nz >= 0
            && nx < volume.width as isize
            && ny < volume.height as isize
            && nz < volume.depth as isize
        {
            out.push(volume.index(nx as usize, ny as usize, nz as usize));
        }
    }
    out
}

// AnalyzeSkeleton's getNextUnvisitedVoxel loops x, then y, then z, but the
// break exits only the innermost z loop. Its effective visit priority is
// decreasing dx, decreasing dy, increasing dz.
fn neighbour_indices_fiji_visit_order(
    volume: &BinaryVolume,
    x: usize,
    y: usize,
    z: usize,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(26);
    for dx in (-1isize..=1).rev() {
        for dy in (-1isize..=1).rev() {
            for dz in -1isize..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                let nz = z as isize + dz;
                if nx >= 0
                    && ny >= 0
                    && nz >= 0
                    && nx < volume.width as isize
                    && ny < volume.height as isize
                    && nz < volume.depth as isize
                {
                    out.push(volume.index(nx as usize, ny as usize, nz as usize));
                }
            }
        }
    }
    out
}

#[inline]
fn get_data(data: &[u8], volume: &BinaryVolume, x: isize, y: isize, z: isize) -> u8 {
    if x < 0
        || y < 0
        || z < 0
        || x >= volume.width as isize
        || y >= volume.height as isize
        || z >= volume.depth as isize
    {
        0
    } else {
        data[volume.index(x as usize, y as usize, z as usize)]
    }
}

#[inline]
fn xyz(volume: &BinaryVolume, idx: usize) -> (usize, usize, usize) {
    let slice = volume.width * volume.height;
    let z = idx / slice;
    let rem = idx % slice;
    let y = rem / volume.width;
    let x = rem % volume.width;
    (x, y, z)
}

#[inline]
fn voxel_point(volume: &BinaryVolume, idx: usize, spacing: [f64; 3]) -> [f64; 3] {
    let (x, y, z) = xyz(volume, idx);
    [
        x as f64 * spacing[0],
        y as f64 * spacing[1],
        z as f64 * spacing[2],
    ]
}

#[inline]
fn step_length(volume: &BinaryVolume, a: usize, b: usize, spacing: [f64; 3]) -> f64 {
    distance(
        voxel_point(volume, a, spacing),
        voxel_point(volume, b, spacing),
    )
}

#[inline]
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn mean_or_nan(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_four_voxel_diamond_has_cycle_rank_one() {
        // Four degree-2 voxels forming a 26-connected diamond in one z-plane:
        //   . X .
        //   X . X
        //   . X .
        // Every voxel has exactly two neighbors, so this is a pure cycle with
        // no explicit endpoint or junction node.
        let mut data = vec![0u8; 7 * 7 * 7];
        let z = 3usize;
        for (x, y) in [(3usize, 2usize), (4, 3), (3, 4), (2, 3)] {
            data[(z * 7 + y) * 7 + x] = 1;
        }
        let volume = BinaryVolume::new(data, 7, 7, 7).unwrap();
        let stats = graph_stats(&volume, [1.0; 3]);
        assert_eq!(stats.skeleton_voxels, 4);
        assert_eq!(stats.endpoint_voxels, 0);
        assert_eq!(stats.slab_voxels, 4);
        assert_eq!(stats.junction_voxels, 0);
        assert_eq!(stats.graph_components, 1);
        assert_eq!(stats.branches, 1);
        assert_eq!(stats.cycle_rank, 1);
        // AnalyzeSkeleton omits the final closing step to the already-visited starting slab.
        assert!((stats.mean_branch_length - 3.0 * 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn fiji_junction_representative_lengths_match_stage4a_y() {
        let n = 21usize;
        let c = 10usize;
        let mut data = vec![0u8; n * n * n];
        let mut set = |x: usize, y: usize, z: usize| {
            data[(z * n + y) * n + x] = 1;
        };
        for x in 5..=15 {
            set(x, c, c);
        }
        for y in c..=15 {
            set(c, y, c);
        }
        let volume = BinaryVolume::new(data, n, n, n).unwrap();
        let stats = graph_stats(&volume, [1.0; 3]);
        let expected_mean = (4.0 + 6.0 + 4.0 + 2.0_f64.sqrt()) / 3.0;
        assert_eq!(stats.branches, 3);
        assert_eq!(stats.cycle_rank, 0);
        assert!((stats.mean_branch_length - expected_mean).abs() < 1e-12);
        assert!((stats.max_branch_length - 6.0).abs() < 1e-12);
    }

    #[test]
    fn fiji_attached_self_loop_is_one_cycle_edge() {
        let n = 21usize;
        let c = 10usize;
        let mut data = vec![0u8; n * n * n];
        let mut set = |x: usize, y: usize, z: usize| {
            data[(z * n + y) * n + x] = 1;
        };
        set(c, c - 1, c);
        set(c + 1, c, c);
        set(c, c + 1, c);
        set(c - 1, c, c);
        for x in c + 1..=16 {
            set(x, c, c);
        }
        let volume = BinaryVolume::new(data, n, n, n).unwrap();
        let stats = graph_stats(&volume, [1.0; 3]);
        let loop_length = 3.0 * 2.0_f64.sqrt();
        let tail_length = 5.0;
        assert_eq!(stats.branches, 2);
        assert_eq!(stats.cycle_rank, 1);
        assert!((stats.mean_branch_length - (loop_length + tail_length) / 2.0).abs() < 1e-12);
        assert!((stats.max_branch_length - tail_length).abs() < 1e-12);
    }

    #[test]
    fn lee_euler_lut_reference_sentinels() {
        let lut = lee_euler_lut();
        assert_eq!(lut[1], 1);
        assert_eq!(lut[105], 5);
        assert_eq!(lut[129], -7);
        assert_eq!(lut[233], 5);
        assert_eq!(lut[255], -1);
        assert_eq!(lut[2], 0);
    }

    #[test]
    fn lee_simple_point_uses_foreground_connectivity() {
        let mut n = [0u8; 27];
        n[13] = 1;
        n[12] = 1;
        n[4] = 1;
        assert!(lee_is_simple_point(&n));
    }

    #[test]
    fn straight_three_voxel_line_is_preserved_as_graph() {
        let mut data = vec![0u8; 5 * 5 * 5];
        for z in 1..=3 {
            data[(z * 5 + 2) * 5 + 2] = 1;
        }
        let volume = BinaryVolume::new(data, 5, 5, 5).unwrap();
        let stats = graph_stats(&volume, [1.0; 3]);
        assert_eq!(stats.endpoint_voxels, 2);
        assert_eq!(stats.slab_voxels, 1);
        assert_eq!(stats.junction_voxels, 0);
        assert_eq!(stats.branches, 1);
        assert!((stats.mean_branch_length - 2.0).abs() < 1e-12);
        assert!((stats.mean_branch_tortuosity - 1.0).abs() < 1e-12);
    }
}

/// One branch of the exact AnalyzeSkeleton-compatible compressed graph.
#[derive(Clone, Copy, Debug)]
pub struct CompressedGraphEdge {
    pub u: usize,
    pub v: usize,
    pub length_um: f64,
    /// True only for the synthetic self-loop representing an all-degree-2 cycle.
    #[allow(dead_code)]
    pub pure_cycle: bool,
}

/// Compressed graph representation used by the mechanical-graph descriptors.
#[derive(Clone, Debug)]
pub struct FijiCompressedGraph {
    pub node_voxels: Vec<Vec<usize>>,
    pub edges: Vec<CompressedGraphEdge>,
    pub skeleton_voxels: Vec<usize>,
}

/// Materialize the same stateful Fiji graph traversal used by `graph_stats`.
///
/// This function intentionally duplicates the traversal schedule rather than
/// replacing the validated skeleton implementation. This preserves the aggregate
/// path while exposing branch identities and lengths required by the
/// mechanical-graph descriptors.
pub fn compressed_graph_fiji(skeleton: &BinaryVolume, spacing: [f64; 3]) -> FijiCompressedGraph {
    let mut degree = vec![0u8; skeleton.len()];
    let mut occupied = Vec::new();
    for (idx, degree_value) in degree.iter_mut().enumerate() {
        if skeleton.data[idx] == 0 {
            continue;
        }
        occupied.push(idx);
        let (x, y, z) = xyz(skeleton, idx);
        *degree_value = foreground_neighbour_count(&skeleton.data, skeleton, x, y, z) as u8;
    }
    if occupied.is_empty() {
        return FijiCompressedGraph { node_voxels: Vec::new(), edges: Vec::new(), skeleton_voxels: Vec::new() };
    }

    let mut node_of = vec![usize::MAX; skeleton.len()];
    let mut node_voxels = Vec::<Vec<usize>>::new();
    for z in 0..skeleton.depth {
        for x in 0..skeleton.width {
            for y in 0..skeleton.height {
                let idx = skeleton.index(x, y, z);
                if skeleton.data[idx] == 0 || degree[idx] > 1 {
                    continue;
                }
                let node = node_voxels.len();
                node_of[idx] = node;
                node_voxels.push(vec![idx]);
            }
        }
    }
    let endpoint_node_count = node_voxels.len();
    for voxels in group_junction_voxels_fiji(skeleton, &degree) {
        let node = node_voxels.len();
        for &idx in &voxels {
            node_of[idx] = node;
        }
        node_voxels.push(voxels);
    }

    let mut visited = vec![false; skeleton.len()];
    let mut edges = Vec::<CompressedGraphEdge>::new();
    for node in 0..endpoint_node_count {
        let start = node_voxels[node][0];
        if visited[start] {
            continue;
        }
        let visit = visit_branch_fiji(skeleton, &degree, &node_of, &node_voxels, start, spacing, &mut visited);
        let mut length = visit.length;
        if length == 0.0 {
            if let Some(adj) = visited_junction_neighbor_fiji(skeleton, &degree, &node_of, &visited, start, node) {
                let end = node_of[adj];
                length += step_length(skeleton, start, adj, spacing);
                length += step_length(skeleton, node_voxels[end][0], start, spacing);
                edges.push(CompressedGraphEdge { u: node, v: end, length_um: length, pure_cycle: false });
            }
            continue;
        }
        let mut end = visit.final_node;
        if let Some(aux) = visit.aux_point {
            if degree[aux] == 2 {
                if let Some(adj) = visited_junction_neighbor_fiji(skeleton, &degree, &node_of, &visited, aux, node) {
                    let final_node = node_of[adj];
                    end = Some(final_node);
                    length += step_length(skeleton, adj, aux, spacing);
                    length += step_length(skeleton, node_voxels[final_node][0], adj, spacing);
                } else {
                    end = Some(node);
                    length += step_length(skeleton, node_voxels[node][0], aux, spacing);
                }
            }
        }
        if let Some(v) = end {
            edges.push(CompressedGraphEdge { u: node, v, length_um: length, pure_cycle: false });
        }
    }

    for node in endpoint_node_count..node_voxels.len() {
        let representative = node_voxels[node][0];
        let junctions = node_voxels[node].clone();
        for junction in junctions {
            visited[junction] = true;
            while let Some(next) = next_unvisited_voxel_fiji(skeleton, &visited, junction) {
                if degree[next] > 2 {
                    visited[next] = true;
                    continue;
                }
                let mut length = step_length(skeleton, junction, next, spacing);
                let visit = visit_branch_fiji(skeleton, &degree, &node_of, &node_voxels, next, spacing, &mut visited);
                length += visit.length;
                let aux = visit.aux_point.unwrap_or(next);
                let mut end = visit.final_node.or_else(|| {
                    let n = node_of[aux];
                    (n != usize::MAX).then_some(n)
                });
                if degree[aux] == 2 {
                    if let Some(adj) = visited_junction_neighbor_fiji(skeleton, &degree, &node_of, &visited, aux, node) {
                        end = Some(node_of[adj]);
                        length += step_length(skeleton, adj, aux, spacing);
                    } else {
                        end = Some(node);
                    }
                }
                length += step_length(skeleton, representative, junction, spacing);
                if let Some(v) = end {
                    edges.push(CompressedGraphEdge { u: node, v, length_um: length, pure_cycle: false });
                }
            }
        }
    }

    // Each leftover all-degree-2 component is represented as an implicit
    // node plus one self-loop branch.
    let mut leftover: HashSet<usize> = occupied.iter().copied().filter(|&i| !visited[i]).collect();
    while let Some(&start) = leftover.iter().next() {
        leftover.remove(&start);
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some(p) = queue.pop_front() {
            component.push(p);
            let (x,y,z) = xyz(skeleton,p);
            for n in neighbour_indices_fiji_visit_order(skeleton,x,y,z) {
                if skeleton.data[n] != 0 && leftover.remove(&n) {
                    queue.push_back(n);
                }
            }
        }
        debug_assert!(component.iter().all(|&i| degree[i] == 2));
        let node = node_voxels.len();
        let length = pure_cycle_length(skeleton, &component, spacing);
        node_voxels.push(component);
        edges.push(CompressedGraphEdge { u: node, v: node, length_um: length, pure_cycle: true });
    }

    FijiCompressedGraph { node_voxels, edges, skeleton_voxels: occupied }
}

/// Public coordinate helper for downstream graph modules.
pub fn voxel_xyz(volume: &BinaryVolume, idx: usize) -> (usize, usize, usize) {
    xyz(volume, idx)
}
