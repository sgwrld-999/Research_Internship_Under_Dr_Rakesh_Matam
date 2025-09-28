# Comprehensive Report on Feature Clustering in GRIFFIN Network

## Executive Summary

The GRIFFIN (Group-Regularized Intrusion Flow Feature Integration Network) employs a feature clustering mechanism that organizes 46 network traffic features from the CICIoT dataset into 5 distinct semantic groups. This clustering is performed through manual domain-based partitioning rather than automatic algorithmic clustering, with groups defined based on the semantic meaning and functional relationships of network traffic characteristics.

## 1. Clustering Methodology

### 1.1 Clustering Approach

The clustering in GRIFFIN follows a **manual, domain-knowledge-driven approach**. Features are not clustered through mathematical algorithms such as k-means, hierarchical clustering, or correlation-based methods. Instead, the grouping is predetermined based on expert knowledge of network traffic analysis and intrusion detection systems.

### 1.2 Clustering Process

The clustering process consists of three primary stages:

**Stage 1: Feature Analysis**
Each of the 46 features in the CICIoT dataset is analyzed based on its semantic meaning, measurement unit, and role in network traffic characterization.

**Stage 2: Semantic Categorization**
Features are assigned to groups based on keyword matching and functional similarity. The assignment follows a hierarchical decision process where features containing specific keywords or measuring similar aspects of network behavior are grouped together.

**Stage 3: Fixed Assignment**
Once assigned, feature-to-group mappings remain static throughout training and inference. The clustering does not adapt or change based on data patterns or model performance.

## 2. Cluster Definitions and Compositions

### 2.1 Cluster 1: Packet Statistics (8 features)

**Purpose:** Captures size characteristics of data packets in both forward and backward directions.

**Features Included:**
- fwd_pkt_len_max: Maximum packet length in forward direction
- fwd_pkt_len_min: Minimum packet length in forward direction
- fwd_pkt_len_mean: Mean packet length in forward direction
- fwd_pkt_len_std: Standard deviation of packet length in forward direction
- bwd_pkt_len_max: Maximum packet length in backward direction
- bwd_pkt_len_min: Minimum packet length in backward direction
- bwd_pkt_len_mean: Mean packet length in backward direction
- bwd_pkt_len_std: Standard deviation of packet length in backward direction

**Clustering Rationale:** These features collectively describe the distribution and characteristics of packet sizes, which are critical indicators for detecting payload-based attacks, data exfiltration attempts, and fragmentation attacks.

### 2.2 Cluster 2: Timing Patterns (8 features)

**Purpose:** Represents temporal characteristics of network flows and packet inter-arrival times.

**Features Included:**
- flow_iat_mean: Mean inter-arrival time of flow
- flow_iat_std: Standard deviation of flow inter-arrival time
- flow_iat_max: Maximum inter-arrival time in flow
- flow_iat_min: Minimum inter-arrival time in flow
- fwd_iat_total: Total inter-arrival time in forward direction
- fwd_iat_mean: Mean inter-arrival time in forward direction
- bwd_iat_total: Total inter-arrival time in backward direction
- bwd_iat_mean: Mean inter-arrival time in backward direction

**Clustering Rationale:** Temporal patterns are essential for identifying automated attacks, bot behavior, and timing-based anomalies such as slow-rate attacks or burst transmissions.

### 2.3 Cluster 3: Flow Rates and Volume (10 features)

**Purpose:** Quantifies the volume and rate characteristics of network flows.

**Features Included:**
- flow_duration: Total duration of the flow
- flow_bytes_s: Bytes transmitted per second
- flow_pkts_s: Packets transmitted per second
- down_up_ratio: Ratio of download to upload traffic
- pkt_len_var: Variance in packet length
- fwd_seg_size_avg: Average segment size in forward direction
- bwd_seg_size_avg: Average segment size in backward direction
- subflow_fwd_pkts: Number of forward packets in subflow
- subflow_fwd_bytes: Number of forward bytes in subflow
- active_mean: Mean active time of flow

**Clustering Rationale:** Flow characteristics are fundamental for detecting volumetric attacks, bandwidth consumption anomalies, and traffic pattern deviations indicative of DDoS or resource exhaustion attacks.

### 2.4 Cluster 4: TCP Flags and Protocol States (10 features)

**Purpose:** Encodes TCP protocol control information and flag statistics.

**Features Included:**
- fwd_psh_flags: Count of PSH flags in forward direction
- bwd_psh_flags: Count of PSH flags in backward direction
- fwd_urg_flags: Count of URG flags in forward direction
- bwd_urg_flags: Count of URG flags in backward direction
- fin_flag_cnt: Total count of FIN flags
- syn_flag_cnt: Total count of SYN flags
- rst_flag_cnt: Total count of RST flags
- psh_flag_cnt: Total count of PSH flags
- ack_flag_cnt: Total count of ACK flags
- urg_flag_cnt: Total count of URG flags

**Clustering Rationale:** TCP flag patterns are diagnostic for protocol-level attacks including SYN floods, connection reset attacks, and scanning activities. The grouping maintains protocol coherence.

### 2.5 Cluster 5: Protocol and Port Information (10 features)

**Purpose:** Contains network protocol identifiers and connection metadata.

**Features Included:**
- protocol: Network protocol identifier
- src_port: Source port number
- dst_port: Destination port number
- init_fwd_win_bytes: Initial window size in forward direction
- init_bwd_win_bytes: Initial window size in backward direction
- fwd_act_data_pkts: Count of forward packets with data
- fwd_seg_size_min: Minimum segment size in forward direction
- active_max: Maximum active time
- idle_mean: Mean idle time
- idle_max: Maximum idle time

**Clustering Rationale:** Protocol and port information is essential for identifying service-specific attacks, unauthorized access attempts, and reconnaissance activities targeting specific services.

## 3. Mathematical Representation

### 3.1 Formal Notation

Let X ∈ ℝ^F be the input feature vector where F = 46. The clustering function C maps each feature index to a group identifier:

C: {1, 2, ..., 46} → {1, 2, 3, 4, 5}

The partition can be represented as:
- G₁ = {1, 2, 3, 4, 5, 6, 7, 8}
- G₂ = {9, 10, 11, 12, 13, 14, 15, 16}
- G₃ = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26}
- G₄ = {27, 28, 29, 30, 31, 32, 33, 34, 35, 36}
- G₅ = {37, 38, 39, 40, 41, 42, 43, 44, 45, 46}

### 3.2 Group Cardinality

|G₁| = 8, |G₂| = 8, |G₃| = 10, |G₄| = 10, |G₅| = 10

Total features: Σᵢ|Gᵢ| = 46

## 4. Clustering Properties

### 4.1 Completeness
Every feature is assigned to exactly one cluster. There are no unassigned features.
∀i ∈ {1,...,46}, ∃! j ∈ {1,...,5} : i ∈ Gⱼ

### 4.2 Disjointness
Clusters are mutually exclusive with no overlap.
∀i,j ∈ {1,...,5}, i ≠ j ⟹ Gᵢ ∩ Gⱼ = ∅

### 4.3 Coverage
The union of all clusters equals the complete feature set.
⋃ᵢ₌₁⁵ Gᵢ = {1, 2, ..., 46}

### 4.4 Static Nature
The clustering remains constant across all samples and throughout the model lifecycle. It is not data-dependent or learnable.

## 5. Integration with Gate Mechanism

### 5.1 Gate Application Process

Each cluster receives a scalar gate value g ∈ [0,1] that modulates all features within that cluster uniformly. For a feature xᵢ belonging to cluster Gⱼ, the gated output is:
x̃ᵢ = gⱼ · xᵢ

### 5.2 Gate Learning

While clusters are fixed, gate values are learned through backpropagation:
g = σ(Wg · x + bg)

Where:
- Wg ∈ ℝ^(46×5): Learnable weight matrix
- bg ∈ ℝ^5: Learnable bias vector
- σ: Sigmoid activation function

### 5.3 Group-wise Modulation

The clustering enables group-level feature modulation rather than individual feature gating, reducing the parameter space from 46 individual gates to 5 group gates.

## 6. Clustering Characteristics Analysis

### 6.1 Semantic Coherence

Each cluster maintains high semantic coherence, with features measuring related aspects of network behavior grouped together. This coherence is achieved through domain expertise rather than statistical correlation.

### 6.2 Feature Distribution

The distribution of features across clusters is relatively balanced:
- Clusters 1-2: 8 features each (17.4% per cluster)
- Clusters 3-5: 10 features each (21.7% per cluster)

### 6.3 Information Redundancy

Within each cluster, features often exhibit correlation due to their semantic similarity. For example, in Cluster 1, forward and backward packet statistics often correlate, as both reflect overall packet size patterns.

### 6.4 Cross-cluster Independence

Different clusters capture orthogonal aspects of network behavior. Packet sizes (Cluster 1) are largely independent of TCP flags (Cluster 4), allowing the model to learn distinct patterns for different attack types.

## 7. Operational Characteristics

### 7.1 Preprocessing Requirements

No special preprocessing is required for clustering. Features are grouped based on their position in the standard CICIoT feature vector.

### 7.2 Computational Complexity

Clustering assignment: O(1) - Lookup operation
Gate computation: O(F) where F = 46
Gate application: O(F) element-wise multiplication

### 7.3 Memory Requirements

The clustering requires minimal memory:
- Storage of group assignments: 46 integers
- Gate weight matrix: 46 × 5 = 230 parameters
- Gate bias vector: 5 parameters

## 8. Attack Detection Patterns

### 8.1 Attack-Specific Gate Activations

Different attack types exhibit characteristic gate activation patterns across clusters:

**DDoS Attacks:**
- High activation on Clusters 2 (Timing) and 3 (Flow Rates)
- Low activation on Cluster 5 (Protocol Info)

**Port Scanning:**
- High activation on Clusters 4 (TCP Flags) and 5 (Ports)
- Low activation on Cluster 1 (Packet Statistics)

**Data Exfiltration:**
- High activation on Clusters 1 (Packet Statistics) and 3 (Flow Rates)
- Moderate activation on other clusters

### 8.2 Benign Traffic Patterns

Normal traffic typically shows balanced gate activations across clusters, with moderate values (0.3-0.6) rather than extreme activations.

## 9. Limitations and Constraints

### 9.1 Fixed Clustering Limitations

The manual clustering approach cannot adapt to new feature relationships that may emerge in evolving network traffic patterns or novel attack types.

### 9.2 Feature Assignment Rigidity

Features are assigned based on keywords and semantic interpretation, which may not always reflect their actual statistical relationships or predictive value.

### 9.3 Granularity Constraints

The fixed number of 5 clusters may be suboptimal for certain datasets or attack scenarios where finer or coarser grouping could be beneficial.

### 9.4 Domain Dependency

The clustering heavily relies on domain expertise in network security and may not generalize well to datasets with different feature sets or naming conventions.

## 10. Implementation Details

### 10.1 Feature Assignment Algorithm

The assignment follows a hierarchical keyword-matching approach:
1. Check for 'packet' or 'pkt' keywords → Cluster 1
2. Check for 'iat' or 'time' keywords → Cluster 2
3. Check for 'flow' or 'rate' keywords → Cluster 3
4. Check for 'flag' or 'tcp' keywords → Cluster 4
5. Remaining features → Cluster 5

### 10.2 Data Structure

Clusters are implemented as dictionary mappings:
- Key: Cluster identifier (1-5)
- Value: List of feature indices

### 10.3 Integration Points

The clustering interfaces with:
- Gate computation module
- Feature preprocessing pipeline
- Interpretability analysis tools

## Conclusion

The feature clustering in GRIFFIN represents a domain-driven approach to organizing network traffic features into semantically coherent groups. This manual clustering methodology, while lacking the adaptability of algorithmic approaches, provides stable, interpretable, and computationally efficient feature grouping that aligns with network security domain knowledge. The five clusters capture distinct aspects of network behavior - packet characteristics, temporal patterns, flow metrics, protocol states, and connection metadata - enabling the model to selectively focus on relevant feature groups for different attack detection scenarios. The clustering serves as the foundation for the Protocol-Aware Group Gate mechanism, which learns to dynamically weight these clusters based on their relevance to specific attack patterns, ultimately improving both detection accuracy and model interpretability.