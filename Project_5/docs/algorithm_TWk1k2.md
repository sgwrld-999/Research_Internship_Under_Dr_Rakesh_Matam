# Updated Algorithm: Temporal-Weighted K1K2NN Prediction (with mathematical refinements, compute optimizations, and robustness reparameterizations)

```markdown
Algorithm: Temporal-Weighted K1K2NN Prediction (Updated)

Function Predict(
    S,                // Training set {(x_i, Y_i, t_i)} for i=1...m
    x_q, t_q,         // Query feature vector and timestamp
    k1, k2,           // #neighbors in K1 and K2
    λ,                // Exponential decay rate (used if half-life h not provided)
    Options           // Optional, see below
) -> Y_q

Inputs:
  S = {(x_i ∈ ℝ^d, Y_i ⊆ 𝓛, t_i)}_{i=1}^m
  x_q ∈ ℝ^d, t_q ∈ ℝ
  k1, k2 ∈ ℕ
  λ ≥ 0

Options (all optional; defaults keep original behavior):
  // Mathematical refinements
  - h                // half-life; if provided, λ_eff := ln(2)/h; else λ_eff := λ
  - time_scale s     // robust time scale (e.g., MAD or IQR of Δt); default s := 1
  - α ∈ [0,1]        // Laplace smoothing for Jaccard; default α := 0 (plain Jaccard)
  - β ∈ [0,2]        // distance exponent for weighted voting; default β := 0 (majority)
  - τ ≥ 0            // soft-vote threshold; if unset, use majority vote
  - θ ∈ [0,1]        // min label-similarity for K2 (pre-filter); default: none
  - ε > 0            // small constant for numerical stability in logs/divisions

  // Preprocessing / feature-space refinement (no new stage; one-time offline)
  - Standardize: z-score with (μ, σ) estimated on S
  - Whitening: diagonal Mahalanobis via σ (already implied by z-score)
  - PCA matrix P (retain 95–99% variance), applied once offline; default: identity

  // Compute optimizations (no logic change)
  - CacheSquaredNorms: store ||z_i'||^2
  - PrecomputeTopK2: for each label ℓ, store top-K2 similar labels under smoothed Jaccard
  - UsePartialSelection: nth_element/heap instead of full sort for top-k1

Output:
  Y_q ⊆ 𝓛

-----------------------------------------
1. // ===== One-time OFFLINE PRECOMPUTATION =====
2. // Feature standardization / whitening (if not already z-scored)
3. Estimate μ, σ on {x_i}; define z_i := (x_i - μ) / σ  (elementwise);   // W = diag(1/σ^2)
4. If PCA P is provided: z_i' := Pᵀ z_i else z_i' := z_i
5. Cache ||z_i'||^2 for all i  (if CacheSquaredNorms)

6. // Label co-occurrence and similarity with Laplace smoothing
7. C ← ComputeLabelCooccurrenceCounts(S)   // counts C_{ij}, label marginals n_i, n_j
8. For all label pairs (i,j):
9.     J^(α)(i,j) := (C_{ij} + α) / ( (n_i + n_j - C_{ij}) + α )   // α=0 gives plain Jaccard
10. Sim_L ← { J^(α)(i,j) }  // label–label similarity matrix

11. // (Optional) Precompute top-K2 labels per label for fast K2 expansion
12. If PrecomputeTopK2:
13.     For each label ℓ:
14.         SimList[ℓ] := labels sorted by J^(α)(ℓ, ·) in descending order

15. // Robust time scaling and temporal decay
16. If time_scale s not provided: set s := 1   // or compute robust scale (e.g., MAD of Δt on S)
17. If half-life h provided: λ_eff := ln(2)/h else λ_eff := λ

-----------------------------------------
18. // ===== PREDICTION PHASE =====
19. // Transform query to the same feature space
20. z_q := (x_q - μ) / σ
21. If PCA P is provided: z_q' := Pᵀ z_q else z_q' := z_q
22. q2 := ||z_q'||^2

23. // Compute temporally weighted ranking score WITHOUT square root
24. DistScores ← empty list
25. For i = 1..m:
26.     // Squared Euclidean via inner products (BLAS-friendly)
27.     d2_i := q2 + ||z_i'||^2 - 2 * (z_q'ᵀ z_i')               // ≥ 0
28.     Δt_i := max(0, t_q - t_i)                                // enforce non-negativity
29.     // Ranking-equivalent log-sum score (monotone in d2 and Δt)
30.     S_i := 0.5 * log(d2_i + ε) + λ_eff * (Δt_i / s)
31.     Append (S_i, i, d2_i, Δt_i) to DistScores
32. End For

33. // Select top-k1 neighbors by PARTIAL SELECTION (no full sort)
34. K1_Idx ← SelectTopKByScore(DistScores, k1)   // e.g., nth_element/heap on S_i ascending

35. // ===== K1 label aggregation (original majority or optional soft weighted vote) =====
36. If τ is unset (default):   // ORIGINAL MAJORITY
37.     Y_q_initial := MajorityVote({Y_i | i ∈ K1_Idx})          // include label if count > k1/2
38. Else:                    // OPTIONAL SOFT VOTE using distance & time weights
39.     For each label ℓ ∈ 𝓛:
40.         v̂(ℓ) := Σ_{i ∈ K1_Idx} [ exp(-λ_eff * (Δt_i / s)) / (d2_i + ε)^(β/2) ] * 𝟙[ℓ ∈ Y_i]
41.     Y_q_initial := { ℓ : v̂(ℓ) ≥ τ }
42.     // Optional deterministic tie-break (when needed): prefer labels with larger
43.     // Σ_{ℓ' ∈ Y_q_initial \ {ℓ}} J^(α)(ℓ, ℓ').

44. // ===== K2 label expansion (same step, smoothed similarity, optional threshold) =====
45. Y_q_expansion := ∅
46. For each label ℓ_a ∈ Y_q_initial:
47.     if PrecomputeTopK2:
48.         Candidates := SimList[ℓ_a]            // already sorted by J^(α)
49.     else:
50.         Candidates := all labels sorted by J^(α)(ℓ_a, ·) in descending order
51.     if θ is provided:
52.         Candidates := { ℓ : ℓ ∈ Candidates and J^(α)(ℓ_a, ℓ) ≥ θ }
53.     K2_Labels := first k2 elements of Candidates
54.     Y_q_expansion := Y_q_expansion ∪ K2_Labels
55. End For

56. // Final prediction
57. Y_q := Y_q_initial ∪ Y_q_expansion
58. return Y_q
```

### Notes (concise, implementation-oriented)

* **Monotone temporal penalization:** The effective distance is ( |z_q' - z_i'|*2 \cdot \exp(\lambda*{\text{eff}} \Delta t_i / s) ). We rank by the equivalent **log-score** ( S_i = \tfrac12 \log(d2_i + \varepsilon) + \lambda_{\text{eff}} \Delta t_i / s ), which avoids square roots and improves numerical stability.
* **Whitening / standardization:** Using z-scores (or diagonal Mahalanobis) is a reparameterization of the same K1 step; it reduces scale-induced bias in IoT features (counts/bytes/flags).
* **Smoothed Jaccard (K2):** ( J^{(\alpha)}(i,j) = \frac{C_{ij} + \alpha}{(n_i + n_j - C_{ij}) + \alpha} ) stabilizes similarity for rare labels without changing the K2 mechanism. Set ( \alpha=0 ) to recover the original.
* **Half-life parameterization:** If provided, ( \lambda_{\text{eff}} = \ln 2 / h ); this keeps the same temporal mechanism but makes λ interpretable operationally.
* **Partial selection:** Use `SelectTopKByScore` (nth_element/heap) to get k₁ neighbors in linear-time average, rather than sorting all m distances.
* **Weighted voting (optional):** Lines 39–41 keep aggregation inside the existing K1 step. If you prefer the original behavior, skip τ (lines 36–41) and the algorithm reverts to majority voting.
* **PCA (optional):** A one-time linear projection for denoising/acceleration; it does not add a new stage—K1 still runs in the (transformed) feature space.
* **Caching:** Precompute `||z_i'||^2`, co-occurrence counts, `J^(α)`, and, if enabled, per-label top-K2 lists to reduce per-query latency.
