# ITE-IDS: A Linear-Complexity Transformer-Based Framework for Real-Time IoT Intrusion Detection

## Algorithm 1: Linformer-IDS Training and Evaluation

**Input:** 
- Preprocessed Dataset $\mathcal{D}$ with features $x \in \mathbb{R}^{d}$ and labels $y$
- Hyperparameters: learning rate $\eta$, batch size $B$, epochs $E$, number of encoder layers $L$, embedding dimension $d_{model}$, projected dimension $k$

**Output:** Trained Linformer-IDS model $\omega$

### Steps

1. Split $\mathcal{D}$ into training $\mathcal{D}_{train}$, validation $\mathcal{D}_{val}$, and test $\mathcal{D}_{test}$ sets.

2. Initialize Linformer-IDS model $\omega$ and Adam optimizer.

3. Initialize Loss Function $\mathcal{L}$ (e.g., Focal Loss or Cross-Entropy).

4. **Define Linformer Self-Attention**

5. **function** LinformerAttention($Q, K, V$):
   
   - **Input:** Query $Q \in \mathbb{R}^{n \times d_k}$, Key $K \in \mathbb{R}^{n \times d_k}$, Value $V \in \mathbb{R}^{n \times d_v}$
   
   - Initialize projection matrices $E, F \in \mathbb{R}^{k \times n}$
   
   - Project Key and Value to a smaller, fixed-size dimension $k$:
     - $\bar{K} \leftarrow E K$ (where $\bar{K} \in \mathbb{R}^{k \times d_k}$)
     - $\bar{V} \leftarrow F V$ (where $\bar{V} \in \mathbb{R}^{k \times d_v}$)
   
   - Compute attention scores against the compressed Key matrix:
     - $P \leftarrow \text{Softmax}\left(\frac{Q \bar{K}^T}{\sqrt{d_k}}\right)$ (Attention weights $P \in \mathbb{R}^{n \times k}$)
   
   - Apply attention weights to the compressed Value matrix:
     - **return** $P \bar{V}$ (Output $\in \mathbb{R}^{n \times d_v}$)

6. **end function**

### Main Training Loop

7. **for** $e \leftarrow 1$ to $E$ **do**
   
   **Training Phase:**
   
   8. **for each** mini-batch $\{x_i, y_i\}_{i=1}^B$ from $\mathcal{D}_{train}$ **do**
      - Pass input through stacked Linformer Encoder layers: $\hat{y}_i \leftarrow \text{Linformer-IDS}(x_i; \omega)$
      - Compute loss and update weights:
        - loss $\leftarrow \mathcal{L}(\hat{y}_i, y_i)$
        - loss.backward()
        - optimizer.step()
   
   9. **end for**
   
   **Validation Phase:**
   
   10. Evaluate performance on $\mathcal{D}_{val}$; save best model checkpoint $\omega_{best}$

11. **end for**

### Final Evaluation

12. Load $\omega_{best}$

13. Compute final metrics (Accuracy, Precision, Recall, F1-Score) on $\mathcal{D}_{test}$

14. Generate Confusion Matrix and ROC Curve plots

---

## Understanding the Lightweight Transformer (Linformer)

The Linformer's key innovation lies in redesigning the self-attention mechanism to eliminate the computational bottleneck that plagues standard Transformers.

### The Challenge: Quadratic Complexity in Standard Attention

In a standard Transformer, self-attention is computed as:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The computational bottleneck occurs during the $QK^T$ matrix multiplication. When the input sequence has length $n$ (representing, for example, $n$ features from network traffic), both the Query ($Q$) and Key ($K$) matrices have dimensions $n \times d_k$. Their multiplication produces an $n \times n$ attention matrix, resulting in **$O(n^2)$ complexity**. This makes the model computationally expensive and memory-intensive for long sequences.

### The Linformer Solution: Linear Complexity Through Projection

The Linformer operates on the principle that the large $n \times n$ attention matrix contains redundant information and can be effectively approximated using a compressed representation. Rather than computing the full attention matrix, Linformer projects the Key and Value matrices into a smaller, fixed-size representation.

#### Step 1: Projection to Compressed Representation

Before computing attention, Linformer introduces two trainable linear projection matrices, $E$ and $F$. These matrices compress the Key ($K$) and Value ($V$) matrices from sequence length $n$ down to a much smaller, fixed dimension $k$ (for example, $k=256$).

- $\bar{K} = E K$ (Compressed Key)
- $\bar{V} = F V$ (Compressed Value)

This projection creates a compact, fixed-size summary of the sequence's information. Regardless of how long the input sequence $n$ is, the key information is distilled into this compressed representation of size $k$.

#### Step 2: Linear Attention Computation

The Query ($Q$) matrix then interacts with the compressed summary ($\bar{K}$) instead of the full Key matrix ($K$):

$$\text{LinformerAttention} = \text{Softmax}\left(\frac{Q \bar{K}^T}{\sqrt{d_k}}\right)\bar{V}$$

The critical difference is that the matrix multiplication $Q \bar{K}^T$ now operates between a matrix of size $n \times d_k$ and one of size $k \times d_k$. The resulting attention map has dimensions $n \times k$.

Since $k$ is a fixed hyperparameter independent of input length $n$, the overall complexity reduces to **$O(n)$**. This linear scaling makes the Linformer lightweight and efficient.

### Why This Matters for Intrusion Detection Systems

This efficiency is crucial for IDS applications. The linear complexity enables the model to process high-volume network traffic in real-time and makes deployment on network edge devices feasible due to significantly reduced memory and CPU requirements.