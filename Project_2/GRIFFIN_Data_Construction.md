# Comprehensive Analysis of CICIoT and CICIoMT Dataset Construction Pipelines

## 1. Dataset Construction Workflow

### **CICIoT Dataset Pipeline (processing_pipeline.ipynb)**

**Workflow Steps:**
1. **Data Loading**: Reads pre-processed CSV from `/ciciot_processed_datasets_corrected/test_50_50.csv`
2. **Data Cleaning**:
   - Column name normalization (strip spaces, replace '/' with '_')
   - Remove duplicate rows
   - Handle infinity values (replace with NaN, then clip to ±1e10)
   - Fill NaN values with 0
   - Remove constant features (variance < 1e-6)
3. **Feature Selection**: 
   - Uses predefined GRIFFIN feature groups
   - Matches features based on normalized column names
   - Selects up to 46 features organized into 5 semantic groups
4. **Label Encoding**: Converts string labels to integers using LabelEncoder
5. **Feature Scaling**: StandardScaler normalization
6. **Data Splitting**: 60% train, 20% validation, 20% test (stratified)
7. **Output**: Saves processed data as NPY arrays and CSV files with metadata

**Initial Features Count**: 47 features (46 numeric + 1 label)
**Final Dataset Size**: 1,139,511 samples with 34 attack classes

### **CICIoMT Dataset Pipeline (data2_construction.ipynb)**

**Workflow Steps:**
1. **Data Loading**: Loads 12 separate PCAP-derived CSV files
2. **Data Concatenation**: Merges all files into single DataFrame
3. **Feature Alignment**: Drops 'IGMP' column from 11 files (missing in first file)
4. **Label Assignment**: Manual assignment based on filename:
   - Benign (2 files)
   - Various attack types (10 files): Recon-Ping_Sweep, Recon-Port_Scan, Recon-VulScan, TCP_IP-DDoS variants
5. **Feature Processing**: Uses same CICIoTDataPipeline class as CICIoT
6. **Data Splitting**: Same 60/20/20 split strategy
7. **Output**: Combined dataset saved as CSV

**Initial Features Count**: 45 features (44 numeric + 1 label)
**Final Dataset Size**: 1,440,278 samples with 8 attack classes

## 2. Clustering and Grouping

### **Feature Clustering Method (Both Datasets)**

Both datasets use **identical manual, domain-based clustering** through the GRIFFIN architecture:

**5 Semantic Feature Groups:**
1. **Packet Statistics Group** (8-10 features)
2. **Time Features Group** (10-11 features)
3. **Flow Rates Group** (10 features)
4. **TCP Flags Group** (9-10 features)
5. **Protocol Info Group** (10 features)

**Clustering Process:**
- **Not algorithmic**: No k-means, hierarchical clustering, or correlation-based methods
- **Domain-driven**: Features grouped by network traffic semantics
- **Static assignment**: Groups remain fixed throughout training
- **Keyword matching**: Features assigned based on name patterns (e.g., 'packet', 'iat', 'flag')

## 3. Labeling and Feature Engineering

### **CICIoT Dataset**
- **Labels**: 34 classes (1 benign + 33 attack types)
- **Label Distribution**: Heavily imbalanced (658,916 benign vs ~751 per attack type)
- **Label Source**: Pre-existing in loaded CSV

### **CICIoMT Dataset**
- **Labels**: 8 classes (1 benign + 7 attack types)
- **Label Distribution**: More balanced than CICIoT
- **Label Assignment**: Manual based on source filename
- **Distribution**:
  - TCP_IP-DDoS-ICMP2: 390,510
  - TCP_IP-DDoS-ICMP1: 348,945
  - Benign: 230,339
  - Others: 1,034 - 189,710

### **Feature Engineering (Both)**
- StandardScaler normalization
- Removal of constant features
- Clipping extreme values
- No new feature creation
- No dimensionality reduction

## 4. Feature Comparison

### **CICIoT Features (46 features)**
```
Packet Stats: Total Fwd/Bwd Packets, Fwd/Bwd Packet Length (Max/Min/Mean/Std)
Time Features: Flow Duration, Flow IAT (Mean/Std/Max/Min), Fwd/Bwd IAT (Total/Mean/Std)
Flow Rates: Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s, Packet Length (Mean/Std/Variance), Down/Up Ratio, Average Packet Size, Fwd Segment Size Avg
TCP Flags: FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Count, Fwd/Bwd PSH Flags
Protocol Info: Protocol, Source/Destination Port, Init Win bytes (fwd/bwd), Active (Mean/Std/Max), Idle (Mean/Std)
```

### **CICIoMT Features (44 features)**
```
Core Features: Header_Length, Protocol Type, Duration, Rate, Srate, Drate
Flag Features: fin/syn/rst/psh/ack/ece/cwr_flag_number, ack/syn/fin/rst_count
Protocol Features: HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IPv, LLC
Statistics: Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Magnitue, Radius, Covariance, Variance, Weight
```

### **Common Features**
- Duration/Flow Duration
- IAT (Inter-Arrival Time)
- Protocol type information
- TCP flag counts (SYN, ACK, FIN, RST, PSH)
- Statistical measures (Min, Max, Mean/AVG, Std)

### **Unique to CICIoT**
- Detailed packet statistics (Fwd/Bwd packet lengths)
- Flow rates (Bytes/s, Packets/s)
- Down/Up Ratio
- Segment Size averages
- Window sizes
- Active/Idle time statistics

### **Unique to CICIoMT**
- Header_Length
- Rate metrics (Rate, Srate, Drate)
- Application protocol indicators (HTTP, HTTPS, DNS, SSH, etc.)
- Mathematical features (Magnitude, Radius, Covariance, Weight)
- Network layer indicators (IPv, LLC)
- Tot sum, Tot size

## 5. Selection Criteria

### **Feature Selection Criteria (Both Datasets)**
1. **Variance Threshold**: Remove features with variance < 1e-6
2. **Semantic Grouping**: Features selected based on GRIFFIN group membership
3. **Keyword Matching**: Column names matched against predefined feature patterns
4. **Fallback Strategy**: If matching fails, select first 46 numeric features

### **Sample Selection**
- **CICIoT**: Uses pre-filtered samples from test_50_50.csv
- **CICIoMT**: Includes all samples from 12 source files
- Both use stratified sampling for train/val/test splits

## 6. Similarities and Differences

### **Similarities**
1. **Same Processing Class**: Both use `CICIoTDataPipeline`
2. **Identical Clustering**: 5 GRIFFIN semantic groups
3. **Same Preprocessing**:
   - StandardScaler normalization
   - Handling infinities and NaN values
   - Constant feature removal
4. **Same Split Strategy**: 60/20/20 train/val/test
5. **Same Output Format**: NPY arrays + CSV files

### **Key Differences**

| Aspect | CICIoT | CICIoMT |
|--------|--------|---------|
| **Data Source** | Single pre-processed CSV | 12 separate PCAP CSVs |
| **Feature Count** | 46 features | 44 features |
| **Feature Types** | Network flow statistics | Mixed (network + math features) |
| **Label Count** | 34 classes | 8 classes |
| **Label Assignment** | Pre-existing in data | Manual from filename |
| **Dataset Size** | 1.14M samples | 1.44M samples |
| **Class Balance** | Highly imbalanced | More balanced |
| **Feature Nature** | Flow-oriented | Packet-oriented |

### **Methodological Differences**
1. **Data Construction**: 
   - CICIoT: Single file load
   - CICIoMT: Multi-file concatenation with column alignment
2. **Feature Philosophy**:
   - CICIoT: Deep flow analysis features
   - CICIoMT: Broader protocol coverage with mathematical features
3. **Attack Coverage**:
   - CICIoT: Comprehensive (34 attack types)
   - CICIoMT: Focused (mainly DDoS and reconnaissance)

## 7. Final Structured Summary

### **Overall Methodology**

Both datasets employ a **unified processing pipeline** with **manual domain-based feature clustering** for the GRIFFIN architecture. The core methodology involves:

1. **Data Preparation**: 
   - CICIoT uses refined flow features from a single source
   - CICIoMT aggregates packet features from multiple PCAP conversions

2. **Feature Organization**: 
   - **Identical clustering strategy**: 5 semantic groups
   - **Different feature philosophies**: Flow-centric (CICIoT) vs Packet-centric (CICIoMT)

3. **Key Distinctions**:
   - **Feature Depth**: CICIoT provides deeper flow analysis with bidirectional statistics
   - **Feature Breadth**: CICIoMT covers more protocols and includes mathematical descriptors
   - **Attack Diversity**: CICIoT (34 types) vs CICIoMT (8 types)

4. **Clustering Strategy**:
   - **Manual, not algorithmic**: Based on domain expertise
   - **Static assignment**: No adaptive clustering
   - **Purpose**: Enable Protocol-Aware Group Gates in GRIFFIN

5. **Feature Selection Philosophy**:
   - **Semantic coherence** over statistical correlation
   - **Interpretability** prioritized for security analysis
   - **Group-level sparsity** for noise reduction

The pipelines demonstrate a **thoughtful balance** between maintaining consistency (same processing class and clustering) while adapting to different data characteristics (flow vs packet features). The manual clustering approach, while less flexible than algorithmic methods, provides **stable, interpretable feature groups** essential for the GRIFFIN architecture's gate mechanism to selectively focus on relevant attack signatures.