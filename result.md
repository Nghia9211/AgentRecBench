# Evaluation Results

## Metrics Definition

The evaluation of the recommendation agents is based on two primary metrics: **Average Hit Rate @N** and **Normalized Discounted Cumulative Gain (NDCG)**.

### 1. Hit Rate @K (HR@K)

Hit Rate measures the proportion of users for whom the ground-truth item appears within the top $K$ recommendations.

$$
HR@K = \frac{1}{|U|} \sum_{u \in U} \delta(rank_u \leq K)
$$

Where:

- $|U|$ is the total number of users.

- $\delta(\cdot)$ is the indicator function that returns 1 if the condition is true, and 0 otherwise.

- $rank_u$ is the rank of the ground-truth item in the recommendation list for user $u$.

**Average Hit rate @N (N = 1, 3, 5):**

$$
AverageHitRate = \frac{HR@1 + HR@3 + HR@5}{3}
$$

### 2. Normalized Discounted Cumulative Gain @K (NDCG@K)

NDCG evaluates the ranking quality by assigning higher importance to hits at higher ranks.

$$
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}
$$

$$
IDCG@K = \sum_{i=1}^{|REL_K|} \frac{rel_i}{\log_2(i + 1)}
$$

$$
NDCG@K = \frac{DCG@K}{IDCG@K}
$$

Where:

- $rel_i$ is the relevance score of the item at rank $i$ (typically 1 if it is the ground-truth item, 0 otherwise).

- $IDCG@K$ is the ideal DCG, which is the DCG value if the relevant items were perfectly ranked at the top.

## Experimental Results

### Scenario: Classic

#### Average Hit rate @N (N = 1,3,5)

| **Agent**   | **Amazon** | **Goodreads** | **Yelp**  |
| ----------- | ---------- | ------------- | --------- |
| CoT         | 27.90      | 39.44         | 29.77     |
| CoTMemory   | 26.99      | 36.53         | 30.03     |
| Memory      | 26.78      | 35.42         | 29.19     |
| DummyAgent  | 25.24      | 27.23         | 35.76     |
| RecHacker   | 39.52      | 47.37         | 35.78     |
| Baseline666 | 24.99      | 35.13         | 30.73     |
| MoE         | **71.93**  | **69.4**      | **70.2**  |

#### NDCG@5

| **Agent**   | **Amazon** | **Goodreads** | **Yelp**   |
| ----------- | ---------- | ------------- | ---------- |
| CoT         | 0.2721     | 0.3874        | 0.2889     |
| CoTMemory   | 0.2641     | 0.3608        | 0.2918     |
| Memory      | 0.2620     | 0.3492        | 0.2838     |
| DummyAgent  | 0.2468     | 0.2682        | 0.3486     |
| RecHacker   | 0.3867     | 0.4589        | 0.3465     |
| Baseline666 | 0.2450     | 0.3422        | 0.2964     |
| MoE         | **0.7077** | **0.6824**    | **0.6887** |

### Scenario: Cold Start

#### Average Hit rate @N (N = 1,3,5)

| **Agent**   | **Amazon** | **Goodreads** | **Yelp**  |
| ----------- | ---------- | ------------- | --------- |
| CoT         | 27.57      | 43.35         | 29.16     |
| CoTMemory   | 28.19      | 39.51         | 29.87     |
| Memory      | 27.67      | 39.35         | 28.19     |
| DummyAgent  | 29.54      | 29.75         | 33.94     |
| RecHacker   | 43.28      | 45.12         | 33.43     |
| Baseline666 | 30.47      | 38.46         | 30.78     |
| MoE         | **73.87**  | **72.00**     | **77.87** |

#### NDCG@5

| **Agent**   | **Amazon** | **Goodreads** | **Yelp**   |
| ----------- | ---------- | ------------- | ---------- |
| CoT         | 0.2694     | 0.4273        | 0.2897     |
| CoTMemory   | 0.2748     | 0.3867        | 0.2899     |
| Memory      | 0.2696     | 0.3862        | 0.2752     |
| DummyAgent  | 0.2899     | 0.2892        | 0.3329     |
| RecHacker   | 0.4240     | 0.4388        | 0.3254     |
| Baseline666 | 0.2977     | 0.3755        | 0.2986     |
<<<<<<< HEAD
| MoE         | **0.7250** | **0.7080**    | **0.7653** |
=======
| MoE         | -          | -             | -          |
>>>>>>> 7088ba1abb58860e35916f146fb5dcdd99591a6a
