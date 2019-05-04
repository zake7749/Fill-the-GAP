# Fill-the-GAP

This is the 4th solution to the [Gendered Pronoun Resolution Competition](https://www.kaggle.com/c/gendered-pronoun-resolution) on Kaggle.

# Solution Overview

### 1. Input Dropout

I've played with BERT in other tasks where I found there are some redundancies in BERT vector. Even though we only use a small portion (like 50%) of the BERT vector, we still can get desirable performance.

Based on this observation, I placed a dropout with a large rate just after the input layer, which can be considered as a kind of model boosting, just like training several prototypes with subsets that are randomly sampled from the BERT vector.

### 2. Word Encoder

As I mentioned in section 1, it might not be suitable to use the output directly because of redundancies. Therefore I use a word encoder to down-project the BERT vector into a lower-dimensional space where I can extract task-related features efficiently. 

The word encoder is a simple affine transformation with SELU activation and it is shared for A, B, and P. I have tried to design the word encoder for names and pronouns independently or make the word encoder deeper with highway transformations but all of them results in overfitting.

This idea is also inspired by the multi-head transformation. I have implemented a multi-head NLI encoder but it only improved the performance by ~0.0005 and took much computation time. So maybe a single head is good enough for this task.

### 3. Answer selection using NLI architectures

I consider this task a sub-task of answer selection. Given queries A, B, and an answer P, we can model the relations between queries and answers with heuristic interaction:

```
I(Q, A) = [[Q; A], Q - A, Q * A]
```
and then extract features from the interaction vector `I(Q, A)`  with a siamese encoder. The overall architecture would be like this:

![Model](https://i.imgur.com/WGJ9OPK.png)

Finally, here is a simple performacne report of my models:

| Model | 5 fold CV on Stage 1|
| -------- | -------- |
| Base BERT | 0.50    |
| Base BERT + input dropout | 0.45 |
| Base BERT + input dropout + NLI | 0.43 |
| Base BERT + all | 0.39 |
| Large BERT + input dropout | 0.39 |
| Large BERT + all | 0.32 |
| Ensemble of Base BERT and Large BERT | 0.30 |

# Note

The code is still under cleaning. There still exists some dirty methods for the trade-off between efficiency and scalability. For notebook stage 0.1 ~ 0.6, it's not necessary to use a for loop to dump features from each layer. The offical API supports to dump all of them at the same time.
