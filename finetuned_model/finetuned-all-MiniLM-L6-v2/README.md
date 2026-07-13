---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1437
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What new vision Transformer is presented in this paper?
  sentences:
  - 'Paper: Dqn Domain: RL Content: C. Further Comparison


    To further explore the performance of our method, we com- pare our method with
    the latest variants of DQN, the Double DQN [44], and the state-of-the-art DRL
    method CDQN [45]. For fair comparison, we adapt our proposed deep spiking reinforcement
    learning strategy to the comparison baselines, named Double DSQN and CDSQN.


    The Double DQN and DSQN are trained on 1M timesteps, and the CDQN and CDSQN are
    trained on 10M timesteps. The hyperparameters of the Double DQN and CDQN follow
    their references [44] and [45] respectively. Other experimental setups are the
    same as previous experiments.


    The code of baselines are in the open source github 5,6. The experiments are conducted
    based on the OpenAI Gym7.


    TABLE VIII COMPARISON RESULTS OF DOUBLE DQN AND DOUBLE DSQN.


    Game


    Double DQN


    Double DSQN


    Score(% Double DQN)'
  - 'Paper: Swin Transformer Domain: CV Content: 5. Conclusion


    This paper presents Swin Transformer, a new vision Transformer which produces
    a hierarchical feature repre-'
  - 'Paper: Roberta Domain: NLP Content: Table 7: Results on the RACE test set. BERTLARGE
    and XLNetLARGE results are from Yang et al. (2019).


    nating each candidate answer with the correspond- ing question and passage. We
    then encode each of these four sequences and pass the resulting [CLS] representations
    through a fully-connected layer, which is used to predict the correct answer.
    We truncate question-answer pairs that are longer than 128 tokens and, if needed,
    the passage so that the total length is at most 512 tokens.


    Results on the RACE test sets are presented in Table 7. RoBERTaachieves state-of-the-art
    results on both middle-school and high-school settings.'
- source_sentence: What competitions and tracks did ResNet win 1st place in during
    ILSVRC & COCO 2015?
  sentences:
  - 'Paper: Dqn Domain: RL Content: 3


    2023


    2


    0


    2


    r p A 1 1 ] E N . s c [ 3 v 1 1 2 7 0 . 1 0 2


    2


    :


    v


    arXiv


    i


    X


    r


    a


    JOURNAL OF X, VOL. X, NO. X, X X


    Human-Level Control through Directly-Trained Deep Spiking Q-Networks


    Guisong Liu, Wenjie Deng, Xiurui Xie, Li Huang, Huajin Tang


    Abstract—As the third-generation neural networks, Spiking Neural Networks (SNNs)
    have great potential on neuromorphic hardware because of their high energy-efﬁciency.
    However, Deep Spiking Reinforcement Learning (DSRL), i.e., the Reinforcement Learning
    (RL) based on SNNs, is still in its preliminary stage due to the binary output
    and the non-differentiable property of the spiking function. To address these
    issues, we propose a Deep Spiking Q-Network (DSQN) in this paper. Speciﬁcally,
    we propose a directly-trained deep spiking reinforcement learning archi- tecture
    based on the Leaky Integrate-and-Fire (LIF) neurons and Deep Q-Network (DQN).
    Then, we adapt a direct spiking learning algorithm for the Deep Spiking Q-Network.
    We further demonstrate the advantages of using LIF neurons in DSQN theoretically.
    Comprehensive experiments have been conducted on 17 top-performing Atari games
    to compare our method with the state-of-the-art conversion method. The experimental
    results demonstrate the superiority of our method in terms of performance, stability,
    generalization and energy-efﬁciency. To the best of our knowledge, our work is
    the ﬁrst one to achieve state-of-the-art performance on multiple Atari games with
    the directly-trained SNN.


    demonstrated competitive performance of SNNs compared with ANNs on image classiﬁcation
    [5], object recognition [6], [7], speech recognition [8], [9], and other ﬁelds
    [10]–[15].


    The present work focuses on combining SNNs with Deep Reinforcement Learning (DRL),
    i.e., Deep Spiking Rein- forcement Learning (DSRL), on Atari games. Compared to
    image classiﬁcation, DSRL on Atari games involve additional complexity due to
    the pixel image as input and the partial observability of the environment. The
    development of DSRL lags behind DRL, while DRL has made tremendous successes,
    achieved and even surpassed human-level performance in many Reinforcement Learning
    (RL) tasks [16]–[21]. The main reason is that, training SNNs is a challenge, as
    the event- driven spiking activities are discrete and non-differentiable. In addition,
    the activities of spiking neurons are propagated not only in spatial domain layer-by-layer,
    but also along temporal domain [22]. It makes the training of SNNs in reinforcement
    learning more difﬁcult.'
  - 'Paper: Sbert Domain: NLP Content: Table 3: Average Pearson correlation r and
    average Spearman’s rank correlation ρ on the Argument Facet Similarity (AFS) corpus
    (Misra et al., 2016). Misra et al. proposes 10-fold cross-validation. We additionally
    evaluate in a cross-topic scenario: Methods are trained on two topics, and are
    evaluated on the third topic.


    tences in the same section are thematically closer than sentences in different
    sections. They use this to create a large dataset of weakly labeled sen- tence
    triplets: The anchor and the positive exam- ple come from the same section, while
    the neg- ative example comes from a different section of the same article. For
    example, from the Alice Arnold article: Anchor: Arnold joined the BBC Radio Drama
    Company in 1988., positive: Arnold gained media attention in May 2012., negative:
    Balding and Arnold are keen amateur golfers.


    We use the dataset from Dor et al. We use the Triplet Objective, train SBERT for
    one epoch on the about 1.8 Million training triplets and evaluate it on the 222,957
    test triplets. Test triplets are from a distinct set of Wikipedia articles. As
    evaluation metric, we use accuracy: Is the positive example closer to the anchor
    than the negative example?


    Results are presented in Table 4. Dor et al. ﬁne- tuned a BiLSTM architecture
    with triplet loss to derive sentence embeddings for this dataset. As the table
    shows, SBERT clearly outperforms the BiLSTM approach by Dor et al.'
  - 'Paper: Resnet Domain: CV Content: 4.3. Object Detection on PASCAL and MS COCO


    Our method has good generalization performance on other recognition tasks. Table
    7 and 8 show the object de- tection baseline results on PASCAL VOC 2007 and 2012
    [5] and COCO [26]. We adopt Faster R-CNN [32] as the de- tection method. Here
    we are interested in the improvements of replacing VGG-16 [41] with ResNet-101.
    The detection implementation (see appendix) of using both models is the same,
    so the gains can only be attributed to better networks. Most remarkably, on the
    challenging COCO dataset we ob- tain a 6.0% increase in COCO’s standard metric
    (mAP@[.5, .95]), which is a 28% relative improvement. This gain is solely due
    to the learned representations.


    Based on deep residual nets, we won the 1st places in several tracks in ILSVRC
    & COCO 2015 competitions: Im- ageNet detection, ImageNet localization, COCO detection,
    and COCO segmentation. The details are in the appendix.


    8'
- source_sentence: How do EfficientNet (EffNet) variants compare in terms of parameters
    and top-1 accuracy on ImageNet-1K?
  sentences:
  - 'Paper: Ddpg Domain: RL Content: This approach has several beneﬁts, including
    data efﬁciency, and has been applied successfully to a variety of real-world robotic
    manipulation tasks using vision. In these tasks GPS uses a similar convolutional
    policy network to ours with 2 notable exceptions: 1. it uses a spatial softmax
    to reduce the dimensionality of visual features into a single (x,y) coordinate
    for each feature map, and 2. the policy also receives direct low-dimensional state
    information about the conﬁguration of the robot at the ﬁrst fully connected layer
    in the network. Both likely increase the power and data efﬁciency of the algorithm
    and could easily be exploited within the DDPG framework.


    PILCO (Deisenroth & Rasmussen, 2011) uses Gaussian processes to learn a non-parametric,
    proba- bilistic model of the dynamics. Using this learned model, PILCO calculates
    analytic policy gradients and achieves impressive data efﬁciency in a number of
    control problems. However, due to the high computational demand, PILCO is “impractical
    for high-dimensional problems” (Wahlstr¨om et al., 2015). It seems that deep function
    approximators are the most promising approach for scaling rein- forcement learning
    to large, high-dimensional domains.


    Wahlstr¨om et al. (2015) used a deep dynamical model network along with model
    predictive control to solve the pendulum swing-up task from pixel input. They
    trained a differentiable forward model and encoded the goal state into the learned
    latent space. They use model-predictive control over the learned model to ﬁnd
    a policy for reaching the target. However, this approach is only applicable to
    domains with goal states that can be demonstrated to the algorithm.


    Recently, evolutionary approaches have been used to learn competitive policies
    for Torcs from pixels using compressed weight parametrizations (Koutn´ık et al.,
    2014a) or unsupervised learning (Koutn´ık et al., 2014b) to reduce the dimensionality
    of the evolved weights. It is unclear how well these approaches generalize to
    other problems.'
  - 'Paper: Att Unet Domain: CV Content: Table 3: Pancreas segmentation results obtained
    on the TCIA Pancreas-CT Dataset [25]. The dataset contains in total 82 scans which
    are split into training (61) and testing (21) sets. The corresponding results
    are obtained before (BFT) and after ﬁne tuning (AFT) and also training the models
    from scratch (SCR). Statistically signiﬁcant results are highlighted in bold font.


    Method


    Dice Score


    Precision


    Recall


    S2S Dist (mm)'
  - 'Paper: Swin Transformer Domain: CV Content: (a) Regular ImageNet-1K trained models
    method image size #param. FLOPs throughput (image / s) RegNetY-4G [48] 2242 21M
    4.0G 1156.7 80.0 RegNetY-8G [48] 2242 39M 8.0G 591.6 81.7 84M 16.0G 334.7 82.9
    EffNet-B3 [58] 3002 12M 1.8G 732.1 81.6 EffNet-B4 [58] 3802 19M 4.2G 349.4 82.9
    EffNet-B5 [58] 4562 30M 9.9G 169.1 83.6 EffNet-B6 [58] 5282 43M 19.0G 96.9 84.0
    EffNet-B7 [58] 6002 66M 37.0G 55.1 84.3 ViT-B/16 [20] 3842 86M 55.4G 85.9 77.9
    ViT-L/16 [20] 3842 307M 190.7G 27.3 76.5 DeiT-S [63] 2242 22M 4.6G 940.4 79.8
    DeiT-B [63] 2242 86M 17.5G 292.3 81.8 DeiT-B [63] 3842 86M 55.4G 85.9 83.1 Swin-T
    2242 29M 4.5G 755.2 81.3 Swin-S 2242 50M 8.7G 436.9 83.0 Swin-B 2242 88M 15.4G
    278.1 83.5'
- source_sentence: What are the key connections between Trust Region Policy Optimization
    (TRPO) and methods like Relative Entropy Policy Search (REPS) or the KL divergence
    approach by Levine and Abbeel (2014)?
  sentences:
  - 'Paper: Unet++ Domain: CV Content: UNet++ L! A UNet++ L? UNet++ L UNet++ L4 Param
    = 0.1M Param = 0.5M Param = 2.2M Param = 9.0M 95 40 90 80 S 04 36 86 78 > @ ce
    93 . o e@ 2 AG @ “*% 4a 8 92 Aa 28 A 78 . 74 2 91 ) 04 to) 14 72 @ 90 20 70 70
    0 20 40 60 0 20 40 60 0 20 40 60 0 400 800 1200 Inference Time (s) Inference Time
    (s) Inference Time (s) Inference Time (s) (a) (b) (c) (d)


    Fig.3: Complexity, speed, and accuracy of UNet++ after pruning on (a) cell nuclei,
    (b) colon polyp, (c) liver, and (d) lung nodule segmentation tasks respec- tively.
    The inference time is the time taken to process 10k test images using one NVIDIA
    TITAN X (Pascal) with 12 GB memory.


    slices; and thus, a multi-scale approach using all segmentation branches (deep
    supervision) is essential for accurate segmentation. Fig. 2 shows a qualitative
    comparison between the results of U-Net, wide U-Net, and UNet++.


    Model pruning: Fig. 3 shows segmentation performance of UNet++ after ap- plying
    diﬀerent levels of pruning. We use UNet++ Li to denote UNet++ pruned at level
    i (see Fig. 1c for further details). As seen, UNet++ L3 achieves on av- erage
    32.2% reduction in inference time while degrading IoU by only 0.6 points. More
    aggressive pruning further reduces the inference time but at the cost of signiﬁcant
    accuracy degradation.'
  - 'Paper: Trpo Domain: RL Content: 7 Connections with Prior Work


    As mentioned in Section 4, our derivation results in a pol- icy update that is
    related to several prior methods, provid- ing a unifying perspective on a number
    of policy update


    Trust Region Policy Optimization


    schemes. The natural policy gradient (Kakade, 2002) can be obtained as a special
    case of the update in Equation (12) by using a linear approximation to L and a
    quadratic ap- proximation to the DKL constraint, resulting in the follow- ing
    problem:


    maximize [VoLo.rs (|g gy. ° (8 — Bota)| (17)


    subject to


    1


    2


    (θold − θ)TA(θold)(θold − θ) ≤ δ,


    where A(θold)ij =


    00, Ao Dp Esxon De((-|8, Bota) [I 18: 9))] laoara a J


    The update is Qrew = 9o1q + +A(Oo1a)*VoL(9)|5_, a where the stepsize + is typically
    treated as an algorithm parameter. This differs from our approach, which en- forces
    the constraint at each update. Though this difference might seem subtle, our experiments
    demonstrate that it sig- nificantly improves the algorithm’s performance on larger
    problems.


    We can also obtain the standard policy gradient update by using an £5 constraint
    or penalty:


    maximize VoL ons log (O- Go1a)| (18) 1 subject to 56 — Bo1al|? <6.


    Figure 2. 2D robot models used for locomotion experiments. From left to right:
    swimmer, hopper, walker. The hopper and walker present a particular challenge,
    due to underactuation and contact discontinuities.


    Fully Input connected Mean layer layer Sampling @ parameters > > Control Joint
    angles and kinematics . Standard 30 units deviations Input Conv. Conv. Hidden
    Action Samplin layer layer layer layer probabilities ping Y\/> Control Screen
    input 16 filters 16 filters 20 units


    The policy iteration update can also be obtained by solving the unconstrained
    problem maximizeπ Lπold(π), using L as deﬁned in Equation (3).


    Figure 3. Neural networks used for the locomotion task (top) and for playing Atari
    games (bottom).


    Several other methods employ an update similar to Equa- tion (12). Relative entropy
    policy search (REPS) (Peters et al., 2010) constrains the state-action marginals
    p(s,a), while TRPO constrains the conditionals p(a|s). Unlike REPS, our approach
    does not require a costly nonlinear op- timization in the inner loop. Levine and
    Abbeel (2014) also use a KL divergence constraint, but its purpose is to encour-
    age the policy not to stray from regions where the estimated dynamics model is
    valid, while we do not attempt to esti- mate the system dynamics explicitly. Pirotta
    et al. (2013) also build on and generalize Kakade and Langford’s results, and
    they derive different algorithms from the ones here.'
  - 'Paper: Sbert Domain: NLP Content: Table 4: Evaluation on the Wikipedia section
    triplets dataset (Dor et al., 2018). SBERT trained with triplet loss for one epoch.


    The purpose of SBERT sentence embeddings are not to be used for transfer learning
    for other tasks. Here, we think ﬁne-tuning BERT as de- scribed by Devlin et al.
    (2018) for new tasks is the more suitable method, as it updates all layers of
    the BERT network. However, SentEval can still give an impression on the quality
    of our sentence embeddings for various tasks.


    We compare the SBERT sentence embeddings to other sentence embeddings methods
    on the fol- lowing seven SentEval transfer tasks:


    • MR: Sentiment prediction for movie reviews snippets on a ﬁve start scale (Pang
    and Lee, 2005).


    • CR: Sentiment prediction of customer prod- uct reviews (Hu and Liu, 2004).


    • SUBJ: Subjectivity prediction of sentences from movie reviews and plot summaries
    (Pang and Lee, 2004).


    • MPQA: Phrase level opinion polarity classi- ﬁcation from newswire (Wiebe et
    al., 2005).


    • SST: Stanford Sentiment Treebank with bi- nary labels (Socher et al., 2013).


    • TREC: Fine grained question-type classiﬁ- cation from TREC (Li and Roth, 2002).


    • MRPC: Microsoft Research Paraphrase Cor- pus from parallel news sources (Dolan
    et al., 2004).


    The results can be found in Table 5. SBERT is able to achieve the best performance
    in 5 out of 7 tasks. The average performance increases by about 2 percentage points
    compared to In- ferSent as well as the Universal Sentence Encoder. Even though
    transfer learning is not the purpose of SBERT, it outperforms other state-of-the-art
    sen- tence embeddings methods on this task.'
- source_sentence: What are the ROUGE R1, R2, and RL scores for BART on the CNN/DailyMail
    and XSum datasets?
  sentences:
  - 'Paper: Resnet Domain: CV Content: LOC method LOC network testing LOC error on
    GT CLS classiﬁcation network top-5 LOC error on predicted CLS VGG’s [41] VGG-16
    1-crop 33.1 [41] RPN ResNet-101 1-crop 13.3 RPN ResNet-101 dense 11.7 RPN ResNet-101
    dense ResNet-101 14.4 RPN+RCNN ResNet-101 dense ResNet-101 10.6 RPN+RCNN ensemble
    dense ensemble 8.9'
  - 'Paper: Trpo Domain: RL Content: B. Rider Breakout Enduro Pong Q*bert Seaquest
    S. Invaders Random 354 1.2 0 −20.4 157 110 179 Human (Mnih et al., 2013) 7456
    31.0 368 −3.0 18900 28010 3690 Deep Q Learning (Mnih et al., 2013) 4092 168.0
    470 20.0 1952 1705 581 UCC-I (Guo et al., 2014) 5702 380 741 21 20025 2995 692
    TRPO - single path 1425.2 10.8 534.6 20.9 1973.5 1908.6 568.4 TRPO - vine 859.5
    34.2 430.8 20.9 7732.5 788.4 450.2'
  - 'Paper: Bart Domain: NLP Content: CNN/DailyMail XSum R1 R2 RL R1 R2 RL Lead-3
    40.42 17.62 36.67 16.30 1.60 11.95 PTGEN (See et al., 2017) 36.44 15.66 33.42
    29.70 9.21 23.24 PTGEN+COV (See et al., 2017) 39.53 17.28 36.38 28.10 8.02 21.72
    UniLM 43.33 20.21 40.51 - - - BERTSUMABS (Liu & Lapata, 2019) 41.72 19.39 38.76
    38.76 16.33 31.15 BERTSUMEXTABS (Liu & Lapata, 2019) 42.13 19.60 39.18 38.81 16.50
    31.27 BART 44.16 21.28 40.90 45.14 22.27 37.25'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What are the ROUGE R1, R2, and RL scores for BART on the CNN/DailyMail and XSum datasets?',
    'Paper: Bart Domain: NLP Content: CNN/DailyMail XSum R1 R2 RL R1 R2 RL Lead-3 40.42 17.62 36.67 16.30 1.60 11.95 PTGEN (See et al., 2017) 36.44 15.66 33.42 29.70 9.21 23.24 PTGEN+COV (See et al., 2017) 39.53 17.28 36.38 28.10 8.02 21.72 UniLM 43.33 20.21 40.51 - - - BERTSUMABS (Liu & Lapata, 2019) 41.72 19.39 38.76 38.76 16.33 31.15 BERTSUMEXTABS (Liu & Lapata, 2019) 42.13 19.60 39.18 38.81 16.50 31.27 BART 44.16 21.28 40.90 45.14 22.27 37.25',
    'Paper: Resnet Domain: CV Content: LOC method LOC network testing LOC error on GT CLS classiﬁcation network top-5 LOC error on predicted CLS VGG’s [41] VGG-16 1-crop 33.1 [41] RPN ResNet-101 1-crop 13.3 RPN ResNet-101 dense 11.7 RPN ResNet-101 dense ResNet-101 14.4 RPN+RCNN ResNet-101 dense ResNet-101 10.6 RPN+RCNN ensemble dense ensemble 8.9',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.6112,  0.1545],
#         [ 0.6112,  1.0000, -0.0396],
#         [ 0.1545, -0.0396,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,437 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                           |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               |
  | modality | text                                                                               | text                                                                                 |
  | details  | <ul><li>min: 11 tokens</li><li>mean: 26.05 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 212.16 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  |:-----------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What are the key differences between the deterministic variant of SAC used in the ablation study and DDPG?</code>            | <code>Paper: Sac Domain: RL Content: 5.2. Ablation Study<br><br>Humanoid (rllab)<br><br>stochastic policy 600 — deterministic policy 400 average return 0 2 4 6 million steps<br><br>Figure 2. Comparison of SAC (blue) and a deterministic variant of SAC (red) in terms of the stability of individual random seeds on the Humanoid (rllab) benchmark. The comparison indicates that stochasticity can stabilize training as the variability between the seeds becomes much higher with a deterministic policy.<br><br>seeds. Soft actor-critic performs much more consistently, while the deterministic variant exhibits very high variability across seeds, indicating substantially worse stability. As evident from the ﬁgure, learning a stochastic policy with entropy maximization can drastically stabilize training. This becomes especially important with harder tasks, where tun- ing hyperparameters is challenging. In this comparison, we updated the target value network weights with hard updates, by periodically overwriting the target network ...</code> |
  | <code>Why is BERT out-of-the-box unsuitable for sentence similarity tasks and what solution did SBERT propose?</code>              | <code>Paper: Sbert Domain: NLP Content: 8 Conclusion<br><br>We showed that BERT out-of-the-box maps sen- tences to a vector space that is rather unsuit- able to be used with common similarity measures like cosine-similarity. The performance for seven STS tasks was below the performance of average GloVe embeddings.<br><br>To overcome this shortcoming, we presented Sentence-BERT (SBERT). SBERT ﬁne-tunes BERT in a siamese / triplet network architec- ture. We evaluated the quality on various com- mon benchmarks, where it could achieve a sig- niﬁcant improvement over state-of-the-art sen- tence embeddings methods. Replacing BERT with RoBERTa did not yield a signiﬁcant improvement in our experiments.<br><br>SBERT is computationally efﬁcient. On a GPU, it is about 9% faster than InferSent and about 55% faster than Universal Sentence Encoder. SBERT can be used for tasks which are computationally not feasible to be modeled with BERT. For exam- ple, clustering of 10,000 sentences with hierarchi- cal clustering requires...</code>       |
  | <code>How is softmax-temperature (T) used in knowledge distillation, and what is its purpose during training and inference?</code> | <code>Paper: Distilbert Domain: NLP Content: We have made the trained weights available along with the training code in the Transformers2<br><br>library from HuggingFace [Wolf et al., 2019].<br><br>2 Knowledge distillation<br><br>Knowledge distillation [Bucila et al., 2006, Hinton et al., 2015] is a compression technique in which a compact model - the student - is trained to reproduce the behaviour of a larger model - the teacher - or an ensemble of models.<br><br>In supervised learning, a classiﬁcation model is generally trained to predict an instance class by maximizing the estimated probability of gold labels. A standard training objective thus involves minimizing the cross-entropy between the model’s predicted distribution and the one-hot empirical distribution of training labels. A model performing well on the training set will predict an output distribution with high probability on the correct class and with near-zero probabilities on other classes. But some of these "near-zero" probabilities are larger than ...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 42.3 seconds

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.6.0
- Transformers: 5.10.2
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->