5202
yaM
03
]LM.tats[
1v43142.5052:viXra
A Mathematical Perspective On Contrastive Learning
Ricardo Baptista1, Andrew Stuart1,2, and Son Tran1
1Stores Foundational AI, Amazon, Palo Alto CA 94301 and Pasadena CA 91125
ricarsb@amazon.com, andrxstu@amazon.com, sontran@amazon.com
2Computing and Mathematical Sciences, California Institute of Technology, Pasadena CA 91125
Abstract: Multimodalcontrastivelearningisamethodologyforlinkingdifferentdatamodali-
ties;thecanonicalexampleislinkingimageandtextdata.Themethodologyistypicallyframed
astheidentificationofasetofencoders,oneforeachmodality,thatalignrepresentationswithin
acommonlatentspace.Inthiswork,wefocusonthebimodalsettingandinterpretcontrastive
learning as the optimization of (parameterized) encoders that define conditional probability
distributions, for each modality conditioned on the other, consistent with the available data.
This provides a framework for multimodal algorithms such as crossmodal retrieval, which
identifies the mode of one of these conditional distributions, and crossmodal classification,
which is similar to retrieval but includes a fine-tuning step to make it task specific.
The framework we adopt also gives rise to crossmodal generative models. This probabilistic
perspective suggests two natural generalizations of contrastive learning: the introduction of
novel probabilistic loss functions, and the use of alternative metrics for measuring alignment
in the common latent space. We study these generalizations of the classical approach in the
multivariate Gaussian setting. In this context we view the latent space identification as a
low-rank matrix approximation problem. This allows us to characterize the capabilities of loss
functions and alignment metrics to approximate natural statistics, such as conditional means
and covariances; doing so yields novel variants on contrastive learning algorithms for specific
mode-seeking and for generative tasks. The framework we introduce is also studied through
numerical experiments on multivariate Gaussians, the labeled MNIST dataset, and on a data
assimilation application arising in oceanography.
Keywords and phrases: Multimodal data analysis, Contrastive learning, Conditional distri-
butions, Latent space, Low-rank approximations.
1. Introduction
Contrastive learning is a computational methodology for identifying mechanisms by which different
data modalities, derived from a common underlying reality, can communicate with one another; the
canonical example is linking image and text data. The goal of this paper is to formulate the problem
of contrastive learning in the language of probability measures and to use this framing both to shed
light on existing algorithms and to suggest novel variants of them. The subject is illustrated with
explicit theory for Gaussians; by numerical experiments for Gaussians; by demonstrating how image
classification fits within our general framework; and by a novel application of the methodology to a
data assimilation problem arising in the ocean sciences. In Subsection 1.1 we set our work in context,
giving two illustrative examples in Subsection 1.2. Subsection 1.3 describes our contributions and
provides an overview of the paper. A literature review may be found in Subsection 1.4.
1.1. Context
The recent advances in artificial intelligence have been driven to a large extent by rapidly increasing
acquisition of data, by innovations in the design of novel function classes for approximation (architec-
tures), by novel training algorithms (optimizers) to identify suitable candidates from these function
classes, and by advances in computer hardware and software. A fundamental challenge in the use of
1

Baptista, Stuart, Tran/Contrastive Learning 2
data is to combine information from multiple sources, often known as modalities, such as language,
audio, image and video, that are not necessarily represented in the same spaces. Multimodal learning
addresses this fundamental problem. A successful and widely adopted methodology in this arena is
contrastive learning. In the bimodal context, contrastive learning is a way of aligning two different
data modalities. This paper focuses on the study of contrastive learning in the bimodal setting.
Consider the two different modalities as elements u and v, from spaces , . Contrastive learning
U V
is based on observing data pairs from and aims to find a common latent space in which
U ×V
the pairs can be aligned. In contrast to supervised learning, which aims to find a map from to
U
or vice versa, linking the pairs, contrastive learning focuses on learning about the conditional
V
distributions on u v and on v u. A useful way of conceptualizing the problem is to consider u and v
| |
to be distinct noisy and indirect measurements of an element w in a third space :
W
u = f (w,η ), (1a)
u u
v = f (w,η ). (1b)
v v
Here f ( ,η ) (respectively f ( ,η )) captures the measurement process leading to an element in
u u v v
· · U
(resp. ) and η (resp. η ) represents the measurement noise that may enter this process. Assuming
u v
V
that w is a random variable in , this set-up implies a joint distribution µ(u,v) on . In the
W U ×V
bimodal setting, contrastive learning is based on observing data pairs from this joint distribution
and aims to find a common latent space in which the conditional distributions for u v and v u can
| |
be readily computed. The space and the probability measure on it will play no explicit role in
W
our developments of this subject, but it remains a useful conceptual underpinning of contrastive
learning as a source for the implied nontrivial conditional structure existing between the two modes
with joint distribution µ.
1.2. Illustrative Examples
Images and Text Arguably, the canonical example of bimodal information is that of image-
text pairs [28]. Allowing these two modalities to communicate is at the heart of problems such as
document retrieval [25, 18], text-guided image generation [30, 26] and multimodal representation
learning [5, 15, 6]. To place this problem in our general framework, we consider the setting where u is
apixelatedcolour(RGB)image,representedasaflattenedvectorin := Rnu wheren = 3p2,withp
u
U
beingthenumberofpixelsineachofthetwoimagedimensionsandthemultiplier3accountingforthe
three RGB channels. On the other hand text may be represented as a sequence over a finite alphabet
A. Choice of the alphabet is critical to empirical success and byte-pair encoding [13] is commonly
adopted. In practice a finite length N is imposed on the sequence and so = f : [[1,N]] A .1 If
V { → }
we view pixelated image u and text v as, respectively, photographic and textual descriptions
∈ U ∈ V
of a common real scene w , then we expect an implied joint distribution on image-text pairs in
∈ W
, with non-trivial conditional dependencies.
U ×V
This set-up can be generalized in numerous directions. Perhaps most important is to highlight that
multimodal variants, with more than two modalities, arise naturally and can be formulated, analyzed
and studied similarly to the bimodal setting in this paper. In particular we can consider, for example,
video and audio data, derived from the same common real scene. Or we have may have access to
textual data not present in the underlying real scene, but that is implied by or complementary to it.
Eulerian and Lagrangian Visualization of Fluid Flow A common problem in oceanography
is to simultaneously use direct observation of ocean currents (the Eulerian picture [2]) and indirect
1We employ the notation [[1,N]]:={1,··· ,N}.

|     |     |     | Baptista, | Stuart, | Tran/Contrastive | Learning | 3   |
| --- | --- | --- | --------- | ------- | ---------------- | -------- | --- |
observations through the transport of objects moving in those currents (the Lagrangian picture [17]);
see [9]. It is therefore of interest to ask how to align these two different modalities. The problem of
recovering Eulerian velocity fields from Lagrangian information is often referred to as Lagranian
| data | assimilation | [21]. |     |     |     |     |     |
| ---- | ------------ | ----- | --- | --- | --- | --- | --- |
To formulate the problem as one in the framework of contrastive learning we consider an idealized
setting of flow in a periodic geometry. Let Td denote the d dimensional torus, and consider a
−
velocity field w := C1(Td,Rd). Eulerian observations of an element in are defined by
|     |        | Rd×Ju ∈ W |       |          |             | W     |     |
| --- | ------ | --------- | ----- | -------- | ----------- | ----- | --- |
| f ( | ,η ) : |           | where |          |             |       |     |
| u · | u W →  |           |       |          |             |       |     |
|     |        |           |       |          | )+ηj        | Ju    |     |
|     |        |           |       | f u (w,η | u ) = w(x j | ,     |     |
|     |        |           |       |          | {           | u}j=1 |     |
defined through a specified set of observation points x Td, j = 1, ,J , and subject to
j u
∈ ···
i.i.d.additivenoiseηj.Lagrangianobservationsaredefinedbyconsideringtrajectoriesz C1([0,T];Rd)
u
∈
| governed | by the | ordinary | differential | equation |              |     |     |
| -------- | ------ | -------- | ------------ | -------- | ------------ | --- | --- |
|          |        |          |              | z˙ =     | w(z), z(0) = | z , |     |
0
| and | defining f | ( ,η ) | : Rd×Jv | by  |     |     |     |
| --- | ---------- | ------ | ------- | --- | --- | --- | --- |
|     | v          | u      |         |     |     |     |     |
· W →
|     |     |     |     | f (w,η | ) = z(t )+ηj | Jv ;  |     |
| --- | --- | --- | --- | ------ | ------------ | ----- | --- |
|     |     |     |     | v      | v j          |       |     |
|     |     |     |     |        | {            | v}j=1 |     |
thus, map f v is defined through a specified set of observation times t j [0,T], j = 1, ,J v and
∈ ···
ηj.
i.i.d.additive noise v The probability measure on can be defined, for example, by a Gaussian
W
random field and the additive noises η ,η can also be chosen as Gaussian. In this Gaussian setting
u v
u = Rd×Ju given by (1a) is also Gaussian; and it is correlated to non-Gaussian variable
| ∈   | U Rd×Ju |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- |
v = given by (1b). This creates a joint distribution µ(u,v) on the space with
| ∈           | V           |     |            |     |     | U ×V |     |
| ----------- | ----------- | --- | ---------- | --- | --- | ---- | --- |
| non-trivial | conditional |     | structure. |     |     |      |     |
This set-up can be generalized in a number of ways, including observing multiple Lagrangian
trajectories, making the noise structure more complex, for example by adding Brownian noise to the
equations for trajectory z, and by considering time-dependent velocity fields. Indeed, in Subsection
6.3, we will consider an example in which the velocity field is time-dependent and f encodes linear
u
| functionals, | different     |     | from pointwise | evaluations, | of the | velocity fields. |     |
| ------------ | ------------- | --- | -------------- | ------------ | ------ | ---------------- | --- |
| 1.3.         | Contributions |     | and Paper      | Overview     |        |                  |     |
We focus on the bimodal setting in this paper, but the reader will readily generalize to multimodal
settings with three or more modalities. We make the following contributions to our overarching
goal, namely the formulation of contrastive learning in a mathematical framework, shedding light on
existing algorithms, enabling analysis of the algorithm in the Gaussian setting, and suggesting novel
| variants | of those | algorithms: |     |     |     |     |     |
| -------- | -------- | ----------- | --- | --- | --- | --- | --- |
(C1) We formulate contrastive learning as determination of an underlying joint distribution on
the two data modalities, defined through a change of measure, a tilting, from the product of
|     | the marginals |     | of each modality. |     |     |     |     |
| --- | ------------- | --- | ----------------- | --- | --- | --- | --- |
(C2) We introduce new probabilistic loss functions, including matching the joint, either conditional,
or the sum of two conditionals; standard contrastive learning corresponds to the last case.
(C3) We introduce new tiltings which subsume the standard contrastive learning case, based on
|     | cosine similarity, |     | as a special | case. |     |     |     |
| --- | ------------------ | --- | ------------ | ----- | --- | --- | --- |
(C4) We analyze the resulting classes of new contrastive learning methodologies in the Gaussian
setting, formulating latent space identification in terms of low-rank approximation; we shed
light on the capabilities of different approaches in terms of their ability to define point
|     | estimators | and | as generative | models. |     |     |     |
| --- | ---------- | --- | ------------- | ------- | --- | --- | --- |

Baptista, Stuart, Tran/Contrastive Learning 4
(C5) We demonstrate that the basic contrastive learning methodology can be applied to problems
arisinginscienceandengineering,suchasLagrangiandataassimilation;andweshowthatthe
generalized contrastive learning methodology applies to data science applications including
retrieval and MNIST digit classification.
In Section 2 we define contrastive learning in the bimodal setting, and consider the limit of infinite
data, addressing contribution (C1). Section 3 adopts a probabilistic interpretation of the bimodal
contrastive learning problem; we introduce generalizations of the standard problem, both through
the form of tilting and the form of loss (i.e., objective) function defining the learning problem,
addressing contributions (C2) and (C3). Section 4 formulates retrieval and classification using the
framework developed in the two preceding sections, laying the groundwork for later numerical
experiments, illustrating parts of contribution (C5). In Section 5 we analyze this generalized class of
contrastive learning problems in the Gaussian setting, shedding light on the capabilities of different
approaches—contribution (C4). The supplementary meterial in Section 6 is devoted to contribution
(C5): we provide numerical experiments which: (i) validate the Gaussian theory developed in this
paper; (ii) which showcase, in the context of image-text crossmodal problems arising from the MNIST
dataset, the generalized methodology developed in this paper; (iii) show that the Lagrangian data
assimilation problem can be solved in a purely data-driven fashion using the cross-modal approaches
studied in this paper.
1.4. Literature Review
Contrastive learning is often referred to as self-supervised learning, distinguishing it from supervised
learning [1] and from unsupervised learning [34]. Contrastive self-supervised learning approaches have
enabled the comparison of data modalities without explicitly labeled data, but using only pairs of
related samples. In the context of language-image understanding, [18, 28, 29] showed that contrastive
learning approaches can learn representations for visual and text data obtained from the internet.
Moreover, the pretrained representations are immediately useful for downstream tasks, such as image
retrieval and classification, without additional model training. Recently, these pretrained models
have been applied directly, without task-specific training data, for various applications, including
embedding text prompts for image generation in diffusion models [30] and to embed text and/or
image inputs in multimodal language models [23]. Beyond their use in language, contrastive learning
is applied in computer vision tasks based on other modalities including the alignment of text and
video [11, 24], text and audio [10], and 3D scene generation [35]. It has also proven useful for image
classification, object detection [19], or semantic segmentation [20]. Scaling studies have investigated
the generalization behavior of contrastive models with increasing numbers of parameters [7]. While
contrastive learning aims to learn low-dimensional encodings, these can also be used to learn a
mapping from the embedding space back to the original data. One such approach was considered
in [37] to do zero-shot image-captioning using CLIP.
Several objectives for contrastive representation learning have been proposed to match related
samples, often referred to as positive samples, and to bring unrelated samples, often referred to as
negative samples, farther apart in the latent space. These include objectives based on the Euclidean
norm between the mapped embeddings [8], or a metric between the sample covariance of sample
pairs [39]. Several modifications have been proposed to the objective function in CLIP to improve
computational efficiency: one such is SigClip [40], which replaces the softmax with a sigmoid layer
and doesn’t need two passes over the data to evaluate the loss function; another is ClipLITE that
only takes one pass over the training data to compute negative alignments. When supervised datasets
are not available, data augmentation approaches generate multiple representations of the same image.

Baptista, Stuart, Tran/Contrastive Learning 5
For example Barlow twins [39], ‘Bootstrap Your Own Latent’ [15], and SimCLR uses distorted
representations of the same image. Rather than starting from a joint distribution of data from two
modalities, self-supervised learning starts an unlabeled dataset from one distribution and does data
augmentation to generate new samples. Recently, other generalizations of the loss for contrastive
learning include learning the tilting from the recovery of a cost matrix using inverse optimal
transport [31]. Historically, InfoNCE [36] also showed how to identify embeddings by maximizing
mutual information (MI), which upper bounds the contrastive loss underlying CLIP. However, [33]
showed that maximizing MI directly is not correlated with downstream predictive accuracy when
the learned embeddings are used for classification. The paper [38] also used a new alignment metric
motivated by hyperbolic geometry that allows encoders to map into a normalized sphere, rather
than only on the surface.
From a theoretical perspective, previous analysis has studied the optimizers for contrastive self-
supervisedlearningproblems.Inparticular,[41]studiedlosslandscapesandshowedinlinear-Gaussian
settings that the learned representations may collapse to only capture low-dimensional subspaces.
Later, [5] showed that it is necessary to use large batch sizes to learn useful representations and
prevent collapse. Recent work has shown that the choice of contrastive loss is crucial to optimize both
alignment and uniformity of the resulting distribution of normalized features on the hypersphere. In
terms of generalization, [16] proved that contrastive and self-supervised spectral losses can lead to
generalization guarantees. Concerning statistical guarantees, [4] considers the bias in self-supervised
learning from small mini-batches and proposes an augmented variable technique that is unbiased.
In this work our analysis is focused on the solution of contrastive learning in the Gaussian setting.
Here, the standard contrastive learning methodology corresponds to matching conditionals of a joint
multivariate Gaussian distribution. In the context of Bayesian inverse problems, these Gaussian
conditional approximations have been studied in [32, 14].
1.5. Notation
The formulation, analysis and numerical experiments in this paper are grounded in the metric space
of probability measures on product space . We will denote by µ the measure on this space
U ⊗V
from which paired data points, the basis of training, are derived; and we will denote by ν( ;θ) the
·
family of probability models, on the same space, from which a learned approximation of µ is to be
found. We will denote the u marginal and u v conditional measures of µ as µ and µ , respectively,
u u|v
|
with analogous notation when the roles of u and v are swapped, and when applied to measure ν
rather than µ. We will also consider various objective functions, constructed to measure closeness
of ν, parameterized by θ, to µ, namely J , J , J and J . Here D refers to a chosen
cond cond,D joint joint,D
divergence; when D is not specified, it is considered to be the Kullback-Leibler (KL) divergence.
Subtracting constants independent of θ from these objectives, and in two cases making the specific
choice of divergence D to be the maximum mean discrepancy rather than the KL divergence, leads
to loss functions L , L , L and L . The preceding objectives and loss functions
cond cond,mmd joint joint,mmd
are all defined in the population limit (infinite data). Our starting point, however, is the standard
contrastive loss function LN , where N denotes the size of the empirical data set. This is related,
clip
by a θ independent, but N dependent, constant to LN . And LN has a well-defined continuum
cond cond
− −
limit, leading to L and thence to J and the zoo of population level objectives and losses listed
cond cond
above. Finally we will discuss fine-tuning of classification methods, leading to L and LK .
fine fine
We let and , denote the standard Euclidean norm and inner-product on Rne. Given
| · | ⟨· ·⟩
positive definite matrix A, we define weighted version of the standard Euclidean inner-product by
w,w′ = w,A−1w′ ; the induced weighted Euclidean norm is denoted by . We will denote
A A
⟨ ⟩ ⟨ ⟩ |·|
by A the optimal rank-r approximation of the matrix A with respect to the Frobenius-norm—the
r

|     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     |     | Learning |     |     |     | 6   |
| --- | --- | --- | --- | --------- | ------- | ---------------- | --- | --- | -------- | --- | --- | --- | --- |
low-rank truncation of the singular value decomposition of A. Lastly, we highlight the previously
| deployed | notation    | [[1,N]]  | :=  | 1, ,N | .   |     |     |     |     |     |     |     |     |
| -------- | ----------- | -------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|          |             |          |     | { ··· | }   |     |     |     |     |     |     |     |     |
| 2.       | Contrastive | Learning |     |       |     |     |     |     |     |     |     |     |     |
Contrastive methodologies learn to associate two random variables u ,v , living in typically
|     |     |     |     |     |     |     |     |     |     | ∈   | U ∈ | V   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
different spaces , . We assume that both spaces are equipped with a sigma-algebra, allowing
U V
probability measures to be defined on them and on their product space. The methodology proceeds
by learning two encoders g : Rne and g : Rne that map to a common latent space of
|     |     |     | u   |     |     | v   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | U → |     |     | V → |     |     |     |     |     |     |
user-specified dimension n ; typically n is much smaller than the size of and . The training
|     |     |     | e   |     | e   |     |     |     |     |     | U   | V   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
of the encoders is governed by minimizing a contrastive learning objective that rewards data pairs
defined by the joint distribution of the two random variables and penalizes unrelated data pairs.
In standard implementations of contrastive learning, the outputs of the encoders are normalized
L2
by the norm of their output so that only the alignment of the outputs is compared, rather than
their magnitude. Thus, the encoders used to align the two modalities are
|     |     |     |     |        | g u | (u) |     |       | g v (v) |     |     |     |     |
| --- | --- | --- | --- | ------ | --- | --- | --- | ----- | ------- | --- | --- | --- | --- |
|     |     |     |     | g¯ (u) | =   | ,   | g¯  | (v) = |         | ,   |     |     | (2) |
|     |     |     |     | u      |     |     | v   |       |         |     |     |     |     |
|     |     |     |     |        | g u | (u) |     |       | g v (v) |     |     |     |     |
|     |     |     |     |        | |   | |   |     |       | |       | |   |     |     |     |
where denotes the Euclidean norm on Rne so that g¯ : Sne−1 and g¯ : Sne−1. For
|     |       |     |     |     |     |     |     | u   |     |     |     | v   |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | | · | |     |     |     |     |     |     |     | U → |     |     | V → |     |
example, the encoders may be constructed using a text or vision transformer for text and image data,
respectively. Moreover, they may involve pretrained models that are composed with projections (e.g.,
linear maps), which are then learned so as to embed their outputs into a space of the same common
dimension where they are aligned. We make the following standard assumption about the data used
| to  | train encoders | that | reflect | the desired | alignment. |     |     |     |         |     |     |     |     |
| --- | -------------- | ---- | ------- | ----------- | ---------- | --- | --- | --- | ------- | --- | --- | --- | --- |
|     |                |      |         |             |            |     |     |     | (ui,vi) | N   |     |     |     |
Data Assumption 2.1. The available data comprises pairs drawn i.i.d.from a joint
|              |     |           |      |     |     |     |     |     | {   | }i=1 |     |     |     |
| ------------ | --- | --------- | ---- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- |
| distribution |     | µ(u,v) on |      | .   |     |     |     |     |     |      |     |     |     |
|              |     |           | U ×V |     |     |     |     |     |     |      |     |     |     |
The aim of contrastive learning is to maximize the alignment between related data pairs, in
the common latent space, and to minimize the alignment between unrelated pairs. The related
pairs are draws from the joint distribution µ(u,v); unrelated pairs may be considered to be drawn
independently from the two marginal distributions for u and v, which we denote by µ (u) and µ (v),
|     |     |     |     |     |     |     |     |     |     |     |     | u   | v   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
respectively. Given N samples from the joint distribution, as in Data Assumption 2.1, we have the
| following | empirical   | measures |     | for the  | joint and | marginal |            | distributions |      |     |          |     |     |
| --------- | ----------- | -------- | --- | -------- | --------- | -------- | ---------- | ------------- | ---- | --- | -------- | --- | --- |
|           |             |          |     | N        |           |          | N          |               |      |     | N        |     |     |
|           |             |          | 1   | (cid:88) |           |          | 1 (cid:88) |               |      | 1   | (cid:88) |     |     |
|           |             | µN       | :=  | δ        | ,         | µN :=    |            | δ ,           | µN   | :=  | δ        | .   | (3) |
|           |             |          |     | (uℓ,vℓ)  |           |          |            | uℓ            |      |     | vℓ       |     |     |
|           |             |          | N   |          |           | u        | N          |               | v    | N   |          |     |     |
|           |             |          |     | ℓ=1      |           |          | ℓ=1        |               |      |     | ℓ=1      |     |     |
| We        | also define |          |     |          |           |          |            |               |      |     |          |     |     |
|           |             |          |     | N :=     | ui N      | ,        | N          | := vi         | N    | .   |          |     | (4) |
|           |             |          |     |          | }i=1      |          |            |               | }i=1 |     |          |     |     |
|           |             |          | U   | {        |           | ⊂ U      | V          | {             |      | ⊂ V |          |     |     |
In Subsection 2.1 we describe the standard contrastive learning objective defined by the bimodal
datagiveninDataAssumption2.1.Subsection2.2describesapopulationlearningobjective,obtained
| in   | the limit | N .      |         |     |     |     |     |     |     |     |     |     |     |
| ---- | --------- | -------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|      |           | → ∞      |         |     |     |     |     |     |     |     |     |     |     |
| 2.1. | Discrete  | Learning | Problem |     |     |     |     |     |     |     |     |     |     |
Rp
Having chosen an embedding dimension n e , we introduce parameter θ = (θ u ,θ v ) defining
∈
the parameterized encoder models g¯ ( ;θ ) and g¯ ( ;θ ), respectively. We define the conditional
|     |     |     |     |     | u u |     | v   | v   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     | ·   |     | ·   |     |     |     |     |     |     |

|             |     |      |     |          | Baptista, | Stuart, | Tran/Contrastive |     |     | Learning |     |     | 7   |
| ----------- | --- | ---- | --- | -------- | --------- | ------- | ---------------- | --- | --- | -------- | --- | --- | --- |
| probability | of  | data | u   | N, given |           | vi      | N, by            |     |     |          |     |     |     |
|             |     |      | ∈   | U        |           | ∈ V     |                  |     |     |          |     |     |     |
(cid:0)
|     |     |     |     |     |       |           | exp g¯ | (u;θ     | ),g¯ (vi;θ | )     | /τ)     |     |      |
| --- | --- | --- | --- | --- | ----- | --------- | ------ | -------- | ---------- | ----- | ------- | --- | ---- |
|     |     |     |     |     | vi;θ) |           |        | u        | u v        | v     |         |     |      |
|     |     |     |     | p(u |       | :=        | ⟨      |          |            |       | ⟩ ,     |     | (5a) |
|     |     |     |     | |   |       | (cid:80)N |        | (cid:0)  |            |       | (cid:1) |     |      |
|     |     |     |     |     |       |           | exp    | g¯ (uj;θ | ),g¯       | (vi;θ | ) /τ    |     |      |
|     |     |     |     |     |       | j=1       |        | ⟨ u      | u          | v     | v ⟩     |     |      |
where τ > 0 is a hyperparameter, often referred to as a temperature. By symmetry, we may also
| define | the conditional |     | probability |     | of  | data | v   | N, given | uk  | N,  | by  |     |     |
| ------ | --------------- | --- | ----------- | --- | --- | ---- | --- | -------- | --- | --- | --- | --- | --- |
|        |                 |     |             |     |     |      | ∈ V |          |     | ∈ U |     |     |     |
(cid:0)
|     |     |     |     |       |     |           | exp g¯ | (uk;θ    | ),g¯ | (v;θ  | ) /τ)   |     |      |
| --- | --- | --- | --- | ----- | --- | --------- | ------ | -------- | ---- | ----- | ------- | --- | ---- |
|     |     |     |     | uk;θ) |     |           |        | u        | u v  | v     |         |     |      |
|     |     |     |     | p(v   |     | :=        | ⟨      |          |      |       | ⟩ ,     |     | (6a) |
|     |     |     |     | |     |     | (cid:80)N |        | (cid:0)  |      |       | (cid:1) |     |      |
|     |     |     |     |       |     |           | exp    | g¯ (uk;θ | ),g¯ | (vj;θ | ) /τ    |     |      |
|     |     |     |     |       |     | j=1       |        | ⟨ u      | u    | v     | v ⟩     |     |      |
Remark 2.2. The specific construction of probabilities using exponentiation and normalization is
| often referred |     | to as | the | softmax | operation. |     |     |     |     |     |     |     |     |
| -------------- | --- | ----- | --- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
⋄
vi;θ)
Remark 2.3. We note that Np( may also be viewed as a density with respect to the discrete
·|
measure µN on N: reweighting µN by Np( vi;θ) gives another probability measure on N. Likewise
|     | u   | U   |     |     | u   |     | ·|  |     |     |     |     | U   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Np( uk;θ) may be viewed as a density with respect to the discrete measure µN on N.
v
| ·|  |     |     |     |     |     |     |     |     |     |     |     | V   | ⋄   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
The idea behind contrastive learning is to choose parameter θ so that, when summed over the
|     |     | p(ui | vi;θ) |     | p(vi ui;θ) |     |     |     |     |     |     |     |     |
| --- | --- | ---- | ----- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
data set, both and are maximized. The log of the conditional probabilities, also
|     |     |     | |   |     | |   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
known as logits, are commonly used to define an objective function for contrastive learning that
maximizes the likelihood of the paired data (ui,vi) N , defined by Data Assumption 2.1, under the
}i=1
{
models for the conditional probabilities of u v and v u defined by (5) and (6). This line of reasoning
|       |        |               |     |      |     |          | |   | |   |     |     |          |     |     |
| ----- | ------ | ------------- | --- | ---- | --- | -------- | --- | --- | --- | --- | -------- | --- | --- |
| leads | to the | cross-entropy |     | loss |     |          |     |     |     |     |          |     |     |
|       |        |               |     |      |     | (cid:34) |     |     |     |     | (cid:35) |     |     |
N
|     |     |     |     |      |     | 1 1 | (cid:88) |     |               |     |         |     |     |
| --- | --- | --- | --- | ---- | --- | --- | -------- | --- | ------------- | --- | ------- | --- | --- |
|     |     |     | LN  | (θ)  | :=  |     | logp(ui  |     | vi;θ)+logp(vi |     | ui;θ) . |     | (7) |
|     |     |     |     | clip | −2  | N   |          |     |               |     |         |     |     |
|     |     |     |     |      |     |     |          |     | |             |     | |       |     |     |
i=1
The optimal parameter θN minimizes this loss. Rewriting in terms of expectations we have the
clip
| following    | optimization |         | problem: |      |        |          |           |        |             |     |           |     |      |
| ------------ | ------------ | ------- | -------- | ---- | ------ | -------- | --------- | ------ | ----------- | --- | --------- | --- | ---- |
| Optimization |              | Problem |          | 2.4. |        |          |           |        |             |     |           |     |      |
|              |              |         |          |      |        | 1        | (cid:104) |        |             |     | (cid:105) |     |      |
|              |              |         |          | LN   |        | E        |           |        |             |     |           |     |      |
|              |              |         |          | (θ)  | =      | (u,v)∼µN |           | logp(u | v;θ)+logp(v |     | u;θ) ,    |     | (8a) |
|              |              |         |          | clip | −2     |          |           |        | |           |     | |         |     |      |
|              |              |         |          | θN   | argmin |          | LN (θ).   |        |             |     |           |     | (8b) |
|              |              |         |          | clip |        |          | clip      |        |             |     |           |     |      |
∈ θ∈Rp
We now make several remarks which help in the interpretation of the contrastive learning
optimization problem (8) for θ. We first note that the definition of the conditional probabilities uses
cosine similarity:
Remark 2.5. For normalized encoders, i.e., g¯ (u) = g¯ (v) = 1, the inner product satisfies
|     |     |     |           |     |       |        |        | u     | v   |       |           |     |     |
| --- | --- | --- | --------- | --- | ----- | ------ | ------ | ----- | --- | ----- | --------- | --- | --- |
|     |     |     |           |     |       |        | |      | |     | |   | |     |           |     |     |
|     |     |     | g¯ (u),g¯ |     | (v) = | g¯ (u) | g¯ (v) | cos(α | ) = | cos(α | ) [ 1,1], |     |     |
|     |     |     | u         | v   |       | u      | v      |       | uv  |       | uv        |     |     |
|     |     |     | ⟨         |     | ⟩     | |      | ||     | |     |     |       | ∈ −       |     |     |
where α is the angle between the embedding vectors. Hence, the inner product is known as a cosine
uv
similarity. The cosine similarity is invariant to rescaling of g and g since this does not affect
|     |     |     |     |     |     |     |     |     |     | u   | v   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:0) g¯ (u),g¯ (v) (cid:1) . In addition, the cosine similarity is invariant to joint rotations of the two encoders:
| u       | v           |     |        |     |        |     |            |      |     |     |     |     |     |
| ------- | ----------- | --- | ------ | --- | ------ | --- | ---------- | ---- | --- | --- | --- | --- | --- |
| for any | orthonormal |     | matrix | U   | Rne×ne |     | it follows | that |     |     |     |     |     |
∈
(u),U⊤Ug¯
|     |     |     | Ug¯ | (u),Ug¯ |     | (v) = | g¯  |     | (v) | = g¯ | (u),g¯ (v) . |     |     |
| --- | --- | --- | --- | ------- | --- | ----- | --- | --- | --- | ---- | ------------ | --- | --- |
|     |     |     | ⟨   | u       | v   | ⟩     | ⟨ u |     | v   | ⟩ ⟨  | u v ⟩        |     |     |
⋄

|     |     |     |     | Baptista, |     | Stuart, | Tran/Contrastive |     | Learning |     |     |     | 8   |
| --- | --- | --- | --- | --------- | --- | ------- | ---------------- | --- | -------- | --- | --- | --- | --- |
Wehavementionedpreviouslythatcontrastivelearninghastheinterpretationofchoosingencoders
that reward similarity for paired data points from the two modalities and penalizes unpaired data.
| The following | remark |     | makes | this | explicit. |     |     |     |     |     |     |     |     |
| ------------- | ------ | --- | ----- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
Remark 2.6. Recall that, from Data Assumption 2.1, (ui,vi) N denote i.i.d. samples from µ.
}i=1
{
Now consider the unpaired data set (ui,vj) N ; this may be viewed as constituting i.i.d.
}i=1,j=1,i̸=j
{
| samples | from µ | u µ | v . Now | observe | that, | using | (5b) | and | (6b), |     |     |     |     |
| ------- | ------ | --- | ------- | ------- | ----- | ----- | ---- | --- | ----- | --- | --- | --- | --- |
⊗
|     |     |      |     | (cid:34) N |         |               |     |     |       | (cid:35) |     |     |     |
| --- | --- | ---- | --- | ---------- | ------- | ------------- | --- | --- | ----- | -------- | --- | --- | --- |
|     |     |      | 1   | 1 (cid:88) |         |               |     |     |       |          |     |     |     |
|     | LN  |      |     |            | logp(ui | vi;θ)+logp(vi |     |     | ui;θ) |          |     |     |     |
|     |     | (θ)  | =   |            |         |               |     |     |       |          |     |     |     |
|     | −   | clip | 2   | N          |         | |             |     |     | |     |          |     |     |     |
i=1
N
1 (cid:88)
|     |     |     | =   | g¯  | (ui;θ | ),g¯ | (vi;θ | ) /τ | log(N) |     |     |     |     |
| --- | --- | --- | --- | --- | ----- | ---- | ----- | ---- | ------ | --- | --- | --- | --- |
|     |     |     |     |     | u     | u y  | y     |      |        |     |     |     |     |
|     |     |     | N   | ⟨   |       |      |       | ⟩ −  |        |     |     |     |     |
i=1
|     |     |     |     |      |          |     |          |          |          |            |            |         |     |
| --- | --- | --- | --- | ---- | -------- | ---- | -------- | -------- | -------- | ---------- | ---------- | -------- | --- |
|     |     |     |     |      | N        |      | N        |          |          |            |            |          |     |
|     |     |     |     | 1    | (cid:88) | 1    | (cid:88) | (cid:16) |          |            |            | (cid:17) |     |
|     |     |     |     |      | log     |      | exp      | g¯       | (uj;θ    | ),g¯ (vi;θ |            | ) /τ     |     |
|     |     |     |     |      |          |      |          | u        |          | u v        | v          |         |     |
|     |     |     |     | − 2N |          | N    |          | ⟨        |          |            |            | ⟩        |     |
|     |     |     |     |      | i=1      |      | j=1      |          |          |            |            |          |     |
|     |     |     |     |      |          |      |         |          |          |            |            |         |     |
|     |     |     |     |      | 1        | N    | 1        | N        | (cid:16) |            |            | (cid:17) |     |
|     |     |     |     |      | (cid:88) |      |          | (cid:88) |          |            |            |          |     |
|     |     |     |     |      |          | log |          | exp      | g¯ (ui;θ |            | ),g¯ (vj;θ | ) /τ .  |     |
|     |     |     |     |      | 2N       |      | N        |          | u        | u          | v          | v        |     |
|     |     |     |     | −    |          |      |          |          | ⟨        |            |            | ⟩        |     |
|     |     |     |     |      | i=1      |      |          | j=1      |          |            |            |          |     |
LN
The Optimization Problem 2.4 is solved by maximizing ( ). This is achieved by maximizing
|     |     |     |     |     |     |     |     |     | − clip | ·   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- |
the cosine similarities of paired data in the first (single) sum over i alone; and by simultaneously
minimizing the cosine similarities of unpaired data in the second and third (double) sums, over i and
j together. This is the process we have been refering to, and will refer to, as alignment of the two
modalities.
⋄
In the next section we discuss a population level version of the loss function. The following
observation will help interpret this development. To this end it is helpful to recall Remark 2.3.
| 2.2. Population |     | Level |     | Learning | Problem |     |     |     |     |     |     |     |     |
| --------------- | --- | ----- | --- | -------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
The appropriate analogs of (5) and (6) in the population limit are given by
|     |     |     |     |      |     |       | (cid:0)   |         |            |        | (cid:1)   |     |      |
| --- | --- | --- | --- | ---- | --- | ----- | --------- | ------- | ---------- | ------ | --------- | --- | ---- |
|     |     |     |     |      |     | exp   | g¯ (u;θ   | ),g¯    | (v;θ       | ) /τ   |           |     |      |
|     |     |     |     |      |     |       | u         | u       | v          | v      |           |     |      |
|     |     |     | ρ(u | v;θ) | =   |       | ⟨         |         |            | ⟩      | ,         |     | (9a) |
|     |     |     |     |      | E   |       | (cid:0)   | (u′;θ   |            |        | (cid:1)   |     |      |
|     |     |     |     | |    |     | u′∼µu | exp       | g¯ u    | u ),g¯ v   | (v;θ y | ) /τ      |     |      |
|     |     |     |     |      |     |       | ⟨         |         |            |        | ⟩         |     |      |
|     |     |     |     |      |     |       | (cid:0)   |         |            |        | (cid:1)   |     |      |
|     |     |     |     |      |     | exp   | g¯ (u;θ   | ),g¯    | (v;θ       | ) /τ   |           |     |      |
|     |     |     |     |      |     |       | u         | u       | v          | v      |           |     |      |
|     |     |     | ρ(v | u;θ) | =   |       | ⟨ (cid:0) |         |            | ⟩      | (cid:1) . |     | (9b) |
|     |     |     |     | |    | E   |       | exp       | g¯ (u;θ | ),g¯ (v′;θ | )      | /τ        |     |      |
|     |     |     |     |      |     | v′∼µv |           | u       | u v        | y      |           |     |      |
|     |     |     |     |      |     |       | ⟨         |         |            |        | ⟩         |     |      |
Here (9a) defines a density with respect to measure µ on and (9b) defines a density with respect
u
U
to measure µ on . Thus, ρ(u v;θ)µ (du) defines a probability measure on and ρ(v u;θ)µ (dv)
|     | v   | V   |     | |   | u   |     |     |     |     |     |     | U | | u   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
defines a probability measure on . The appropriate analog of Optimization Problem 2.4 is then
V
given by:
| Optimization |     | Problem |     | 2.7. |     |     |           |     |     |     |     |           |     |
| ------------ | --- | ------- | --- | ---- | --- | --- | --------- | --- | --- | --- | --- | --------- | --- |
|              |     |         |     |      | 1   |     | (cid:104) |     |     |     |     | (cid:105) |     |
E
|     |     |     | L    | (θ) = |        |         | logρ(u | v;θ)+logρ(v |     |     | u;θ) | ,   |     |
| --- | --- | --- | ---- | ----- | ------ | ------- | ------ | ----------- | --- | --- | ---- | --- | --- |
|     |     |     | cond |       | −2     | (u,v)∼µ |        | |           |     |     | |    |     |     |
|     |     |     |      | θ     | argmin | L       | (θ).   |             |     |     |      |     |     |
|     |     |     |      | cond  |        |         | cond   |             |     |     |      |     |     |
|     |     |     |      | ∈     | θ∈Rp   |         |        |             |     |     |      |     |     |

|     |     |     |     | Baptista, |     | Stuart, | Tran/Contrastive |     | Learning |     |     |     | 9   |
| --- | --- | --- | --- | --------- | --- | ------- | ---------------- | --- | -------- | --- | --- | --- | --- |
Following the line of reasoning in Remark 2.6, the objective function L (θ) may be written as
cond
|     |     |      |       |         | (cid:104) |      |           |           | (cid:105) |             |        |           |      |
| --- | --- | ---- | ----- | ------- | --------- | ---- | --------- | --------- | --------- | ----------- | ------ | --------- | ---- |
|     |     | L    | (θ) = | E       | g¯        | (u;θ | ),g¯ (v;θ | )         | /τ        |             |        |           |      |
|     |     | cond |       | (u,v)∼µ |           | u    | u v       | v         |           |             |        |           |      |
|     | −   |      |       |         | ⟨         |      |           | ⟩         |           |             |        |           |      |
|     |     |      |       | 1       |           |      | (cid:104) |           |           |             |        | (cid:105) |      |
|     |     |      |       |         | E         | logE |           | exp(      | g¯ (u′;θ  | ),g¯        | (v;θ ) | /τ)       | (11) |
|     |     |      |       |         | v∼µv      |      | u′∼µu     |           | u         | u v         | v      |           |      |
|     |     |      |       | − 2     |           |      |           | ⟨         |           |             |        | ⟩         |      |
|     |     |      |       |         | 1         |      |           | (cid:104) |           |             |        | (cid:105) |      |
|     |     |      |       |         | E         |      | logE      |           |           |             | (v′;θ  |           |      |
|     |     |      |       |         |           | u∼µu | v′∼µv     | exp(      | g¯ u      | (u;θ u ),g¯ | v      | v ) /τ)   |      |
|     |     |      |       |         | − 2       |      |           |           | ⟨         |             |        | ⟩         |      |
Replacing the expectations under µ,µ and µ in L by their empirical versions (3) results in the
|           |           |      |          |     | u   |     | v   | cond |     |     |     |     |     |
| --------- | --------- | ---- | -------- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
| following | empirical | risk | function |     |     |     |     |      |     |     |     |     |     |
N
|     |     |      | 1   | (cid:88) |       |      |       |      |     |     |     |     |     |
| --- | --- | ---- | --- | -------- | ----- | ---- | ----- | ---- | --- | --- | --- | --- | --- |
|     | LN  |      |     |          | (ui;θ |      | (vi;θ |      |     |     |     |     |     |
|     |     | (θ)  | =   |          | g¯    | ),g¯ |       | ) /τ |     |     |     |     |     |
|     | −   | cond | N   | ⟨        | u     | u    | v     | v ⟩  |     |     |     |     |     |
i=1
|               |     |          |             |      |          |          |          |          |          |       |       |        |      |
| ------------- | --- | -------- | ----------- | ---- | -------- | --------- | -------- | -------- | -------- | ----- | ----- | ------- | ---- |
|               |     |          |             | 1    | N        |           | 1 N      |          |          |       |       |         |      |
|               |     |          |             |      | (cid:88) |           | (cid:88) |          | (uj;θ    | (vi;θ |       |         |      |
|               |     |          |             |      | log     |           | exp(     | g¯       |          | ),g¯  | )     | /τ)    |      |
|               |     |          |             | − 2N |          | N         |          | ⟨        | u        | u v   | v     | ⟩       |      |
|               |     |          |             |      | i=1      |           | j=1      |          |          |       |       |         |      |
|               |     |          |             |      |          |           |         |          |          |       |       |        |      |
|               |     |          |             |      |          | N         |          | N        |          |       |       |         |      |
|               |     |          |             |      | 1        | (cid:88)  | 1        | (cid:88) |          |       |       |         |      |
|               |     |          |             |      |          | log      |          | exp(     | g¯ (ui;θ | ),g¯  | (vj;θ | ) /τ). | (12) |
|               |     |          |             |      |          |           |          |          | u        | u     | v     | v       |      |
|               |     |          |             | −    | 2N       |           | N        |          | ⟨        |       |       | ⟩       |      |
|               |     |          |             |      |          | i=1       |          | j=1      |          |       |       |         |      |
| The following | is  | a direct | consequence |      |          | of Remark | 2.6:     |          |          |       |       |         |      |
|               |     |          |             |      |          | LN        |          | LN       |          |       |       |         |      |
Theorem 2.8. The objective functions ( ) and ( ) differ by log(N) :
|     |     |     |     |     |      | clip  | ·    | cond        | ·   |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | ----- | ---- | ----------- | --- | --- | --- | --- | --- |
|     |     |     |     |     | LN   |       | LN   |             |     |     |     |     |     |
|     |     |     |     |     |      | (θ) = |      | (θ)+log(N). |     |     |     |     |     |
|     |     |     |     |     | clip |       | cond |             |     |     |     |     |     |
If θN minimizes LN (θ) and θN solves Optimization Problem 2.4, then θN = θN .
| cond |     | cond |     |     | clip |     |     |     |     |     |     | cond clip |     |
| ---- | --- | ---- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --------- | --- |
In what follows we work with L ( ), and hence its empirical counterpart LN ( ). The latter is
|     |     |     |     |     | cond |     |     |     |     |     |     | cond |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | ---- | --- |
|     |     |     |     |     |      | ·   |     |     |     |     |     | ·    |     |
well-defined in the population loss limit; working in the population loss limit clarifies understanding
and is adopted throughout the remainder of the paper. The presence of the constant log(N) in
LN
Theorem 2.8, which leads to a divergence in the population loss limit of ( ), relates to the fact
|     |     |     |     |     |     |     |     |     |     |     |     | clip · |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
LN
that the loss ( ) is not defined via densities with respect to a reference probability measure, but
clip ·
via probabilities. Working with densities with respect to a reference probability measure enables
seamless passage between population and empirical representations of the problem.
| 3. Probabilistic |     | Perspective |     |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
In Subsection 3.1 we formulate contrastive learning in terms of a minimization problem over
probability measures on the joint space . This suggests several natural generalizations of
|     |     |     |     |     |     | U   | ⊗ V |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
contrastive learning; the first class of such generalizations follow from using different probabilistic
loss functions for the joint distribution and the second class from considering different measures of
alignment of the two data modalities in the latent space. These two classes of generalizations are
considered in Subsections 3.2 and 3.3, respectively. Although we work at the population level, it is
important that we are constrained by formulations which can be deployed in the empirical setting by
replacing the measures µ,µ and µ with their empirical counterparts given in (3). We will discuss
|                  |     |            |     | u          | v     |     |           |      |        |     |     |     |     |
| ---------------- | --- | ---------- | --- | ---------- | ----- | --- | --------- | ---- | ------ | --- | --- | --- | --- |
| empiricalization |     | explicitly |     | in several | cases | to  | highlight | this | issue. |     |     |     |     |

|                  |     |          |     | Baptista, |       | Stuart, | Tran/Contrastive |       | Learning     |     |     |     | 10  |
| ---------------- | --- | -------- | --- | --------- | ----- | ------- | ---------------- | ----- | ------------ | --- | --- | --- | --- |
| 3.1. Contrastive |     | Learning |     | in        | Terms | of      | the              | Joint | Distribution |     |     |     |     |
In this subsection, we relate the solution identified by contrastive learning to a joint distribution
that is learned over the product space . Let ν(du,dv;θ) be a probability measure for the joint
|                 |     |       |     |         |     | U ×V |     |     |     |     |     |     |     |
| --------------- | --- | ----- | --- | ------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
| random variable |     | (u,v) |     | defined |     | by   |     |     |     |     |     |     |     |
∈ U ×V
|     |     |     |     | ν(du,dv;θ) |     | =   | ρ(u,v;θ)µ |     | (du)µ | (dv), |     |     | (13) |
| --- | --- | --- | --- | ---------- | --- | --- | --------- | --- | ----- | ----- | --- | --- | ---- |
|     |     |     |     |            |     |     |           |     | u     | v     |     |     |      |
where
|     |     |          |     |     | 1   | (cid:16) |      |      | (cid:17) |     |     |     |       |
| --- | --- | -------- | --- | --- | --- | -------- | ---- | ---- | -------- | --- | --- | --- | ----- |
|     |     | ρ(u,v;θ) |     | =   | exp | g¯ (u;θ  | ),g¯ | (v;θ | ) /τ     | ,   |     |     | (14a) |
|     |     |          |     |     |     | u        | u    | v    | v        |     |     |     |       |
|     |     |          |     | Z   |     | ⟨        |      |      | ⟩        |     |     |     |       |
(cid:90)
|     |     |     |     | Z = | exp( | g¯  | (u;θ | ),g¯ (v;θ | ) /τ)µ | (du)µ (dv). |     |     | (14b) |
| --- | --- | --- | --- | --- | ---- | --- | ---- | --------- | ------ | ----------- | --- | --- | ----- |
|     |     |     |     |     |      | ⟨   | u    | u v       | v ⟩    | u v         |     |     |       |
U×V
We refer to the change of measure ρ as a tilting; this tilting links the product of marginals of the
data-generating distribution µ ,µ to a (θ parameterized) joint distribution ν = ν( ;θ). The joint
|     |     |     |     | u   | v   | −   |     |     |     |     | ·   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
distribution defines a coupling of u and v to encode dependence between the two random variables.
We show that the contrastive learning objective from the previous section can be reformulated
in terms of a loss function which aligns ν with µ. We refer to the specific choice of ρ here as an
| exponential | tilting. |     |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
In the following, we let ν (resp. ν ) denote the conditional measure for u v (resp. v u) under
|     |     |     |     | u|v |     | v|u |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| |
| the joint measure |         | ν in    | (15). | We       | then have |          |       |       |       |     |     |     |      |
| ----------------- | ------- | ------- | ----- | -------- | --------- | -------- | ----- | ----- | ----- | --- | --- | --- | ---- |
|                   |         |         |       |          | ν         | (du v;θ) | = ρ(u | v;θ)µ | (du), |     |     |     | (15) |
|                   |         |         |       |          | u|v       |          |       |       | u     |     |     |     |      |
|                   |         |         |       |          |           | |        |       | |     |       |     |     |     |      |
| where ρ(          | v;θ) is | defined | in    | equation | (9a);     | likewise |       |       |       |     |     |     |      |
·|
|     |     |     |     |     | ν   | (dv u;θ) | = ρ(v | u;θ)µ | (dv), |     |     |     | (16) |
| --- | --- | --- | --- | --- | --- | -------- | ----- | ----- | ----- | --- | --- | --- | ---- |
|     |     |     |     |     | v|u |          |       |       | v     |     |     |     |      |
|     |     |     |     |     |     | |        |       | |     |       |     |     |     |      |
where ρ( u;θ) is defined in equation (9b). We let µ (resp. µ ) denote the conditional measure
|     |     |     |     |     |     |     |     | u|v |     | v|u |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
·|
| for u v (resp. | v   | u) under | the | data | generating |     | measure | µ.  |     |     |     |     |     |
| -------------- | --- | -------- | --- | ---- | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- |
| |              | |   |          |     |      |            |     |         |     |     |     |     |     |     |
WiththisnotationwemaynowdefineanoptimizationproblemwhichisequivalenttoOptimization
Problem2.7.Itcaststheproblemasminimizationofasumofdistancesbetweenprobabilitymeasures;
in so doing it suggests paths for the generalization of contrastive learning.
| Optimization |     | Problem |         | 3.1. |     |     |     |         |     |         |     |         |     |
| ------------ | --- | ------- | ------- | ---- | --- | --- | --- | ------- | --- | ------- | --- | ------- | --- |
|              |     | 1       |         |      |     |     |     |         | 1   |         |     |         |     |
|              |     | E       | (cid:2) |      |     |     |     | (cid:3) | E   | (cid:2) |     | (cid:3) |     |
J cond (θ) = v∼µv D kl (µ ( v) ν ( v;θ)) + u∼µu D kl (µ ( u) ν ( u;θ)) ,
|     |     | 2   |     |     | u|v ·| | || u|v | ·|  |     | 2   | v|u ·| || | v|u ·| |     |     |
| --- | --- | --- | --- | --- | ------ | ------ | --- | --- | --- | --------- | ------ | --- | --- |
θ∗
|     |     | argmin | J   | (θ). |     |     |     |     |     |     |     |     |     |
| --- | --- | ------ | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | ∈      |     | cond |     |     |     |     |     |     |     |     |     |
θ∈Rp
Theorem 3.2. Let the conditionals of µ(du,dv) satisfy the finite relative entropy conditions:
|     |     |     |     |     | E    | D   | (µ ( | v) µ  | ) < |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | ---- | ----- | --- | --- | --- | --- | --- |
|     |     |     |     |     | v∼µv | kl  | u|v  |       | u   |     |     |     |     |
|     |     |     |     |     |      |     |      | ·| || | ∞   |     |     |     |     |
E
|     |     |     |     |     |      | D   | (µ ( | u) µ  | ) < | .   |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | ---- | ----- | --- | --- | --- | --- | --- |
|     |     |     |     |     | u∼µu | kl  | v|u  | ·| || | v ∞ |     |     |     |     |
Then, J (θ) = L (θ)+C where C depends only on the relative entropy of the µ conditionals,
| cond |     | cond |     |     |     |     |     |     |     |     |     |     |     |
| ---- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
and not on θ. Consequently, the Optimization Problems 2.7 and 3.1 coincide: the minimizer of J cond
| and the minimizer |             | of  | L    | satisfy | θ∗ =  | θ           | .   |     |     |     |     |     |     |
| ----------------- | ----------- | --- | ---- | ------- | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- |
|                   |             |     | cond |         |       | cond        |     |     |     |     |     |     |     |
| Proof of          | the theorem |     | may  | be      | found | in Appendix |     | A.  |     |     |     |     |     |
Remark 3.3. The preceding theorem does not use any properties of the specific change of measure
(tilting) defined by (14). Thus, the theorem generalizes to the other tiltings of the product measure
| µ (du)µ (dv) | that | we  | discuss | in  | Subsection |     | 3.3. |     |     |     |     |     |     |
| ------------ | ---- | --- | ------- | --- | ---------- | --- | ---- | --- | --- | --- | --- | --- | --- |
u v
⋄

|                  |     |               |     | Baptista, | Stuart,        | Tran/Contrastive |     | Learning |     |     |     | 11  |
| ---------------- | --- | ------------- | --- | --------- | -------------- | ---------------- | --- | -------- | --- | --- | --- | --- |
| 3.2. Generalized |     | Probabilistic |     |           | Loss Functions |                  |     |          |     |     |     |     |
In the previous subsection we have shown that the population level formulation of the standard
constrastive learning problem can be recast as learning a joint distribution, from a parameterized
class, that best matches the true joint distribution of the data. Matching is determined by a sum of
two terms measuring Kullback-Liebler divergences between the conditionals of the model and true
joint distributions. This idea may be generalized in a number of directions, outlined in the next two
subsubsections.
| 3.2.1. Generalized |     | Conditional |     | Losses |     |     |     |     |     |     |     |     |
| ------------------ | --- | ----------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
One natural direction to generalize Optimization Problem 3.1 is to replace the Kullback-Liebler
divergence D by an arbitrary divergence, or metric, D (resp. D ) in the term leading to the
|     | kl  |     |     |     |     |     |     | U   | V   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
u conditional (resp. v conditional). Furthermore, since the two conditionals for u and v may have
| −   |     | −   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
different representations, the two terms in the loss function may not have a similar scale; indeed
this observation applies to the original Optimization Problem 3.1. It may then be desirable to add
R
scalar parameters λ ,λ that balance the two terms. These two generalizations result in the
|     |     | u   | v ∈ | +   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
following minimization problem, in which D denotes the pair (D ,D ) :
|              |     |         |      |         |     |     |     |            | U V     |     |     |         |
| ------------ | --- | ------- | ---- | ------- | --- | --- | --- | ---------- | ------- | --- | --- | ------- |
| Optimization |     | Problem | 3.4. |         |     |     |     |            |         |     |     |         |
|              |     | λ       |      |         |     |     |     | λ          |         |     |     |         |
|              |     |         | uE   | (cid:2) |     |     |     | (cid:3) vE | (cid:2) |     |     | (cid:3) |
J cond,D (θ;λ u ,λ v ) = v∼µv D U (µ u|v ( v) ν u|v ( v;θ)) + u∼µu D V (µ v|u ( u) ν v|u ( u;θ)) ,
|      |     | 2          |     |        | ·|   | ||    | ·|  | 2   |     | ·| || | ·|  |     |
| ---- | --- | ---------- | --- | ------ | ---- | ----- | --- | --- | --- | ----- | --- | --- |
| θ (λ | ,λ  | ;D) argmin |     | J      | (θ;λ | ,λ ). |     |     |     |       |     |     |
| cond | u v | ∈          |     | cond,D | u    | v     |     |     |     |       |     |     |
θ∈Rp
|     |     |     |     | (cid:0) |     | (cid:1) |     |     |     |     |     |     |
| --- | --- | --- | --- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- |
Note in particular that θ 1,1;(D ,D ) = θ . The ratio of hyperparameters λ /λ needs
|     |     |     | cond |     | kl  | kl  | cond |     |     |     | u   | v   |
| --- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
to be tuned to balance the two contributions to the objective function. Furthermore, by choosing
λ = 0 (resp. λ = 0) we obtain an objective function tuned to choose the joint distribution simply
| v   |     | u   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
to match the u v (resp. v u) conditional and not the sum of both; this is desirable in applications
|     |     | |   | |   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
where the desired downstream tasks focus on only one of the two conditionals. An example of this,
covered in Subsection 4.2, is the fine-tuning of classifiers based on contrastive learning. Indeed, in
Subsection 6.2 we show that the original MNIST digit classification algorithm can be viewed as a
generalization of the standard CLIP methodology, using such a one-sided loss.
Remark 3.5. Optimization problem 3.4 is well-defined for any divergence pair D. But a critical
and practical issue is whether it defines an optimization problem for θ which is actionable given only
samples from µ. This is possible when D ( ),D ( ) are both chosen to be the forward Kullback-
|     |     |     |     |     | U   |      | V    |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | ---- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     | ·||· | ·||· |     |     |     |     |     |
Liebler divergence = D kl ( ); but it is not possible for the reverse Kullback-Liebler divergence,
·||·
which requires knowledge of the change of measure r := dµ/(dµ dµ ). Nor is it possible for
u v
⊗
the χ2 divergence or for the Hellinger or TV metrics, all of which also require knowledge of r.
−
However, the optimization problem is still actionable for some choices other than the Kullback-Liebler
divergence. The energy distances, of which maximum mean discrepancy (MMD) is a special case,
provide a useful class of examples. We illustrate this with Example B.1 in Appendix B.
⋄
| 3.2.2. Generalized |     | Joint | Losses |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Nowweproposeasomewhatdifferentclassofobjectives,basedaroundmatchingthejointdistribution
rather than conditionals. We assume that µ has density r with respect to µ µ , so that
|     |     |     |     |          |     |           |     |             |     | u ⊗ v |     |      |
| --- | --- | --- | --- | -------- | --- | --------- | --- | ----------- | --- | ----- | --- | ---- |
|     |     |     |     | µ(du,dv) |     | = r(u,v)µ |     | (du)µ (dv). |     |       |     | (19) |
|     |     |     |     |          |     |           | u   | v           |     |       |     |      |

|           |     |        |       |         | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     |     | 12  |
| --------- | --- | ------ | ----- | ------- | --------- | ------- | ---------------- | --- | -------- | --- | --- | --- |
| Recalling |     | ν( ;θ) | given | by (13) | we may    | now     | consider         |     |          |     |     |     |
·
| Optimization |     |     | Problem | 3.6. |     |             |           |       |       |     |     |     |
| ------------ | --- | --- | ------- | ---- | --- | ----------- | --------- | ----- | ----- | --- | --- | --- |
|              |     |     |         |      |     | J joint (θ) | = D kl (µ | ν( ,  | ;θ)), |     |     |     |
|              |     |     |         |      |     |             |           | || ·  | ·     |     |     |     |
|              |     |     |         |      |     | θ           | argmin    | J     | (θ).  |     |     |     |
|              |     |     |         |      |     | joint       |           | joint |       |     |     |     |
|              |     |     |         |      |     |             | ∈ θ∈Rp    |       |       |     |     |     |
Now note that, recalling function ρ is defined by (14), we are attempting to model r( , ) by
· ·
parameterized function ρ( , ;θ). Minimization of J over θ reflects this goal as the following
joint
|          |             |     |        | ·    | ·         |     |              |     |               |     |     |      |
| -------- | ----------- | --- | ------ | ---- | --------- | --- | ------------ | --- | ------------- | --- | --- | ---- |
| explicit | calculation |     | shows: |      |           |     |              |     |               |     |     |      |
|          |             |     |        | D (µ | ν( , ;θ)) | =   | E [logr(u,v) |     | logρ(u,v;θ)]. |     |     | (21) |
|          |             |     |        | kl   |           |     | (u,v)∼µ      |     |               |     |     |      |
|          |             |     |        |      | || · ·    |     |              |     | −             |     |     |      |
The first term does not involve θ and may be ignored for the purposes of minimization to determine
the optimal θ∗. Thus we see that Optimization Problem 3.6 may be formulated as:
| Optimization |     |     | Problem | 3.7. |     |     |      |     |     |     |     |     |
| ------------ | --- | --- | ------- | ---- | --- | --- | ---- | --- | --- | --- | --- | --- |
|              |     |     | E       |      |     |     | logE |     |     |     |     |     |
L (θ) = [ g¯ (u;θ ),g¯ (v;θ ) /τ] [exp( g¯ (u;θ ),g¯ (v;θ ) /τ)],
|     | joint  |         | (u,v)∼µ | u     | u    | v       | v       | (u,v)∼µu⊗µv |     | u u v | v   |     |
| --- | ------ | ------- | ------- | ----- | ---- | ------- | ------- | ----------- | --- | ----- | --- | --- |
|     | −      |         |         | ⟨     |      |         | ⟩ −     |             |     | ⟨     | ⟩   |     |
|     | θ      |         | argmin  | L     | (θ). |         |         |             |     |       |     |     |
|     |        | joint   |         | joint |      |         |         |             |     |       |     |     |
|     |        | ∈       | θ∈Rp    |       |      |         |         |             |     |       |     |     |
|     | Indeed | we have | proved: |       |      |         |         |             |     |       |     |     |
|     |        |         |         |       |      | (cid:0) | (cid:1) |             |     |       |     |     |
Theorem 3.8. Assume that E logr(u,v) < . Then, J (θ) = L (θ)+C where C does
|     |     |     |     |     | (u,v)∼µ |     |     |     | joint | joint |     |     |
| --- | --- | --- | --- | --- | ------- | --- | --- | --- | ----- | ----- | --- | --- |
∞
not depend on θ. Consequently, the Optimization Problems 3.6 and 3.7 coincide.
Remark 3.9. Evaluating the joint loss in Optimization Problem 3.7 has computational advantages
over the commonly used loss L for contrastive learning, defined in Optimization Problem 2.7. The
cond
joint loss L is efficiently evaluated by computing the cosine similarities for one batch of data from
joint
the joint distribution µ and one batch from the tensor product of marginal distributions µ µ . The
u v
⊗
contrastive loss L cond , on the other hand, requires a negative batch for each sample from one of the
marginal distributions to evaluate the second and third terms in Optimization Problem 2.7.
⋄
The following theorem connects the two optimization problems and is proved in Appendix A:
Rp,
Theorem 3.10. For all θ the objectives in Optimization Problems 2.7 and 3.7 are related by
∈
the inequality
|     |     |     |     |     |     | L   | (θ) L | (θ).  |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | cond  | joint |     |     |     |     |
≤
Thus,
|     |     |     |     |     | L (θ | )    | L (θ       | )   | L (θ        | ).  |     |     |
| --- | --- | --- | --- | --- | ---- | ---- | ---------- | --- | ----------- | --- | --- | --- |
|     |     |     |     |     | cond | cond | cond joint |     | joint joint |     |     |     |
|     |     |     |     |     |      |      | ≤          | ≤   |             |     |     |     |
Thus, solution of the joint minimization problem provides an upper bound for the conditional
optimization problem. Finally we note that it is also possible to generalize Optimization Problem 3.6
| by  | using | a different | divergence, |     | or metric: |      |            |      |         |     |     |      |
| --- | ----- | ----------- | ----------- | --- | ---------- | ---- | ---------- | ---- | ------- | --- | --- | ---- |
|     |       |             |             |     |            | θ∗ = | argmin D(µ | ν(   | , ;θ)). |     |     | (23) |
|     |       |             |             |     |            |      | θ∈Rp       | || · | ·       |     |     |      |
The comments in Remark 3.5, concerning actionable loss functions, apply also to this loss function.
One example of an actionable loss function is to take D to be the maximum mean discrepancy D
mmd
as defined in Example B.1. This results in the following optimization problem:
| Optimization |     |     | Problem     | 3.11. |        |                |           |     |                      |         |     |     |
| ------------ | --- | --- | ----------- | ----- | ------ | -------------- | --------- | --- | -------------------- | ------- | --- | --- |
|              |     |     |             |       | 2E     |                | k(x,y)+E  |     |                      | k(y,y′) |     |     |
|              |     |     | L joint,mmd | (θ)   | =      |                |           |     | (y,y′)∼ν(·;θ)⊗ν(·;θ) |         |     |     |
|              |     |     |             |       | −      | (x,y)∼µ⊗ν(·;θ) |           |     |                      |         |     |     |
|              |     |     | θ           |       | argmin | L              | (θ).      |     |                      |         |     |     |
|              |     |     | joint,mmd   |       |        |                | joint,mmd |     |                      |         |     |     |
∈ θ∈Rp

|                  |     |         |     | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     |     | 13  |
| ---------------- | --- | ------- | --- | --------- | ------- | ---------------- | --- | -------- | --- | --- | --- |
| 3.3. Generalized |     | Tilting |     |           |         |                  |     |          |     |     |     |
Recall that contrastive learning may be thought of as learning a probability measure ν, given by
(13), (14), so that it is close to the data generating distribution µ. The change of measure ρ defined
in (14) is referred to as a tilting. In this subsection we develop variants on this standard tilting.
We concentrate on two variants both of which we will return to in the Gaussian setting studied
in Section 5. However, the reader will readily identify numerous other generalizations based on
| different choices   | of  | parameterized |     | function |     | ρ in the | expression |     | (13). |     |     |
| ------------------- | --- | ------------- | --- | -------- | --- | -------- | ---------- | --- | ----- | --- | --- |
| 3.3.1. Unnormalized |     | Encoders      |     |          |     |          |            |     |       |     |     |
Recall that contrastive learning introduces encoders g ,g on the two modalities, but defines loss
|     |     |     |     |     |     |     | u   | v   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
functions through use of the cosine distance based on the normalized encoders given in (2). However
it is possible to simply drop the constraint that the embedding vectors are normalized. Making this
change in (13), (14) leads to a new model class of joint distributions with the form
|     |     | ν(du,dv;θ) |     | = ρ(u,v;θ)µ |          | (du)µ | (dv),    |      |          |     | (25a) |
| --- | --- | ---------- | --- | ----------- | -------- | ----- | -------- | ---- | -------- | --- | ----- |
|     |     |            |     |             |          | u     | v        |      |          |     |       |
|     |     |            |     | 1           | (cid:16) |       |          |      | (cid:17) |     |       |
|     |     | ρ(u,v;θ)   |     | = exp       | g        | (u;θ  | ),g (v;θ | ) /τ | ,        |     | (25b) |
|     |     |            |     |             |          | u u   | v        | v    |          |     |       |
|     |     |            |     | Z           | ⟨        |       |          | ⟩    |          |     |       |
(cid:90)
|     |     |     | Z   | =   | exp( | g (u;θ | ),g (v;θ | )   | /τ)µ (du)µ | (dv). | (25c) |
| --- | --- | --- | --- | --- | ---- | ------ | -------- | --- | ---------- | ----- | ----- |
|     |     |     |     |     |      | ⟨ u    | u v      | v   | ⟩ u        | v     |       |
U×V
These unnormalized encoders provide our first example of a generalized tilting.
| 3.3.2. L2 Distance |     |     |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
−
Now note that, if the embedding vectors in (13), (14) are normalized, then the density function of
| the joint distribution |     | (14) | can | be written | as         |     |     |     |          |     |     |
| ---------------------- | --- | ---- | --- | ---------- | ---------- | --- | --- | --- | -------- | --- | --- |
|                        |     |      |     | 1          | (cid:16) 1 |     |     |     | (cid:17) |     |     |
2
|     |     | ρ(u,v;θ) |     | exp |     | g¯ (u;θ | ) g¯ | (v;θ | ) µ (du)µ | (dv). |     |
| --- | --- | -------- | --- | --- | --- | ------- | ---- | ---- | --------- | ----- | --- |
|     |     |          | ∝   | Z   | −2τ | | u     | u −  | v v  | | u       | v     |     |
Using this form for the tilting, but in the unnormalized setting, leads to the model:
|     | ν(du,dv;θ) |          | =   | ρ(u,v;θ)µ  | (du)µ    | (dv),  |     |      |             |       | (26a) |
| --- | ---------- | -------- | --- | ---------- | -------- | ------ | --- | ---- | ----------- | ----- | ----- |
|     |            |          |     |            | u        | v      |     |      |             |       |       |
|     |            |          |     | 1 (cid:16) | 1        |        |     |      | (cid:17)    |       |       |
|     |            | ρ(u,v;θ) | =   | exp        |          | g (u;θ | ) g | (v;θ | ) 2 µ (du)µ | (dv), | (26b) |
|     |            |          |     |            |          | u      | u v | v    | u           | v     |       |
|     |            |          |     | Z          | −2τ      | |      | −   |      | |           |       |       |
|     |            |          |     | (cid:90)   | (cid:16) | 1      |     |      | (cid:17)    |       |       |
2
|     |     |     | Z = | exp |     | g (u;θ | )   | g (v;θ | ) µ (du)µ | (dv). | (26c) |
| --- | --- | --- | --- | --- | --- | ------ | --- | ------ | --------- | ----- | ----- |
|     |     |     |     |     | −2τ | | u    | u − | v      | v | u     | v     |       |
U×V
These unnormalized encoders provide our second example of a generalized tilting.
| 3.3.3. The General |     | Setting |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Both of the preceding modified tiltings can be used within all of the generalized probabilistic
loss functions described in Subsection 3.2; in particular, Optimization Problems 3.4 and 3.6 are
well-defined for any tilting of the form (13). In this context, the following observation will be useful
| to us in what | follows. |     |     |     |     |     |     |     |     |     |     |
| ------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

|     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     |     |     |     | 14  |
| --- | --- | --- | --- | --------- | ------- | ---------------- | --- | -------- | --- | --- | --- | --- | --- |
Remark 3.12. Theorem 3.2 may be generalized to Optimization Problem 3.4 in the case where D
U
and D V are both chosen to be D kl . The analogous asymmetric loss function and minimization problem
| is given     | by  |         |        |         |         |          |          |       |     |        |          |     |     |
| ------------ | --- | ------- | ------ | ------- | ------- | -------- | -------- | ----- | --- | ------ | -------- | --- | --- |
| Optimization |     | Problem |        | 3.13.   |         |          |          |       |     |        |          |     |     |
|              |     |         |        |         |         | (cid:20) |          |       |     |        | (cid:21) |     |     |
|              |     |         |        |         |         | λ        |          |       | λ   |        |          |     |     |
|              |     |         | L (θ;λ | ,λ ) =  | E       |          | u logρ(u | v;θ)+ | v   | logρ(v | u;θ) .   |     |     |
|              |     |         | cond   | u v     | (u,v)∼µ |          |          |       |     |        |          |     |     |
|              |     | −       |        |         |         | 2        |          | |     | 2   |        | |        |     |     |
|              |     | θ       | (λ     | ,λ ;D ) | argminL |          | (θ;λ     | ,λ ). |     |        |          |     |     |
|              |     |         | cond u | v kl    |         | cond     |          | u v   |     |        |          |     |     |
|              |     |         |        | ∈       | θ∈Rp    |          |          |       |     |        |          |     |     |
Following the same proof of Theorem 3.2, the minimizers of J ( ;λ ,λ ) and L ( ;λ ,λ )
|     |     |     |     |     |     |     |     |     | cond |     | u v cond |     | u v |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | -------- | --- | --- |
|     |     |     | R   |     |     |     |     |     |      | ·   |          | ·   |     |
coincide for all λ ,λ . The reader will be able to identify a similar generalization of Theorem 3.8.
|     |     | u   | v   | +   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∈
⋄
| 4. Retrieval |     | and | Classification |     |     |     |     |     |     |     |     |     |     |
| ------------ | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Two information science applications that make use of contrastive learning are retrieval and clas-
sification. In Subsection 4.1 we describe the problem of retrieval, framing it using the perspective
on contrastive learning that we have developed over the two preceding sections; Subsection 4.2 is
| devoted | to a | similar | treatment | of classification. |     |     |     |     |     |     |     |     |     |
| ------- | ---- | ------- | --------- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
4.1. Retrieval
In this subsection we formalize the use of contrastive learning as a building block in the retrieval
task. The methodology we describe in this subsection proceeds with θ = (θ ,θ ) fixed at the optimal
|     |     |     |     |     |     |     |     |     |     |     | u v |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
value found through the contrastive learning procedure described in Subsection 3.1, or its subsequent
variants detailed in Subsections 3.2 and 3.3. Given one realization from a data modality, crossmodal
retrieval identifies relevant items, from the prescribed marginal distribution of the other modality,
that are aligned with it. In practice, this is accomplished by computing the cosine similarity of
the given realization with samples from the other data modality and returning the elements (or
possiblyasingleelement)withhighestsimilarity.ThecompleteprocedureisdescribedinAlgorithm1.
This procedure is referred to as zero-shot retrieval as it does not require any additional training of
| parameter | θ after | learning   |      | the encoders. |        |          |     |         |       |     |     |     |     |
| --------- | ------- | ---------- | ---- | ------------- | ------ | -------- | --- | ------- | ----- | --- | --- | --- | --- |
| Algorithm | 1       | Crossmodal |      | Retrieval     |        |          |     |         |       |     |     |     |     |
| 1: Input: | Input   | v∈V,       | data | samples UN    | ={ui}N | , number | of  | similar | items | K   |     |     |     |
i=1
|            |        |              |     | :=⟨g¯ (ui),g¯ |        |          |        | ui  |     |     |           |     |     |
| ---------- | ------ | ------------ | --- | ------------- | ------ | -------- | ------ | --- | --- | --- | --------- | --- | --- |
| 2: Compute | cosine | similarities |     | s i u         | v (v)⟩ | for each | sample |     |     |     |           |     |     |
|            |        |              |     | σ∗ ∈NK        |        |          |        | σ∗  |     |     | (cid:80)K |     |     |
3: Identify K distinct indices with largest similarities, i.e., ∈argmax s σ(i)
|            |         |     |        |               |     |     |     |     |     | σ∈[[1,K]] | k=1 |     |     |
| ---------- | ------- | --- | ------ | ------------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- |
| 4: Output: | Samples |     | uσ∗(k) | for k=1,...,K |     |     |     |     |     |           |     |     |     |
Example 4.1. Take to denote images and to denote text prompts. In this setting v
|     |     |     | U   |     |     |     | V   |     |     |     |     |     | ∈ V |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
represents a text prompt, typically not in the training data set N; and N is the collection of N
|     |     |     |     |     |     |     |     |     | V   |     | U N |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
images in the training dataset. Crossmodal retrieval seeks the top K images from that are most
U
aligned, in the precise sense described by Algorithm 1, to the given prompt v. The embedding of
all images in N is precomputed offline. Given any specific text prompt v encountered, the online
U
| computational |     | cost | is then | (n N). |     |     |     |     |     |     |     |     |     |
| ------------- | --- | ---- | ------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
O e

|     |     |     | Baptista, |     | Stuart, Tran/Contrastive |     |     | Learning |     |     | 15  |
| --- | --- | --- | --------- | --- | ------------------------ | --- | --- | -------- | --- | --- | --- |
Here we show how it is performed in practice: with empirical measures computed from the discrete
data set in Data Assumption 2.1. A generalization of retrieval to the population level setting of
| measures | with continuous |     | densities | is  | found in Appendix |     | C.1. |     |     |     |     |
| -------- | --------------- | --- | --------- | --- | ----------------- | --- | ---- | --- | --- | --- | --- |
In practice µ may not have density with respect to Lebesgue measure and, more fundamentally,
u
is only available through samples: the marginal distribution for u is specified by the equally weighted
empirical data measure µN in (3) supported on N in (4). Then, the learned conditional distribution
u
U
| for u v | is given by |     |     |      |             |      |      |              |     |     |      |
| ------- | ----------- | --- | --- | ---- | ----------- | ---- | ---- | ------------ | --- | --- | ---- |
| |       |             |     |     |      | (cid:16)    |      |      | (cid:17)     |     |     |      |
|         |             | νN  | (du | v;θ) | exp g¯ (u;θ | ),g¯ | (v;θ | ) /τ µN(du). |     |     | (28) |
|         |             | u|v |     |      | u           | u    | v    | v u          |     |     |      |
|         |             |     | |   | ∝    | ⟨           |      |      | ⟩            |     |     |      |
We may still give an interpretation as a Bayesian inverse problem: we take as µN from (3) as the prior
u
andthefunctionρ( v)definedin(9a)asthelikelihood(uptoaconstantofproportionality).However,
|     | ·|  |     | νN  |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the conditional distribution , which is the resulting posterior in the Bayesian interpretation, is
u|v
only supported at N points N = ui N . The measure parameterized by v can be written as the
|          |                   |     | U   | {   | }i=1 |     |     |     |     |     |     |
| -------- | ----------------- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
| weighted | empirical measure |     |     |     |      |     |     |     |     |     |     |
N
(cid:88)
|     |     |     |     | νN  | ( v;θ) := | w   | (v;θ)δ | .   |     |     |     |
| --- | --- | --- | --- | --- | --------- | --- | ------ | --- | --- | --- | --- |
|     |     |     |     | u|v |           |     | i      | ui  |     |     |     |
·|
i=1
The weights w (v;θ) [0,1] sum to 1 for any input v. They are defined as
i
∈
|     |         |     | ω         | (v;θ)   |           |     |       | (cid:16)     |       | (cid:17) |     |
| --- | ------- | --- | --------- | ------- | --------- | --- | ----- | ------------ | ----- | -------- | --- |
|     | w (v;θ) | :=  | i         |         | , ω (v;θ) |     | = exp | g¯ (ui;θ),g¯ | (v;θ) | /τ .     |     |
|     | i       |     | (cid:80)N |         | i         |     |       | u            | v     |          |     |
|     |         |     |           | ω (v;θ) |           |     |       | ⟨            |       | ⟩        |     |
|     |         |     | ℓ=1       | ℓ       |           |     |       |              |       |          |     |
Continuing the interpretation as a Bayesian inverse problem, the mode or MAP point in this context
is defined as the point in N, which maximizes the density of the posterior νN with respect to
|     |     |     | U   |     |     |     |     |     |     | u|v |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | µN; |     | µN  |     |     |     |     |     |     |     |
the empirical measure because comprises equally weighted points, this is equivalent to
|     |     | u   |     |     | u   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
maximizing the likelihood over the data set. Given that the weights w are proportional to the
i
density, on the data set, we have the following theorem. The proof is found in Appendix A.
Theorem 4.2. The retrieval process in Algorithm 1 with K = 1 finds a mode of the empirical
νN
| conditional | distribution |     | . For | each      | v , retrieval |     | computes |     |     |     |      |
| ----------- | ------------ | --- | ----- | --------- | ------------- | --- | -------- | --- | --- | --- | ---- |
|             |              | u|v |       |           | ∈ Y           |     |          |     |     |     |      |
|             |              |     |       | argmax    | g¯ (ui;θ),g¯  |     | (v;θ)    | .   |     |     | (29) |
|             |              |     |       |           | u             |     | v        |     |     |     |      |
|             |              |     |       | i=1,...,N | ⟨             |     |          | ⟩   |     |     |      |
4.2. Classification
In this subsection we formalize the use of contrastive learning as a building block in the classification
task. Whilst finding an image from text is the canonical retrieval task, as explained in Example 4.1,
the canonical classification task is the assignation of a text classifier to an image. The complete
procedure is described in Algorithm 2 (recall Remark 2.2 for definition of the softmax operation)
and a key point to appreciate is that the labels are not necessarily taken from the training data.
| Algorithm | 2 Crossmodal |        | Classifier |     |     |     |     |     |     |     |     |
| --------- | ------------ | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1: Input: | Input u∈U,   | labels | C ={vi}K   |     |     |     |     |     |     |     |     |
i=1
2: Compute cosine similarities s :=⟨g¯ (u),g¯ (vi)⟩ for each label vi and i=1,...,K
|     |     |     | i   | u   | v   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i∗
3: Output: Most likely label =argmax i=1,...,K s i and (optional) probabilities softmax(s i ) for each label

|     |     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     |     | Learning |     |     | 16  |
| --- | --- | --- | --- | --- | --------- | ------- | ---------------- | --- | --- | -------- | --- | --- | --- |
Example 4.3. Take to denote the space of images and to denote text prompts. Although
|     |     |     |     | U   |     |     |     |     |     | V   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
image-to-text classification might, in principle, be achieved by simply reversing the roles of text and
image in Example 4.1 for retrieval, this typically results in poor performance because the set of all
text prompts in the data set is typically inadequate to classify arbitrary new images. Moreover, often
interest is focused on classifying an image using a small set of labels—for example diagnosing whether
tissue in medical images is healthy or not. To solve classification problems of this type, an initial
pre-training step of contrastive learning is performed using a large population of text and image
pairs N and N as defined in (4); typically , or at least all of , is not in N. Then, fine-tuning
|     | U   | V   |     |     |     |     | C   |     |     | C   |     | V   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
specializes the classifier to increase its accuracy as a predictor of a label in , given an image, by
C
updating encoder parameters on the basis of a new (and typically smaller) set of labels and images,
that are different from those considered during pre-training, but now containing .
C
When pretrained encoders are used to classify inputs among a finite set without additional
fine-tuning, Algorithm 2 is referred to as zero-shot classification. We will concentrate, henceforth,
on using fine-tuning, motivated by the preceding example, but not working in the specific context
of text-image data. Next, we describe actionable algorithms for classification, based on discrete
data with a finite number of labels. A description of the population level problem with continuous
| densities |     | is presented |     | in Appendix | C.2. |     |     |     |     |     |     |     |     |
| --------- | --- | ------------ | --- | ----------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
Our starting point is to assume that we have a pretrained model, via access to the data in Data
Assumption 2.1, which is used to determine the parameters of the encoder (θ ,θ ). We first describe
u v
zero-shot classification: how to use the encoders to perform classification over a new distribution of
labels, using this pretrained model. We then update the parameters of the encoder for v, and learn a
new reference marginal distribution, by use of a second data set of size M N—the fine-tuning
≪
process, with goal being improved bespoke classification. This second data set is defined in:
Data Assumption 4.4. The fine-tuning data comprises pairs (ui,vi) M drawn i.i.d.from a joint
|     |     |     |     |     |     |     |     |     |     | (cid:98) (cid:98) | }i=1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | ---- | --- | --- |
{
| distribution |     | µ(u,v) | on  |     | .   |     |     |     |     |     |     |     |     |
| ------------ | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:98)
U ×V
|     | To describe | zero-shot |     | classification |     | we  | first define    | finite | set           |     |     |     |     |
| --- | ----------- | --------- | --- | -------------- | --- | --- | --------------- | ------ | ------------- | --- | --- | --- | --- |
|     |             |           |     |                |     |     | vi K            |        | vi M          |     |     |     |     |
|     |             |           |     |                |     | =   |                 |        | .             |     |     |     |     |
|     |             |           |     |                |     | C   | { (cid:98) }i=1 | ⊆ {    | (cid:98) }i=1 |     |     |     |     |
The set comprises a set of labels which we wish to assign in the classification task. Without loss
C
of generality we have ordered them to be the first K members of the entire fine-tuning data set,
| marginalized |     | on  | . We | then | define | the empirical |     | measures |     |     |     |     |     |
| ------------ | --- | --- | ---- | ---- | ------ | ------------- | --- | -------- | --- | --- | --- | --- | --- |
V
|     |     |     |          |     | K        |     |           |            | K           |           | K          |             |      |
| --- | --- | --- | -------- | --- | -------- | --- | --------- | ---------- | ----------- | --------- | ---------- | ----------- | ---- |
|     |     |     |          | 1   | (cid:88) |     |           | 1 (cid:88) |             |           | 1 (cid:88) |             |      |
|     |     |     | µK       | :=  | δ        | ,   | µK :=     |            | δ ,         | µK :=     |            | δ .         | (30) |
|     |     |     | (cid:98) |     | (uℓ,vℓ)  |     | (cid:98)u |            | u(cid:98) ℓ | (cid:98)v |            | v(cid:98) ℓ |      |
|     |     |     |          | K   |          |     |           | K          |             |           | K          |             |      |
|     |     |     |          |     | ℓ=1      |     |           | ℓ=1        |             |           | ℓ=1        |             |      |
We may use µK as reference measure to define a conditional measure from a pair of pretrained
(cid:98)v
encoders with parameter θ = (θ ,θ ); this leads to the weighted empirical measure
|     |     |     |     |     | u v |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
K
(cid:88)
νK (v u;θ) exp( g¯ (u;θ ),g¯ (v;θ ) /τ)µK(dv) = w (u;θ)δ . (31)
|     |     |     | (cid:98)v|u |     |     | u   | u v | v   | (cid:98)v |     | i   | vi  |     |
| --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- |
|     |     |     |             | | ∝ | ⟨   |     |     |     | ⟩         |     |     |     |     |
i=1
Here, w (u;θ) [0,1] are normalized weights, defined for any sample u, and given by
i
∈
ω (u;θ)
|     |     | w   | (u;θ) | :=        | i       | ,   | ω   | (u;θ) | = exp( | g¯ (u;θ | ),g¯ (vi;θ | ) /τ). |     |
| --- | --- | --- | ----- | --------- | ------- | --- | --- | ----- | ------ | ------- | ---------- | ------ | --- |
|     |     |     | i     |           |         |     | i   |       |        | u u     | v          | v      |     |
|     |     |     |       | (cid:80)K | ω (u;θ) |     |     |       | ⟨      |         |            | ⟩      |     |
ℓ=1 ℓ
| The | most | likely | label | for each | input    | u is | then given  | by   | the mode  |     |      |     |     |
| --- | ---- | ------ | ----- | -------- | -------- | ---- | ----------- | ---- | --------- | --- | ---- | --- | --- |
|     |      |        |       | v∗(u)    | argmaxνK |      | (vi         | u;θ) | = argmaxw |     | (u). |     |     |
|     |      |        |       |          |          |      | (cid:98)v|u |      |           | i   |      |     |     |
|     |      |        |       |          | ∈        | vi∈C |             | |    | i=1,...,K |     |      |     |     |

Baptista, Stuart, Tran/Contrastive Learning 17
Example 4.5. In the Bayesian interpretation, where we view the titling proportional to ρ(u,v;θ) as
a likelihood function, classification over an equally weighted marginal distribution of labels is a MAP
estimator. It can also be interpreted as the restricted maximum likelihood estimator
argmaxρ(vi u;θ) = argmax g¯ (u;θ ),g¯ (vi;θ ) .
u u v v
vi∈C | vi∈C ⟨ ⟩
Asinthepopulationlosssettingonemayfine-tunetheparametersofthev encoderandredefinethe
prior to align the classification algorithm with the fine-tuning data set given in Data Assumption 4.4.
To this end we now fix the pretrained parameters θ of the u encoder. We then define a new prior
u
µK defined by parameter θ := F = (F , ,F ) RK:
(cid:101)v ϕ 1
···
K
∈
K
(cid:88) exp(F )
µK(dv;θ ) = i δ . (32)
(cid:101)v ϕ (cid:80)K
exp(F )
v(cid:98) i
i=1 j=1 j
Thus µK is a reweighting of µK, analogous to the re-weighting of µ , to define µ , defined at the
(cid:101)v (cid:98)v (cid:98)v (cid:101)v
population level. Then, the conditional measure for v u depending on parameter ϑ = (θ ,F) is given
v
|
by the weighted empirical measure
K
(cid:88)
νK (v u;ϑ) exp( g¯ (u;θ ),g¯ (v;θ ) /τ)µK(dv;θ ) = w (u;θ)δ ,
(cid:101)v|u
| ∝ ⟨
u u v v
⟩
(cid:101)v ϕ i vi
i=1
where w (u;θ) [0,1] are normalized weights defined as
i
∈
w (u;θ) = ω i (u;θ) , ω (u;θ) = exp (cid:0) g¯ (u;θ ),g¯ (vi;θ ) /τ +F (cid:1) . (33)
i (cid:80)M ω (u;θ) i ⟨ u u v v ⟩ i
l=1 l
We wish to adjust θ , along with F, to improve performance on the specified classifiers . But, to
v
C
classify an input u among a finite set of labels from , we only need the action of the encoder at
C
vi . Thus, let the matrix G Rne×K be a set of weights where each column G = g¯ (vi;θ ) Rne
i v v
∈ C ∈ ∈
contains the evaluation of the encoder for label vi. We let e be the unit-vector corresponding to the
v
label v (i.e., one-hot encoding) and now redefine ϑ = (G,F), noting that we only need to optimize
over (G,F) not (θ ,F). We now fine-tune ϑ using the dataset µ prescribed in Data Assumption 4.4
v (cid:98)
by minimizing the discrete version of the one-sided loss in (3.4)2:
(cid:104) (cid:105)
LK (ϑ) = E (cid:10) e , (cid:0) G⊤g¯ (u;θ )/τ +F (cid:1)(cid:11) (34a)
−
fine (u,v)∼µ(cid:98) K v u u RK
(cid:104) (cid:105)
E logE exp (cid:10) e , (cid:0) G⊤g¯ (u;θ )/τ +F (cid:1)(cid:11) ,
−
u∼µ(cid:98) K
u
v∼µ(cid:98) K
v
v u u RK
ϑ argminLK (ϑ). (34b)
fine fine
∈
Given the learned parameter of the fine-tuned model, the classifier for each input v is defined as the
mode of the conditional distribution for v given u: 3
v∗(u) argmaxνK (v u;ϑ ) = argmin (cid:0) G⊤g¯ (u;θ )+F (cid:1) . (35)
∈ (cid:101)v|u | fine i u u i
vi∈C i=1,...,K
2Throughout the paper ⟨·.·⟩ denotes the inner-product on Rne; note that the inner-product here is on RK.
3The optimization problem defined by equations (34) has population level analog Optimization Problem C.2;
likewise (35) has population level analog (67).

|     |          |         |     |     | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     |     | 18  |
| --- | -------- | ------- | --- | --- | --------- | ------- | ---------------- | --- | -------- | --- | --- | --- |
| 5.  | Gaussian | Setting |     |     |           |         |                  |     |          |     |     |     |
In this section we study the contrastive learning problem when the data distribution is a multivariate
Gaussian. This enables explicit insights into the capabilities of the standard approach to contrastive
learning and the suggested variants on it proposed in Section 3; these insights are obtained via theory
and via straightforward numerical experiments. The main theoretical highlights are contained in
Corollaries 5.2, 5.4 and 5.7. And these results are supported by further theory concerning low-rank
approximation and by numerical studies that take our understanding beyond the theory.
We let (u,v) := Rnu Rnv be a multivariate centered Gaussian random variable with
|     |     |     | ∈ U ×V |     | ×   |     |     |     |     |     |     |     |
| --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
distribution µ = (0, ). We assume that the covariance matrix has block form
|     |     |     | N   | C   |     |     |          |      |          | C   |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---- | -------- | --- | --- | --- |
|     |     |     |     |     |     |     | (cid:20) |      | (cid:21) |     |     |     |
|     |     |     |     |     |     |     | = C      | uu C | uv ,     |     |     |     |
|     |     |     |     |     |     |     | C        | vu   | vv       |     |     |     |
|     |     |     |     |     |     |     | C        | C    |          |     |     |     |
where is strictly positive-definite. Matrices and are also then necessarily strictly positive-
|     |     |     |     |     |     |     | uu  |     | vv  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | C   |     |     |     |     |     | C   |     | C   |     |     |     |
definite and hence invertible. The two marginal distributions for u and v are given by
|     |     |     |     | µ   | ( ) = | (0, | ),   | µ   | ( ) = | (0,  | );  | (36) |
| --- | --- | --- | --- | --- | ----- | --- | ---- | --- | ----- | ---- | --- | ---- |
|     |     |     |     |     | u ·   | N   | C uu |     | v · N | C vv |     |      |
moreover, the two conditional distributions for u v and v u are given by
|     |     |     |     |            |     |        |     | |   | |   |      |            |       |
| --- | --- | --- | --- | ---------- | --- | ------ | --- | --- | --- | ---- | ---------- | ----- |
|     |     |     |     |            |     | −1v,   |     |     | :=  |      | −1         |       |
|     |     |     | µ   | u|v ( v) = | (   | uv     | u|v | ),  | u|v | uu   | uv vu ,    | (37a) |
|     |     |     |     | ·|         | N C | Cvv    | C   |     | C   | C −C | Cvv C      |       |
|     |     |     | µ   | ( u) =     | (   | −1u,   |     | ),  | :=  |      | −1 .       | (37b) |
|     |     |     | v|u |            |     | vu Cuu | v|u |     | v|u | vv   | vu CuuC uv |       |
|     |     |     |     | ·|         | N C |        | C   |     | C   | C −C |            |       |
In Section 3 we formulate the contrastive approach to learning in terms of finding a representation
of the joint distribution as a change of measure (tilting) from a reference measure defined as the
independent product of the marginals in (36). The key question we focus on in this section is the
ability of the learned joint distribution to accurately replicate the conditionals in (37). To enable
explicit analysis of this question we employ linear encoders and log-quadratic tiltings so that the
| learned | joint | distribution |     | is also | Gaussian. |     |     |     |     |     |     |     |
| ------- | ----- | ------------ | --- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
We introduce embedding (latent space) dimension n and seek tiltings based on the linear encoders
e
|     |     |     |     |     | g   | (u) = | Gu, | g   | (v) = Hv, |     |     | (38) |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --------- | --- | --- | ---- |
|     |     |     |     |     |     | u     |     | v   |           |     |     |      |
with G Rne×nu and H Rne×nv and n min(n ,n )4. We will work primarily in the settings of
|     |     |     |     |     |     | e   |     | u   | v   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | ∈   |     |     | ∈   |     |     | ≤   |     |     |     |     |     |
the tiltings defined in (25) and (26), making an exception to include normalization of the encoders
only in some specific numerical experiments. We will also take the temperature parameter τ = 1,
since this can be absorbed into the matrices G,H. To determine parameters G,H we will study both
Optimization Problems 3.13 and 3.6 for the generalized conditional and the joint losses, respectively.
In Subsection 5.1 we study the standard Optimization Problem 3.1, i.e., the particular symmetric
caseofOptimizationProblem3.13.Weshowwecanmatchtheconditionalmeansofbothdistributions
in (37), i f the latent space has high enough dimension (Corollary 5.2); otherwise we identify the
best low-rank approximation of the conditional means. However this methodology fails to represent
the conditional covariances. Subsection 5.2 uses a one-sided choice in Optimization Problem 3.13,
matching only the conditional of one modality on the other, not both. We show that in this setting
we can match the mean and covariance of the one-sided conditional distribution if the latent space
4Itisstraightforwardtogeneralizetheanalysistoasettingwithnon-zeromeanminµ.Toconsistentlyapproximate
theconditionalmeansinthissetting,onemayconsideraffineencodersoftheformg (u)=Gu+g,andg (v)=Hv+h
|      |       |        |     |     |     |     |     |     |     |     | u v |     |
| ---- | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|      | g∈Rnu | h∈Rnv. |     |     |     |     |     |     |     |     |     |     |
| with |       | and    |     |     |     |     |     |     |     |     |     |     |

Baptista, Stuart, Tran/Contrastive Learning 19
has high enoughdimension (Corollary 5.4);and weidentify the optimizationproblem forthe bestlow-
rank approximation. In Subsection 5.3, we employ the Optimization Problem 3.6 based on matching
the joint distribution rather than conditionals, and explicitly identify the family of minimizers which
attain the minimum value of the objective function; we show that the joint loss leads to better
representation of the marginals (Corollary 5.7). Subsection 5.4 is devoted to numerical illustrations
of the proven theoretical results, together with empirical extensions of the theory to new settings;
these new settings include the addition of normalization constraints on the encoders in the setting of
Subsection 5.1, loss matching both conditionals in the setting of Subsection 5.2 and the study of the
loss function minimizing divergence of the joint distributions from Subsection 5.3.
5.1. Cosine Distance: Conditional Loss
Consider Optimization Problem 3.1 with model class ν defined by the linear encoders in (38):
1 (cid:16) (cid:17)
ν(du,dv;θ) = exp Gu,Hv µ (du)µ (dv). (39)
u v
Z ⟨ ⟩
We may define parameter θ := (G,H). Note, however, that ν is invariant to re-scaling of the
parametersG αUGandH 1UH foranyscalarα > 0andanyorthonormalmatrixU Rne×ne.
(cid:55)→ (cid:55)→ α ∈
To avoid this invariance we redefine θ = A Rnu×nv, where A := G⊤H.
∈
For the Gaussian marginal distributions µ ,µ in (36), ν is a multivariate Gaussian distribution
u v
ν = (m , ) where the mean m and inverse covariance matrix −1 have the block-form
N θ C θ θ Cθ
(cid:20) 0 (cid:21) (cid:20) −1 A (cid:21)
m = , −1 = Cuu − . (40)
θ 0 Cθ A⊤ −1
− Cvv
Moreover, the learnable model ν has the conditional distributions
1 (cid:16) (cid:17)
ν (du v;A) = exp u,Av µ (du) = (du; Av, ), (41a)
u|v | Z ⟨ ⟩ u N C uu C uu
1 (cid:16) (cid:17)
ν (dv u;A) = exp u,Av µ (dv) = (dv; A⊤u, ). (41b)
v|u | Z ⟨ ⟩ v N C vv C vv
By Remark 3.12, the relevant optimization problem over θ := A is to minimize the loss function
(cid:104) (cid:105)
L (A) = E u,Av
cond (u,v)∼µ
− ⟨ ⟩
1 (cid:104) (cid:105) 1 (cid:104) (cid:105)
+ E logE exp( u,Av ) + E logE exp( u,Av ) . (42)
2
u∼µu v∼µv
⟨ ⟩ 2
v∼µv u∼µu
⟨ ⟩
The following result provides a closed form solution for the minimizer of this loss function. The
proof of the theorem may be found in Appendix D. Recall from Subsection 1.5 that for any matrix
Σ, Σ denotes the rank r truncation of its singular value decomposition.
r
−
Theorem 5.1. The minimizer of L (A) over all matrices of size n n is
cond u v
×
A∗ = −1 −1. (43)
CuuC uv Cvv
The minimizer of L (A) over the set of rank-r matrices = A Rnu×nv, rank(A) r for
cond r
A { ∈ ≤ }
0 < r min(n ,n ) is given by
u v
≤
A∗(r) = −1/2( −1/2 −1/2) −1/2. (44)
Cuu Cuu C uv Cvv r Cvv

Baptista, Stuart, Tran/Contrastive Learning 20
The following corollary is also proved in Appendix D:
Corollary 5.2. If the parameter in the learnable model ν (39) is chosen to be A = G⊤H = A∗,
Theorem 5.1 results in the approximate conditional distributions:
ν (u v;A∗) = ( −1v, ) (45a)
u|v | N C uv Cvv C uu
ν (v u;A∗) = ( −1u, ), (45b)
v|u | N C vuv Cuu C vv
Thus, the conditional means of ν in (45a) and (45b) match those of the data distribution µ in (37a)
and (37b), respectively; however, the conditional variances of ν are strictly larger, in the cone of
positive definite matrices, than those of the data distribution µ, unless the data distribution is in
product form with respect to u and v when they coincide.
This corollary shows that the conditional loss will not be consistent if used in a generative sense
for sampling from approximate conditional distributions. Sections 5.2 and 5.3 consider a different
alignmentmetricandlossfunction,respectively,andshowthatbydoingsowecanbettercharacterize
the approximated covariances. In particular, Corollary 5.4 shows that it is possible to exactly match
one of the conditional covariances. And Corollary 5.7 shows that using the joint loss rather than the
conditional loss yields a closer approximation to the marginal covariances of the data distribution.
For further discussion of the material in this subsection see Remarks D.1 and D.2; there we discuss
an alternative formulation of the loss function, and the setting with empirical data.
5.2. Positive Quadratic Form: Conditional Loss
Corollary 5.2 shows that, with learnable measure of the form (39), it is only possible to match the
conditional means, not the conditional covariances. To address this we now consider the following
generalization:
1 (cid:16) 1 (cid:17)
ν(du,dv;θ) = exp Gu Hv 2 µ (u)µ (v). (46)
u v
Z −2| − |
The parameters of this model are θ = (G,H). The measure ν is a multivariate Gaussian distribution
that depends on three matrix products A = G⊤H Rnu×nv, B = G⊤G Rnu×nu and C = H⊤H
∈ ∈ ∈
Rnv×nv. Moreover, the joint measure has the form ν(u,v;θ) = (m , ), where the mean and
θ θ
N C
inverse covariance matrix have the block form
(cid:20) 0 (cid:21) (cid:20) B+ −1 A (cid:21)
m = , −1 = Cuu − . (47)
θ 0 Cθ A⊤ C + −1
− Cvv
The parameterized model has the conditional distributions
ν (u v;θ) = 1 exp (cid:16) 1 Gu Hv 2 (cid:17) µ (du) = (cid:0) du;(B+ −1)−1Av,(B+ −1)−1(cid:1) , (48a)
u|v | Z −2| − | u N Cuu Cuu
ν (v u;θ) = 1 exp (cid:16) 1 Gu Hv 2 (cid:17) µ (dv) = (cid:0) dv;(C + −1)−1A⊤u,(C + −1)−1(cid:1) . (48b)
v|u | Z −2| − | v N Cvv Cvv
The additional degrees of freedom introduced by using (46) in place of (39) enable us to match the
mean and covariance of either one of the conditional distributions. To this end we study the optimal
solution when the objective only aims to match the conditional distribution for u v. Analogous
|
results may be derived for the v u conditional. To achieve this matching we employ Optimization
|
Problem 3.13 with (λ ,λ ) = (2,0) to obtain the following objective that is minimized to determine
u v
the parameters θ:
(cid:20) (cid:21)
(cid:16) 1 (cid:17) (cid:16) 1 (cid:17)
L (θ;2,0) = E E Gu Hv 2 +logE exp Gu Hv 2 .
cond v∼µv − u∼µu|v(·|v) −2| − | u∼µu −2| − |

Baptista, Stuart, Tran/Contrastive Learning 21
Expanding the quadratic forms, the objective simplifies to the form
(cid:104)1 (cid:105) (cid:104) (cid:16) 1 (cid:17)(cid:105)
L (θ;2,0) = E Gu 2 u,G⊤Hv +E logE exp Gu 2+ u,G⊤Hv ,
cond (u,v)∼µ 2| | −⟨ ⟩ v∼µv u∼µu −2| | ⟨ ⟩
which depends only on the matrix products A = G⊤H and B = G⊤G. Thus, we can consider
optimization over parameters (A,B) and hence define
(cid:104)1 (cid:105) (cid:104) (cid:16) 1 (cid:17)(cid:105)
L (A,B;2,0) := E u,Bu u,Av +E logE exp u,Bu + u,Av .
cond (u,v)∼µ 2⟨ ⟩−⟨ ⟩ v∼µv u∼µu −2⟨ ⟩ ⟨ ⟩
Recall, again, that in Subsection 1.5 we introduce the following notation: for any matrix Σ, Σ
r
denotes the rank r truncation of its singular value decomposition. The following theorem presents a
−
closed form for the optimal matrix pair (A,B) without and with rank constraints arising from using
an embedding dimension n < min(n ,n ). The proof may be found in Appendix D.
e u v
Theorem 5.3. The minimizer of L (A,B;2,0) over all matrices A of size n n and matrices
cond u v
×
B of size n n is
u u
×
A∗ = −1 −1 (49a)
Cu|vC uv Cvv
B∗ = −1 −1 −1. (49b)
CuuC uv Cv|uC vu Cuu
The minimizer of L (A,B;2,0) over the rank-constrained sets of matrices = A Rnu×nv :
cond r
A { ∈
rank(A) r and = B Rnu×nu : rank(B) r for 0 < r min(n ,n ) is given by
r u v
≤ } B { ∈ ≤ } ≤
A∗(r) = (B∗(r)+ −1)1/2((B∗(r)+ −1)1/2 −1/2) −1/2, (50a)
Cuu Cuu C uv Cvv r Cvv
B∗(r) = argminTr((B+ Cu − u 1) C u|v )+log (cid:12) (cid:12)(B+ Cu − u 1) C u|v (cid:12) (cid:12) + (50b)
B∈Br
(cid:13)(cid:16) (cid:17) (cid:13)2
(cid:13) (B+ −1)1/2 −1/2 (B+ −1)1/2 −1/2(cid:13) .
(cid:13) Cuu C uv Cvv r − Cuu C uv Cvv (cid:13) F
Corollary 5.4. If the parameters (A,B) in the learnable model ν given by (46) are chosen to
be (A∗,B∗), the minimizers of L (A,B;2,0), then the conditional distribution ν , which only
cond u|v
depends on the parameters (A,B), is given by
ν (u v;A∗,B∗) = ( −1v, −1 ), (51)
u|v | N C uv Cvv C uu −C uv Cvv C vu
which exactly matches the conditional of the data distribution µ .
u|v
The proof establishing this corollary is contained in Appendix D. See Remark D.3 for discussion
of how to link matrices A,B to G,H.
5.3. Cosine Distance: Joint Loss
In this subsection we return to the proposed model form of learnable measure ν given by (39), but
we consider the Optimization Problem 3.7; recall that the loss function appearing therein is defined
by minimizing the KL divergence between the learnable joint measure ν and the data measure µ,
rather than through conditionals. The relevant optimization problem over the parameter A = G⊤H
is to minimize the objective function
L (A) = E [ u,Av ]+logE [exp( u,Av )]. (52)
joint
−
(u,v)∼µ
⟨ ⟩
(u,v)∼µu⊗µv
⟨ ⟩
In what follows the following function, and its properties, will be useful in evaluating the properties
of minimizers of L .
joint

|     |     |     | Baptista, | Stuart, | Tran/Contrastive |     |          | Learning |            | 22  |
| --- | --- | --- | --------- | ------- | ---------------- | --- | -------- | -------- | ---------- | --- |
|     |     |     |           |         |                  |     | (cid:16) |          | 1 (cid:17) |     |
Definition 5.5. Define h: (0,1] R+ by h(σ) = σ−1 1(1+4σ2) 1 .
|     |     |     |     |     |     |     | 2   |     | 2 2 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | →   |     |     |     |     | −   |     |
We note that h(σ) [0,σ), that lim σ−1h(σ) = 1 and that h(1) = 1(√5 1) (0,1). The
|     |     |     |     | σ→0 |     |     |     |     | 2   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | ∈   |     |     |     |     |     |     | − ∈ |     |
following theorem describes the form of the minimizer of the joint loss in closed form; proof may be
found in Appendix D. Once again, recall that in Subsection 1.5 we introduce the following notation:
for any matrix Σ, Σ denotes the rank-r truncation of its singular value decomposition.
r
Theorem 5.6. Let UΣV⊤ be the singular value decomposition of −1/2 −1/2 where Σ is a
|     |     |     |     |     |     |     |     |     | uu uv vv |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- |
|     |     |     |     |     |     |     |     |     | C C C    |     |
diagonal matrix of size min(n ,n ) min(n ,n ). The minimizer of L (A) over all matrices of
|     |     |     | u   | v   | u   | v   |     |     | joint |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
×
| size n | n is given | by  |     |      |             |     |       |     |     |      |
| ------ | ---------- | --- | --- | ---- | ----------- | --- | ----- | --- | --- | ---- |
| u ×    | v          |     |     |      |             |     |       |     |     |      |
|        |            |     |     | A∗ = | −1/2Uh(Σ)V⊤ |     | −1/2, |     |     | (53) |
|        |            |     |     |      | Cuu         |     | Cvv   |     |     |      |
where h in Definition 5.5 is applied elementwise to the diagonal entries of Σ. Furthermore, the
minimizer of L (A) over the rank-constrained set of matrices = A Rnu×nv : rank(A) r
|         | joint   |         |       |       |               |     |       |       | r   |      |
| ------- | ------- | ------- | ----- | ----- | ------------- | --- | ----- | ----- | --- | ---- |
|         |         |         |       |       |               |     |       | A     | { ∈ | ≤ }  |
| for 0 < | r min(n | ,n ) is | given | by    |               |     |       |       |     |      |
|         | ≤ u     | v       |       |       |               |     |       |       |     |      |
|         |         |         |       | A∗(r) | −1/2(Uh(Σ)V⊤) |     |       | −1/2. |     |      |
|         |         |         |       | =     |               |     |       |       |     | (54) |
|         |         |         |       |       | Cuu           |     | r Cvv |       |     |      |
The following corollary shows that the approximation to the joint data distribution resulting from
Theorem 5.6 has marginal distributions that are closer to the marginals of the data distribution µ
than the approximation implied by minimizing the conditional losses in Subsection 5.1. Without loss
of generality, we study the approximation to marginal for u. An analogous result may be stated for
the marginal on v, by symmetry. The proof of the corollary can be found in Appendix D.
Corollary 5.7. Recall that the true marginal distribution µ is the centred Gaussian with covariance
u
(0, ). Consider the marginal distribution implied with the model form (39) under the (condi-
uu
| N C |     |     |     | (du;A∗ |     |     |     |     |     |     |
| --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
tional) loss function (42), denoted ν ), and under the (joint) loss function (52), denoted
|          |         |     |     | u   | cond |     |     |     |     |     |
| -------- | ------- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
| ν (du;A∗ | ). Then |     |     |     |      |     |     |     |     |     |
u
joint
|     |     | (du;A∗ |     |         |          | (cid:0) | Σ2)−1U⊤    |     | 1/2(cid:1) |       |
| --- | --- | ------ | --- | ------- | -------- | ------- | ---------- | --- | ---------- | ----- |
|     |     | ν      |     | ) =     | (0, 1/2U | I       |            |     | ,          | (55a) |
|     |     | u      |     | cond    | Cuu      | nu      |            |     | Cuu        |       |
|     |     |        |     | N       |          |         | −          |     |            |       |
|     |     | (du;A∗ |     |         | 1/2U     | (cid:0) | h(Σ)2)−1U⊤ |     | 1/2(cid:1) |       |
|     |     | ν      |     | ) =     | (0,      | I       |            |     | ,          | (55b) |
|     |     | u      |     | joint N | Cuu      | nu      | −          |     | Cuu        |       |
where U and Σ are comprised of the left singular vectors and singular values of −1/2 −1/2. In
uu uv vv
C C C
view of the stated properties of function h, this shows that minimizing the joint loss results in a
marginal distribution on u which is closer to the true marginal distribution than that obtained by
| minimizing     | the conditional |     | loss. |     |     |     |     |     |     |     |
| -------------- | --------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
| 5.4. Numerical | Illustrations   |     |       |     |     |     |     |     |     |     |
In this section, we consider a two-dimensional Gaussian target distribution µ and visualize the
approximations that result from various contrastive learning problems. The target distribution we
consider is µ = (m, ) with mean m and covariance matrix given by
|     | N   | C   |     |          |          |          |       | C        |     |     |
| --- | --- | --- | --- | -------- | -------- | -------- | ----- | -------- | --- | --- |
|     |     |     |     | (cid:20) | (cid:21) | (cid:20) |       | (cid:21) |     |     |
|     |     |     |     |          | 0        |          | 1.5   | 1        |     |     |
|     |     |     |     | m =      | ,        | =        |       | .        |     |     |
|     |     |     |     |          | 0        | C        | 1 1.5 |          |     |     |
|     | R   |     |     |          |          |          | R2    |          |     |     |
Here u,v are the two components of the vector in governed by Gaussian µ. Because of the
∈
low dimensionality of the example the latent space also has dimension n = 1. (In Section 6 we will
e
consider Gaussian numerical examples where the embedding dimension is smaller than the dimension

|     |     |     |     | Baptista, Stuart, | Tran/Contrastive |     | Learning |         |     | 23  |
| --- | --- | --- | --- | ----------------- | ---------------- | --- | -------- | ------- | --- | --- |
|     |     |     | µ   | (u) µ (v)         |                  |     |          | µ(u,v)  |     |     |
|     |     |     | u   | v                 |                  |     |          |         |     |     |
|     |     | 4   |     | ⊗                 |                  | 4   |          |         |     |     |
|     |     | 2   |     |                   |                  | 2   |          |         |     |     |
|     |     | 0   |     |                   |                  | 0   |          |         |     |     |
|     |     | v   |     |                   |                  | v   |          |         |     |     |
|     |     | 2   |     |                   |                  | 2   |          |         |     |     |
|     |     | −   |     |                   |                  | −   |          |         |     |     |
|     |     | 4   |     |                   |                  | 4   |          |         |     |     |
|     |     | −   | 2.5 | 0.0               | 2.5              | −   | 2.5      | 0.0 2.5 |     |     |
|     |     |     | −   |                   |                  |     | −        |         |     |     |
|     |     |     |     | u                 |                  |     |          | u       |     |     |
Fig 1: Two-dimensional densities for the reference Gaussian distribution given by the product of
marginals µ µ (left) and the target Gaussian distribution µ (right). CLIP aims to learn a model
u v
⊗
that tilts the reference distribution (on the left) to match the target distribution (on the right).
of the spaces in which u and v lie.) In Figure 1 we plot the reference measure µ µ , used to define
u v
⊗
various contrastive learning models through tilting, and the target distribution µ.
First, we investigate the approximate distributions that result from the learning problems consid-
ered in Sections 5.1-5.3 for various alignment metrics and objective functions. Figures 2-4 plot the
conditional and marginal densities of the models ν( , ;θ∗) corresponding to the optimal parameter θ∗
· ·
for the learning problems arising from: (i) the cosine distance with the two-sided conditional loss; (ii)
the positive quadratic form with the one-sided conditional loss; and (iii) the cosine distance with the
joint loss, respectively. As expected from Theorem 5.1, we observe that the cosine alignment correctly
identifies the two conditional means for µ and µ in Figure 2. However, the model structure
u|v v|u
produces Gaussian conditionals whose covariances follow the marginal covariance of the reference
distribution, which have larger variance in all directions than the true conditional covariances. With
the additional parameter in the positive quadratic form, the model correctly captures both the
conditional means and variance for the u v variable in Figure 3, but the one-sided loss does not
|
accurately describe the moments for the v u variable; see Theorem 5.3. Lastly, using the cosine
|
alignment metric with the joint loss results in a closer match to the true marginal covariances of the
| data distribution, |     | as expected |     | from Theorem | 5.6. |     |     |     |     |     |
| ------------------ | --- | ----------- | --- | ------------ | ---- | --- | --- | --- | --- | --- |
InFigure5,weplottheresultingapproximationtothedatadistribution.Ascomparedtothetarget
µ, we observe that using the cosine distance with the conditional loss results in a joint distribution
with inflated variance along both variables. With the positive quadratic form, the variance for u v is
|
reduced, while the v u conditional is not correctly specified when using the one-sided loss. Lastly,
|
the joint loss results in the closest variance in both variables to the joint distribution.
Lastly we go beyond the Gaussian setting and consider the effect of normalizing the encoders.
We consider the approximation resulting from normalized encoders that are defined by parameters
|        | R   |        | R:  |             |     |         |     |     |     |     |
| ------ | --- | ------ | --- | ----------- | --- | ------- | --- | --- | --- | --- |
| θ := g | and | θ := h |     |             |     |         |     |     |     |     |
| u ∈    |     | v      | ∈   |             |     |         |     |     |     |     |
|        |     |        |     |             | gu  |         |     | hv  |     |     |
|        |     |        |     | g¯ (u;θ ) = | ,   | g¯ (v;θ | ) = | .   |     |     |
|        |     |        |     | u u         |     | v       | v   |     |     |     |
|        |     |        |     |             | gu  |         |     | hv  |     |     |
|        |     |        |     |             | | | |         | |   | |   |     |     |
S0
Note that g¯ (u;θ ),g¯ (v;θ ) := 1 . Due to the nonlinearity of g¯ ( ;θ ) and g¯ ( ;θ ) the
|     | u   | u v | v   | ∈ {± | }   |     |     | u · u | v · | v   |
| --- | --- | --- | --- | ---- | --- | --- | --- | ----- | --- | --- |
resulting learned joint and conditional distributions are non-Gaussian; this is illustrated in Figure 6.
In this case, the model captures the positive correlation of the (u,v) variables, which is observed in
the approximation of the joint distribution. The expressiveness of the normalized models tilts the

|     |     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     |     |     | 24  |
| --- | --- | --- | --- | --- | --------- | ------- | ---------------- | --- | -------- | --- | --- | --- | --- |
Truth
|     |          | 0.4   | Model   |          | 0.4   |     |     | 0.3       |         |     | 0.3       |         |     |
| --- | -------- | ----- | ------- | -------- | ----- | --- | --- | --------- | ------- | --- | --------- | ------- | --- |
|     | )0.1=vu( |       |         | )0.1=uv( |       |     |     |           |         |     |           |         |     |
|     |          | 0.3   |         |          | 0.3   |     |     |           |         |     |           |         |     |
|     |          |       |         |          |       |     |     | )u(uν 0.2 |         |     | )v(vν 0.2 |         |     |
|     |          | 0.2   |         |          | 0.2   |     |     |           |         |     |           |         |     |
|     | |        |       |         |          | |     |     |     |           |         |     |           |         |     |
|     | v        |       |         |          | u     |     |     |           |         |     |           |         |     |
|     | u |      |       |         |          | v |   |     |     | 0.1       |         |     | 0.1       |         |     |
|     | ν        | 0.1   |         | ν        | 0.1   |     |     |           |         |     |           |         |     |
|     |          | 0.0   |         |          | 0.0   |     |     | 0.0       |         |     | 0.0       |         |     |
|     |          | − 2.5 | 0.0 2.5 |          | − 2.5 | 0.0 | 2.5 | −         | 2.5 0.0 | 2.5 | − 2.5     | 0.0 2.5 |     |
|     |          |       | u       |          |       | v   |     |           | u       |     |           | v       |     |
Fig 2: Densities for the conditional distributions ν ,ν and the marginals ν ,ν resulting from
|     |     |     |     |     |     |     |     | u|v v|u |     |     | u   | v   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- |
the cosine distance with the two-sided conditional loss. The conditional means are matched, but not
| the | variance; |     | see Theorem | 5.1. |     |     |     |     |     |     |     |     |     |
| --- | --------- | --- | ----------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Truth
|     |     | 0.4 |     |     | 0.4 |     |     | 0.3 |     |     | 0.3 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Model
|     | )0.1=vu( |     |         | )0.1=uv( |       |     |     |       |         |     |       |         |     |
| --- | -------- | --- | ------- | -------- | ----- | --- | --- | ----- | ------- | --- | ----- | ------- | --- |
|     |          | 0.3 |         |          | 0.3   |     |     |       |         |     |       |         |     |
|     |          |     |         |          |       |     |     | 0.2   |         |     | 0.2   |         |     |
|     |          |     |         |          |       |     |     | )u(uν |         |     | )v(vν |         |     |
|     |          | 0.2 |         |          | 0.2   |     |     |       |         |     |       |         |     |
|     | |        |     |         |          | |     |     |     |       |         |     |       |         |     |
|     | v u |    |     |         |          | u |   |     |     | 0.1   |         |     | 0.1   |         |     |
|     | ν        | 0.1 |         | ν        | v 0.1 |     |     |       |         |     |       |         |     |
|     |          | 0.0 |         |          | 0.0   |     |     | 0.0   |         |     | 0.0   |         |     |
|     |          | 2.5 | 0.0 2.5 |          | 2.5   | 0.0 | 2.5 |       | 2.5 0.0 | 2.5 | 2.5   | 0.0 2.5 |     |
|     |          | −   | u       |          | −     | v   |     | −     | u       |     | −     | v       |     |
Fig 3: Densities for the conditional distributions ν u|v ,ν v|u and the marginals ν u ,ν v resulting from
the cosine distance with the joint loss. The mean and variance for the u v conditional is matched,
|
| but | not | for v | u; see Theorem |     | 5.3. |     |     |     |     |     |     |     |     |
| --- | --- | ----- | -------------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
|
Truth
|     |     | 0.4 |     |     | 0.4 |     |     | 0.3 |     |     | 0.3 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Model
|     | )0.1=vu( |     |         | )0.1=uv( |       |     |     |       |         |     |       |         |     |
| --- | -------- | --- | ------- | -------- | ----- | --- | --- | ----- | ------- | --- | ----- | ------- | --- |
|     |          | 0.3 |         |          | 0.3   |     |     |       |         |     |       |         |     |
|     |          |     |         |          |       |     |     | 0.2   |         |     | 0.2   |         |     |
|     |          |     |         |          |       |     |     | )u(uν |         |     | )v(vν |         |     |
|     | |        | 0.2 |         |          | | 0.2 |     |     |       |         |     |       |         |     |
|     | v |      |     |         |          | u |   |     |     | 0.1   |         |     | 0.1   |         |     |
|     | ν u      | 0.1 |         | ν        | v 0.1 |     |     |       |         |     |       |         |     |
|     |          | 0.0 |         |          | 0.0   |     |     | 0.0   |         |     | 0.0   |         |     |
|     |          | 2.5 | 0.0 2.5 |          | 2.5   | 0.0 | 2.5 |       | 2.5 0.0 | 2.5 | 2.5   | 0.0 2.5 |     |
|     |          | −   | u       |          | −     | v   |     | −     | u       |     | −     | v       |     |
Fig 4: Densities for the marginals ν ,ν and the conditional distributions ν ,ν resulting from
|     |     |     |     |     |     | u v |     |     |     |     | u|v | v|u |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the cosine distance with the joint loss. The marginal variances are better approximated, but a bias
| is  | introduced |     | in the conditional |     | means; | see | Theorem | 5.6. |     |     |     |     |     |
| --- | ---------- | --- | ------------------ | --- | ------ | --- | ------- | ---- | --- | --- | --- | --- | --- |
marginal distributions by weighting the probability mass to the left and right side of the origin. As a
result of the symmetry of the joint target distribution and the low-dimensional embedding, we note
that the marginal distributions for this example are captured exactly with the learned model, but
we do not present this result because this property is not guaranteed in arbitrary dimensions.
| 6.  | Numerical |     | Experiments |     |     |     |     |     |     |     |     |     |     |
| --- | --------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
In this section we present numerical experiments that complement the material presented in the
previous sections. Subsection 6.1 is devoted to a high dimensional Gaussian example where an exactly

|     |     |     |     |     | Baptista, |     | Stuart, Tran/Contrastive |     | Learning |     |     |     | 25  |
| --- | --- | --- | --- | --- | --------- | --- | ------------------------ | --- | -------- | --- | --- | --- | --- |
CosineDistance,ConditionalLoss PositiveQuadraticForm CosineDistance,JointLoss
|     | 4   |     |     |     |     | 4   |     |     |     | 4   |     |         |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- |
|     | 2   |     |     |     |     | 2   |     |     |     | 2   |     |         |     |
|     | v 0 |     |     |     |     | v 0 |     |     |     | v 0 |     |         |     |
|     | 2   |     |     |     |     | 2   |     |     |     | 2   |     |         |     |
|     | −   |     |     |     |     | −   |     |     |     | −   |     |         |     |
|     | 4   |     |     |     |     | 4   |     |     |     | 4   |     |         |     |
|     | −   | 2.5 | 0.0 | 2.5 |     | −   | 2.5 | 0.0 | 2.5 | −   | 2.5 | 0.0 2.5 |     |
|     |     | −   |     |     |     |     | −   |     |     |     | −   |         |     |
|     |     |     | u   |     |     |     |     | u   |     |     |     | u       |     |
Fig 5: Two-dimensional densities for the distributions ν learned with various alignment metrics
and objective functions: cosine distance with the conditional loss from Subsection 5.1 (left), the
positivequadratic formwith the conditional loss formatching theu v conditionalfrom Subsection5.2
|
(middle) and the cosine distance with the joint loss from Section 5.3 (right). These densities provide
different approximations of the joint distribution on the right of Figure 1. Unsurprisingly, the joint
| loss | provides | the | closest | approximation |     |     | to the | target distribution. |     |     |     |     |     |
| ---- | -------- | --- | ------- | ------------- | --- | --- | ------ | -------------------- | --- | --- | --- | --- | --- |
NormalizedCLIP
|     |     | 0.6 |     |     |     |     | 0.6 |     |     | 4   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Truth
Model
|     |     | )0.1=vu( |     |     |     | )0.1=uv( |     |     |     | 2   |     |     |     |
| --- | --- | -------- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
|     |     | 0.4      |     |     |     |          | 0.4 |     |     |     |     |     |     |
v 0
|     |     | |     |     |     |     | |   |     |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | v 0.2 |     |     |     | u   | 0.2 |     |     |     |     |     |     |
|     |     | u |   |     |     |     | v | |     |     |     |     |     |     |     |
|     |     | ν     |     |     |     | ν   |     |     |     | 2   |     |     |     |
−
|     |     | 0.0 |     |     |     |     | 0.0 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
4
|     |     |     | 2.5 | 0.0 | 2.5 |     | 2.5 | 0.0 2.5 |     | − 2.5 | 0.0 | 2.5 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | ----- | --- | --- | --- |
|     |     |     | −   | u   |     |     | −   | v       |     | −     | u   |     |     |
Fig 6: One-dimensional densities for the conditional distributions ν ,ν (left) and the two-
|     |     |     |     |     |     |     |     |     |     |     | u|v v|u |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
dimensional joint distribution ν (right) that is learned with the cosine metric and the normalized
| encoder |     | models. |     |     |     |     |     |     |     |     |     |     |     |
| ------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
computable true joint distribution may be used to evaluate sensitivity of CLIP to parameters such
as embedding dimension, size of the data set and batch size used in training. In Subsection 6.2 we
show how classification with the MNIST data set may be formulated as a specific generalization of
the basic contrastive learning set-up, by changing the embeddings and working with a one-sided loss.
Subsection 6.3 demonstrates how a problem in Lagrangian data assimilation may be addressed by
| use  | of contrastive   |     | learning. |     |          |     |     |     |     |     |     |     |     |
| ---- | ---------------- | --- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1. | High-dimensional |     |           |     | Gaussian |     |     |     |     |     |     |     |     |
In this section we consider a pair of data modalities that are different linear projections of a Gaussian
process, and hence themselves follow a multivariate joint distribution. As in Subsection 5.1, the
conditionals of this distribution are known analytically, allowing us to evaluate the learned model
L2(D;R).
approximations. Let D = (0,1) and C be a trace-class covariance operator on :=
W
Denote the eigenpairs of C by (ψ j ,λ j ) with non-negative eigenvalues ordered to be decreasing
in j 1,2,3,... . From this we may define draws from w (0,C) via the Karhunen-Loéve
|     | ∈   | {   | }   |     |     |     |     |     |     | ∼ N |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Baptista, Stuart, Tran/Contrastive Learning 26
w conditioned on v u conditioned on v v conditioned on u
ConditionalMean 2
0.050
0.050 95%ConfidenceInterval
1
0.025
0.025
0.000 0.000 0
0.025 0.025 1
− − −
0.050 0.050
− 0.00 0.25 0.50 0.75 1.00 − 0.00 0.25 0.50 0.75 1.00 1 2 3 4 5
x x Index
Fig 7: Conditional distributions for the Gaussian process w (left) and the noisy observations u
(middle) conditioned on a realization of the coefficients v as well as the conditional distribution for v
conditioned on a realization of the noisy observation u (right). For each conditional, we plot 400
samples from the conditional distribution, the conditional mean and a 95% confidence interval.
decomposition
∞
(cid:88)(cid:112)
w(x) = λ ξ ψ (x). (56)
j j j
j=1
Let ∆ denote the Laplacian equipped with homogeneous Neumann boundary conditions on D and
viewedasactingonfunctionsinH2(D;R)withmeanzerooverD.WeassumethatC = ( ∆+τ2I)−α.
−
Then, for all positive integers j, ψ (x) = cos(jπx) and λ = (j2π2+τ2)−α, where τ represents an
j j
inverse length scale and α defines the regularity of the process. We note the index in (56) starts
at j = 1 to ensure that w has zero mean when integrated over D. In our experiments we set the
hyper-parameters to τ = 3 and α = 2.
The two data modalitiesare linear projectionsof the Gaussian processgiven bythe noisy pointwise
evaluation of w at n uniformly-spaced grid locations (x ,...,x ) D and the first n random
u 1 nu
∈
v
elements in its orthonormal basis. That is,
u = (w(x ),...,w(x ))+η := Rnu, (57a)
1 nu u
∈ U
(cid:32) (cid:33)
(cid:90) (cid:90)
1 1
v = w(x)ψ (x)dx,..., w(x)ψ (x)dx := Rnv, (57b)
√λ
1 (cid:112)
λ
nv
∈ V
u nv
where η (0,σ2I ). In our experiments we set n = 12 and n = 5, to emphasize an asymmetry
u
∼ N
nu u v
that arises in practice between the information content of the two modalities, and employ noise
standard deviation σ = 0.05. We consider the true random process based on a truncation of the
Karhunen-Loéve expansion (56) to 1000 modes. Figure 7 displays 400 independent realizations
of the underlying true process w, conditional on one realization of v, as well as the conditional
distribution for each of the two data modalities, µ ( v) and µ ( u), given one realization of v
u|v v|u
·| ·|
and u, respectively. For the linear transformations of the Gaussian process w in (57), the conditional
distributions of each modality are also Gaussian. Thus, the mean and covariance of these conditionals
are computable in closed-form using the expressions in (37) and hence the mean and confidence
intervals capturing 95% of the conditional probability mass are also plotted in Figure 7.
To define a model for the joint distribution µ(u,v) as a tilting of the marginals µ µ , we choose
u v
⊗
linear encoders with the form in (38). The encoders are not normalized, in order to exactly capture
the true Gaussian conditional expectations for each modality, as described in Section 5. The encoders

|     |     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     |     |     | 27  |
| --- | --- | --- | --- | --- | --------- | ------- | ---------------- | --- | -------- | --- | --- | --- | --- |
are learned using the cosine alignment metric with the two-sided conditional loss. We approximate
the loss using N i.i.d.samples from the joint distribution µ, which is estimated at every step of
the optimizer with a small batch of samples to build the empirical loss function as in Remark 2.6.
The loss is minimized using the Adam optimizer with a learning rate of 10−4 over 500 epochs. In
this subsection, we study the effect of the choice of embedding dimension on the approximation
of the conditional expectations. We also study various practical approximations introduced in the
optimization problem including the number of data samples and the batch size.
First, we study the effect of increasing the embedding dimension of the model with N = 10,000
training samples and a fixed batch size of 512. Figure 8 plots the estimated conditional expectations
for both data modalities E[u v] and E[v u] for the same realization of the conditioning variables of v
|     |     |     |     | |   |     | |   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
and u, respectively, as in Figure 7. We observe that increasing the embedding dimension up to the
dimension n = min(n ,n ) = 5 improves the approximation to the true conditional expectations in
|     |     | e   |     | u v |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
black. As expected from Theorem 5.1, the solution does not improve for larger embedding dimensions
| with | a fixed         | total | number | of  | samples          | N and | batch | size.          |                 |                |                 |     |     |
| ---- | --------------- | ----- | ------ | --- | ---------------- | ----- | ----- | -------------- | --------------- | -------------- | --------------- | --- | --- |
|      | uconditionedonv |       |        |     |                  |       |       |                |                 |                | v conditionedon |     | u   |
|      |                 |       |        |     | v conditionedonu |       |       |                | uconditionedonv |                |                 |     |     |
|      |                 |       | ne=2   |     |                  |       |       |                |                 |                | 1.5             |     |     |
| 0.04 |                 |       |        |     |                  |       |       | 0.12           |                 |                |                 |     |     |
|      |                 |       | ne=3   |     |                  |       |       |                |                 | rorre2LderauqS |                 |     |     |
|      |                 |       | n e =  | 4   |                  |       |       | rorre2LderauqS |                 |                |                 |     |     |
| 0.02 |                 |       | n =    | 5 1 |                  |       |       | 0.10           |                 |                |                 |     |     |
e
|      |     |     | Truth |     |     |     |     |      |     |     | 1.0 |     |     |
| ---- | --- | --- | ----- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
| 0.00 |     |     |       |     |     |     |     | 0.08 |     |     |     |     |     |
0
0.06
− 0.02
0.5
| 0.04 |     |     |     | − 1 |     |     |     | 0.04 |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
−
|     | 0.0 | 0.5 |     | 1.0 | 2   |       | 4   |     | 2 4                | 6   | 2                  | 4   | 6   |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | ------------------ | --- | ------------------ | --- | --- |
|     |     | x   |     |     |     | Index |     |     | Embeddingdimension |     | Embeddingdimension |     |     |
Fig 8: Left: Conditional expectations E[u v] and E[v u] with increasing embedding dimension for
|     |     |     |     |     |     |     | |   |     | |   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
a fixed realization of v and u, respectively. Right: Squared errors in the conditional expectations
in expectation over 104 realizations of the conditioning variables. We observe convergence of the
conditional expectations toward the truth when increasing the embedding dimension.
Next, we study the effect of increasing the batch size during training with a fixed embedding
dimension of n = 5 and N = 10,000 training samples; see Figure 9. We observe that the errors for
e
both conditionals converge to zero as the batch size increases, which reduces the bias in the finite
sample estimator given in (7) as an approximation of the population objective in (11). Note that the
minimizer of the population loss yields the true conditional means; see Theorem 5.1.
uconditionedonv v conditionedonu uconditionedonv v conditionedonu
| 0.050 |     | Batchsize=128 |     | 2   |     |     |     |                |     |                |     |     |     |
| ----- | --- | ------------- | --- | --- | --- | --- | --- | -------------- | --- | -------------- | --- | --- | --- |
|       |     | Batchsize=256 |     |     |     |     |     | rorre2LderauqS |     | rorre2LderauqS | 0.5 |     |     |
Batchsize=512
| 0.025 |     | Batchsize=1024 |     | 1   |     |     |     | 0.06 |     |     |     |     |     |
| ----- | --- | -------------- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
|       |     | Truth          |     |     |     |     |     |      |     |     | 0.4 |     |     |
0.000
0
0.04
| 0.025 |     |     |     |     |     |     |     |     |     |     | 0.3 |     |     |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
−
1
−
|     | 0.0 | 0.5 |     | 1.0 | 2   |       | 4   |     | 500       | 1000 |     | 500       | 1000 |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --------- | ---- | --- | --------- | ---- |
|     |     | x   |     |     |     | Index |     |     | Batchsize |      |     | Batchsize |      |
Fig 9: Conditional expectations with increasing batch size. We observe that the conditional expecta-
| tions | converge | toward |     | the truth | when | increasing |     | the | batch size. |     |     |     |     |
| ----- | -------- | ------ | --- | --------- | ---- | ---------- | --- | --- | ----------- | --- | --- | --- | --- |
Lastly, we study the effect of increasing the training sample size N in Figure 10 with a fixed
batch size of 512 and embedding dimension set to n = 5. We observe similar convergence of
e

Baptista, Stuart, Tran/Contrastive Learning 28
the expectations to the true expectations for u and v in expectation over 104 realizations of the
conditioning variables.
uconditionedonv v conditionedonu
2
0.050 N=2000 0.06
N=5000
N=10000
0.025 N=20000 1
N=50000
N=100000 0.05
0.000 Truth 0
0.025
− 1 0.04
−
0.0 0.5 1.0 2 4 103 104 105
x Index Samplesize,N
rorre2LderauqS
uconditionedon v
1.0
0.8
0.6
0.4
103 104 105
Samplesize,N
rorre2LderauqS
v conditionedonu
Fig 10: Left: Conditional expectations E[u v] and E[v u] with increasing sample size for a fixed
| |
realization of v and u, respectively. Conditional expectations with increasing sample size. Right:
Squared errors in the conditional expectations in expectation over 104 realizations of the conditioning
variables. We observe convergence of the conditional expectations toward the truth when increasing
the sample size.
6.2. The MNIST Data Set
In this experiment, we consider our two data modalities to be images of MNIST digits u =
∈ U
[0,1]28×28 and their corresponding labels v = 0,...,9 determined by the prescribed labelling
∈ V { }
of the dataset. Our goal is to identify encoders g and g , acting on and respectively, to jointly
u v
U V
characterize the conditional distributions of images given labels µ and labels given images µ .
u|v v|u
We note that while µ ( v) is supported on a continuous space for each v, µ ( u) is a categorical
u|v v|u
·| ·|
distribution for each u.
In order to relate representation learning to a classification problem, we choose the image encoder
g to be a feed-forward neural network whose outputs represent the log-probabilities associated to
u
each label in ; and we choose g (v) = e to be the one-hot encoding of the label v. In this case,
v v
V
the embedding space dimension of the two encoders equals the number of labels, i.e., n = . In
e
|V|
particular, we use a LeNet architecture for g : R|V| consisting of convolution and linear layers
u
U →
with ReLu activation functions and set θ as the weights and biases of these layers. In a classification
u
task, the learned map g is used to predict a label for each input image u, which is given by the
u
output g (u) R|V| with the largest log-probability. What we have just outlined is exactly the
u
∈
architecture used in the seminal paper [22]. In that paper an optimization problem is defined by
minimizing the cross-entropy loss function L ( ).
mnist
·
Proposition 6.1. Minimizing L (θ) with respect to θ is equivalent to minimizing the one-sided
mnist
conditional loss J (θ;2,0) using the unnormalized embedding g and the one-hot encoding g as
cond,D u v
defined in the preceding paragraph.
Next, we show the results of learning the model parameters with three choices of loss functions
between conditionals. First, we consider the standard classification methodology leading to L (θ),
mnist
which (by the preceding proposition) corresponds to minimizing the one-sided conditional loss
J (θ;2,0). This in turn is equivalent to minimizing the loss function L (θ;2,0) given by
cond,D cond,D
(cid:104) (cid:105) (cid:104) (cid:105)
L (θ;2,0) = E g (u;θ ),g (v) +E logE exp( g (u;θ ),g (v) ) ,
cond,D
−
(u,v)∼µ
⟨
u u v
⟩
v∼µv u∼µu
⟨
u u v
⟩
θ∗ = argminL (θ;2,0).
cond,D
θ∈Rp

|     |     | Baptista,             | Stuart, | Tran/Contrastive |                            | Learning |            |                 |     | 29  |
| --- | --- | --------------------- | ------- | ---------------- | -------------------------- | -------- | ---------- | --------------- | --- | --- |
|     |     | Two-SidedConditionals |         |                  | One-Sided: LabelGivenImage |          | One-Sided: | ImageGivenLabel |     |     |
|     |     | 1.00                  |         |                  | 1.00                       |          |            |                 |     |     |
0.8
|     |     | 0.75 |     |     | 0.75 |     |     |     |     |     |
| --- | --- | ---- | --- | --- | ---- | --- | --- | --- | --- | --- |
0.6
|     |     | 0.50       |         |       | 0.50         |           | 0.4   |             |         |     |
| --- | --- | ---------- | ------- | ----- | ------------ | --------- | ----- | ----------- | ------- | --- |
|     |     | 0.25       |         |       | 0.25         |           | 0.2   |             |         |     |
|     |     | 0.00 0 1 2 | 3 4 5 6 | 7 8 9 | 0.00 0 1 2 3 | 4 5 6 7 8 | 9 0.0 | 0 1 2 3 4 5 | 6 7 8 9 |     |
|     |     | 1.00       |         |       | 1.00         |           | 1.00  |             |         |     |
|     |     | 0.75       |         |       | 0.75         |           | 0.75  |             |         |     |
|     |     | 0.50       |         |       | 0.50         |           | 0.50  |             |         |     |
|     |     | 0.25       |         |       | 0.25         |           | 0.25  |             |         |     |
|     |     | 0.00       |         |       | 0.00         |           | 0.00  |             |         |     |
|     |     | 0 1 2      | 3 4 5 6 | 7 8 9 | 0 1 2 3      | 4 5 6 7 8 | 9     | 0 1 2 3 4 5 | 6 7 8 9 |     |
|     |     |            |         |       | 1.00         |           | 1.00  |             |         |     |
0.8
|     |     |     |     |     | 0.75 |     | 0.75 |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | ---- | --- | --- | --- |
0.6
|     |     |     |     |     | 0.50 |     | 0.50 |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | ---- | --- | --- | --- |
0.4
|     |     | 0.2   |         |       | 0.25    |           | 0.25 |             |         |     |
| --- | --- | ----- | ------- | ----- | ------- | --------- | ---- | ----------- | ------- | --- |
|     |     | 0.0   |         |       | 0.00    |           | 0.00 |             |         |     |
|     |     | 0 1 2 | 3 4 5 6 | 7 8 9 | 0 1 2 3 | 4 5 6 7 8 | 9    | 0 1 2 3 4 5 | 6 7 8 9 |     |
Fig 11: Predicted probabilities of the image encoder g relative to the true label for three candidate
u
| images (rows) | from the | test set using | three | loss | functions. |     |     |     |     |     |
| ------------- | -------- | -------------- | ----- | ---- | ---------- | --- | --- | --- | --- | --- |
This loss seeks the encoder parameters θ to best match the conditional distribution for labels
u
given a candidate image µ . Secondly, we consider minimizing the loss function L (θ;0,2) that
|     |     | v|u |     |     |     |     |     |     | cond,D |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
matches the conditional distribution µ u|v for images given a label. And finally we consider the loss
function that aims to match both conditionals, L (θ;1,1). In all three cases we use unnormalized
cond,D
encoders as defined in the paragraph preceding the proposition. We learn the model parameters with
N = 60,000 training samples of paired images and labels. We learn the model parameters using the
Adam optimizer by training for 300 epochs with a batch size of 512 and a learning rate of 10−4.
First, we compare the approximate models for the conditional distribution of labels given images
ν . Figure 11 plots the results of mapping three images to a label using the three different learned
v|u
models. We observe that the one-sided conditional loss on label given image is most accurate for
predicting the true label, while the two other losses yield predicted distributions with non-zero
probabilities on potential labels (for example 8 as well as 3 for the first digit) that are consistent
| with the image | from subjective | interpretations. |     |     |     |     |     |     |     |     |
| -------------- | --------------- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Second, we compare the approximate models for the conditional distribution of images given each
label, namely ν . Given an empirical marginal distribution over N images µN, the model in (28)
|     | u|v |     |     |     |     |     |     | u   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
defines a weighted empirical distribution νN ( v;θ ) over the N images with the weights given by the
|     |     |     |     | u|v | u   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
·|
normalized exponential of the cosine similarities g (u;θ ),g (v) for a given label v. Figure 12 plots
|     |     |     |     |     | u u | v   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     | ⟨   | ⟩   |     |     |     |     |
16 images sampled with replacement from the conditional probability distribution of the learned
model for each loss function for the label v = 7 with N = 10,000 test samples. We note that the
test
test set consists of images for all 10 digits. While all sampled images are correctly associated with
the prescribed label, we observe that only the distribution learned using the two-sided conditional
loss (left plot) and the one-sided conditional loss for images given label (right plot) show a diverse
set of images with multiple styles. Instead, the model learned with the one-sided loss for labels given
image, that is more accurate at predicting class probabilities in Figure 11, displays many repeated
images of the digit 7. This shows that the conditional distribution, with this choice of loss function,
has its probability mass concentrated on a single image; such a distribution does not accurately

|     |     |     | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     | 30  |
| --- | --- | --- | --------- | ------- | ---------------- | --- | -------- | --- | --- |
Two-SidedConditionals One-Sided: LabelGivenImage One-Sided: ImageGivenLabel
Fig 12: Sixteen images sampled from the MNIST test dataset according to the weights given by the
cosine similarity between the learned image encoder for three different loss functions and the one-hot
encoding of the label v = 7. We observe that the encoders learned with the two-sided conditional
loss (left) and the loss for image given labels (right) demonstrate a diverse set of images, while
the encoder learned using the one-sided loss for label given image (middle) does not capture the
| distribution | well, instead | concentrating |     | on  | a single | digit. |     |     |     |
| ------------ | ------------- | ------------- | --- | --- | -------- | ------ | --- | --- | --- |
capture the true data distribution consisting of different images that are consistent with the label.
| 6.3. Lagrangian | Data | Assimilation |     |     |     |     |     |     |     |
| --------------- | ---- | ------------ | --- | --- | --- | --- | --- | --- | --- |
In this subsection we study the problem of recovering an Eulerian velocity field from Lagrangian
fluid flow data [21], introduced in Subsection 1.2. We generalize the simplified setting described
there in two ways: (i) we work with a time-dependent divergence free velocity, expressible as the
skew-gradient of a potential (streamfunction); (ii) we assume that the Eulerian data is already
encoded via the coefficients of an expansion of the potential in a set of time-oscillating, and spatially
| Fourier, | modes. |     |     |     |     |     |     |     |     |
| -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
Given potential ψ: T2 [0,T] R we may define a time-dependent incompressible velocity field
× →
| w C1(T2 | [0,T],R2) | as  | follows: |     |     |     |          |          |     |
| ------- | --------- | --- | -------- | --- | --- | --- | -------- | -------- | --- |
| ∈       | ×         |     |          |     |     |     |          |          |     |
|         |           |     |          |     |     |     | (cid:20) | (cid:21) |     |
0 1
|         |                 |     | w(x,t)   | = J | ψ(x,t), |     | J = − | .   | (58) |
| ------- | --------------- | --- | -------- | --- | ------- | --- | ----- | --- | ---- |
|         |                 |     |          | ∇   |         |     | 1     | 0   |      |
| We work | with potentials | in  | the form |     |         |     |       |     |      |
K
(cid:88)
|     |     |     | ψ(x,t) | =   |     | ψ exp(iω | t)e (x) |     | (59) |
| --- | --- | --- | ------ | --- | --- | -------- | ------- | --- | ---- |
|     |     |     |        |     |     | k        | k k     |     |      |
k=1
where the e K denote a finite collection of Fourier modes and the ω K denote a set of
|     | k }k=1 |     |     |     |     |     |     | k }k=1 |     |
| --- | ------ | --- | --- | --- | --- | --- | --- | ------ | --- |
|     | {      |     |     |     |     |     |     | {      |     |
temporal frequencies, chosen at random. We identify the complex numbers ψ K with a vector
k }k=1
| R2K |     |     |     |     |     |     |     | {   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
in and view this as a prescribed embedding of the Eulerian observations of the velocity field:
g (u) = ψ K ; we will learn only the embedding of the Lagrangian data.
| u   | k   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{ }k=1
To define the Lagrangian data we integrate the velocity field in-time starting from the initial
condition x(0) = (0.75,0.75) to define a trajectory x(t). In our experiment, we use a fourth-order
10−5
Runge-Kutta method with time-step ∆t = and integrate up to T = 0.1 The trajectory
position is recorded every 10 time steps to produce a data sequence v R2×Jv given by the vectors
∈
v(j) = x(∆t10j) R2 for j = 1,...,J . In our experiment we have J = 1000 observation times
|     |     |     |     | v   |     |     |     | v   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∈

|     |         |     | Baptista, Stuart, | Tran/Contrastive | Learning    |         | 31  |
| --- | ------- | --- | ----------------- | ---------------- | ----------- | ------- | --- |
|     | 1.0     |     |                   | 1.0              |             |         |     |
|     | 0.8     |     |                   | 0.8              |             |         |     |
|     | 0.6     |     |                   | 0.6              |             |         |     |
|     | 0.4     |     |                   | 0.4              |             |         |     |
|     | 0.2     |     |                   | 0.2              |             |         |     |
|     | 0.0     |     |                   | 0.0              |             |         |     |
|     | 0.0 0.2 | 0.4 | 0.6 0.8           | 1.0 0.0          | 0.2 0.4 0.6 | 0.8 1.0 |     |
Fig 13: Visualization of two paired samples of the Eulerian potential field and Lagrangian trajectory.
and K = 49 coefficients. Figure 13 visualizes two potential fields (frozen at time t = 0) and their
| associated | trajectories | in the two-dimensional |     | periodic domain | [0,1]2. |     |     |
| ---------- | ------------ | ---------------------- | --- | --------------- | ------- | --- | --- |
We use a text transformer to define the encoder for the trajectory g (v). Similarly to text, the
v
transformer follows the architecture of a typical text encoder that processes the values of the
trajectory at each time-step as a separate token. More precisely, the encoder is given by the following
composition
|     |     |     | g (v) = | P A A  | L(v), |     |     |
| --- | --- | --- | ------- | ------ | ----- | --- | --- |
|     |     |     | v       | M      | 1     |     |     |
|     |     |     |         | ◦ ◦··· | ◦     |     |     |
Rd Rd′
where L: is an initial lifting layer acting on each sample along the trajectory of length
|     | RJv×d →′ RJv×d′ |     |     |     |     |     |     |
| --- | --------------- | --- | --- | --- | --- | --- | --- |
J , A : is a residual attention block (consisting of a multi-head attention, layer
| v k | →   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
normalization a nd a multi-layer perception network), and P: RJv×d′ Rne is a pooling operator
→
that extracts the last element (i.e., token) of the ensemble and projects it into the embedding
dimension. In this example, we select n = 2K to embed the Lagrangian data into the space of
e
| coefficients | for the | potential and | we use M | = 12 layers. |     |     |     |
| ------------ | ------- | ------------- | -------- | ------------ | --- | --- | --- |
In our experiments, we minimize the training loss L on the two conditionals using the Adam
cond
optimizer with mini-batches of size 128, a learning rate set to 10−4 and using 20 epochs that each
see N = 32,268 training samples. For this study we evaluate the retrieval capabilities of the learned
embeddings on a test set of the same size comprising a total of N = 32,268 paired samples of
test
coefficients and trajectories. Figure 14 plots the retrieval accuracies (normalized to 1) on the training
and test sets across the training iterations. In particular, we evaluate the accuracy of retrieving
the coefficients from Lagrangian trajectories in the top row and retrieving the trajectories from the
coefficients in the bottom row. For a set of samples from a given modality, the retrieval accuracy is
computed by evaluating the fraction of inputs for which the true paired sample of the other modality
is the top retrieved sample according to the learned cosine similarities (R@1 left) or within the
retrieved samples with the top five similarities (R@5 right). We observe that the model shows an
improvement in mapping potential functions to trajectories and trajectories to potential functions
across training, approaching 100% accuracy for R@5 retrieval. We also observe that the retrieval
accuracy is generally higher for recovering trajectories from potentials, which is expected given this
mapping is well-defined by (58) and (59). Lastly, Figure 15 shows an example of the more challenging
retrieval problem of recovering a velocity field potential from a Lagrangian trajectory.

|     |            |        |            | Baptista, | Stuart, Tran/Contrastive |            | Learning |            |           |      | 32  |
| --- | ---------- | ------ | ---------- | --------- | ------------------------ | ---------- | -------- | ---------- | --------- | ---- | --- |
|     | 1.0        |        |            |           |                          | 1.0        |          |            |           |      |     |
|     | 1@R        |        |            |           |                          | 5@R        |          |            |           |      |     |
|     | 0.8        |        |            |           |                          | 0.8        |          |            |           |      |     |
|     | -          |        |            |           |                          | -          |          |            |           |      |     |
|     | yrotcejarT |        |            |           |                          | yrotcejarT |          |            |           |      |     |
|     | 0.6        |        |            |           |                          | 0.6        |          |            |           |      |     |
|     | 0.4        |        |            |           |                          | 0.4        |          |            |           |      |     |
|     | ot         |        |            |           |                          | ot         |          |            |           |      |     |
|     | laitnetoP  |        |            |           |                          | laitnetoP  |          |            |           |      |     |
|     | 0.2        |        |            | Training  | set                      | 0.2        |          |            |           |      |     |
|     |            |        |            | Test      | set                      |            |          |            |           |      |     |
|     | 0.0        |        |            |           |                          | 0.0        |          |            |           |      |     |
|     |            | 0 2000 | 4000       | 6000      | 8000                     | 0          |          | 2000       | 4000 6000 | 8000 |     |
|     |            |        | Iterations |           |                          |            |          | Iterations |           |      |     |
|     | 1.0        |        |            |           |                          | 1.0        |          |            |           |      |     |
|     | 1@R        |        |            |           |                          | 5@R        |          |            |           |      |     |
|     | 0.8        |        |            |           |                          | 0.8        |          |            |           |      |     |
|     | -          |        |            |           |                          | -          |          |            |           |      |     |
|     | laitnetoP  |        |            |           |                          | laitnetoP  |          |            |           |      |     |
|     | 0.6        |        |            |           |                          | 0.6        |          |            |           |      |     |
|     | ot         |        |            |           |                          | ot         |          |            |           |      |     |
|     | 0.4        |        |            |           |                          | 0.4        |          |            |           |      |     |
|     | yrotcejarT |        |            |           |                          | yrotcejarT |          |            |           |      |     |
|     | 0.2        |        |            |           |                          | 0.2        |          |            |           |      |     |
|     | 0.0        |        |            |           |                          | 0.0        |          |            |           |      |     |
|     |            | 0 2000 | 4000       | 6000      | 8000                     | 0          |          | 2000       | 4000 6000 | 8000 |     |
|     |            |        | Iterations |           |                          |            |          | Iterations |           |      |     |
Fig 14: Improvement in retrieval accuracy during training of the encoders for the Lagrangian data
assimilation problem. We compute the retrieval accuracies from coefficients to trajectories (top) and
trajectories to coefficients (bottom) for both top retrieval (left) and top 5 retrieved items (right),
| which | shows | improvement           | on both | the           | training and | test | data.                  |            |             |     |     |
| ----- | ----- | --------------------- | ------- | ------------- | ------------ | ---- | ---------------------- | ---------- | ----------- | --- | --- |
|       |       | Lagrangian trajectory |         |               |              |      | Retrieved field at t=0 |            |             |     |     |
|       |       | 1.0                   |         |               |              | 1.0  |                        |            |             |     |     |
|       |       | 0.8                   |         |               |              | 0.8  |                        |            |             |     |     |
|       |       | 0.6                   |         |               |              | 0.6  |                        |            |             |     |     |
|       |       | 0.4                   |         |               |              | 0.4  |                        |            |             |     |     |
|       |       | 0.2                   |         |               |              | 0.2  |                        |            |             |     |     |
|       |       | 0.0                   |         |               |              | 0.0  |                        |            |             |     |     |
|       |       | 0.0 0.2               | 0.4     | 0.6           | 0.8 1.0      | 0.0  | 0.2                    | 0.4        | 0.6 0.8     | 1.0 |     |
|       |       | Fig 15: Retrieval     |         | of a Eulerian | potential    | from | a                      | Lagrangian | trajectory. |     |     |

|               |     |             |     | Baptista, | Stuart, | Tran/Contrastive |     | Learning |     |     | 33  |
| ------------- | --- | ----------- | --- | --------- | ------- | ---------------- | --- | -------- | --- | --- | --- |
| 7. Discussion |     | and Outlook |     |           |         |                  |     |          |     |     |     |
This work presents a mathematical framework for contrastive learning, introducing several general-
izations of the standard approach through novel alignment metrics and probabilistic loss functions.
By analyzing Gaussian models, we develop theory to elucidate the relative merits of the different
variants on the standard methodology. Our numerical experiments support and extend the theoreti-
cal findings, demonstrating their relevance beyond the Gaussian setting. These experiments also
demonstrate connections to image classification tasks and suggest broader applicability in scientific
and engineering domains. Future directions include extending the framework to handle problems
involving more than two modalities, leveraging the flexibility of the proposed methodological variants,
utilizing the probabilistic insights emphasized in our formulation, and further developing applications
| in science and | engineering. |     |     |     |     |     |     |     |     |     |     |
| -------------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Acknowledgments
The work of AMS on Lagrangian data assimilation is supported by a Department of Defense (DoD)
Vannevar Bush Faculty Fellowship (award N00014-22-1-2790). The authors thank Niall Siegenheim
for his assistance in creating the Lagrangian dataset used in this study.
References
| Bengio,  | Y., | Goodfellow, |     | I., Courville, |     | A.     |         |               |              |            |     |
| -------- | --- | ----------- | --- | -------------- | --- | ------ | ------- | ------------- | ------------ | ---------- | --- |
| [1]      |     |             |     |                |     | et al. | (2017). | Deep Learning | 1. MIT press | Cambridge, |     |
| MA, USA. |     |             |     |                |     |        |         |               |              |            |     |
[2] Bennett, A. F. (1992). Inverse methods in physical oceanography. Cambridge university press.
| Carlsson, | M.  |     |     |     |     |     |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[3] (2021). von Neumann’s trace inequality for Hilbert–Schmidt operators. Expositiones
| Mathematicae |     | 39 149–157. |     |     |     |     |     |     |     |     |     |
| ------------ | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Chen, C., Zhang, J., Xu, Y., Chen, L., Duan, J., Chen, Y., Tran, S., Zeng, B. Chilimbi, T.
| [4] |     |     |     |     |     |     |     |     | and |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(2022). Why do we need large batchsizes in contrastive learning? A gradient-bias perspective. Advances
| in Neural | Information    |     | Processing | Systems  |     | 35 33860–33875. |     |                  |           |                 |     |
| --------- | -------------- | --- | ---------- | -------- | --- | --------------- | --- | ---------------- | --------- | --------------- | --- |
| Chen,     | T., Kornblith, |     | S.,        | Norouzi, | M.  | Hinton,         | G.  |                  |           |                 |     |
| [5]       |                |     |            |          |     | and             |     | (2020). A simple | framework | for contrastive |     |
learning of visual representations. In International Conference on Machine Learning 1597–1607. PMLR.
| Chen, | X.  | He, K. |     |     |     |     |     |     |     |     |     |
| ----- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[6] and (2021). Exploring simple Siamese representation learning. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition 15750–15758.
[7] Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuh-
| mann, | C., Schmidt, |     | L.  | Jitsev, | J.  |     |     |     |     |     |     |
| ----- | ------------ | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
and (2023). Reproducible Scaling Laws for Contrastive Language-
Image Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
| tion (CVPR) |     | 2818–2829. |     |     |     |     |     |     |     |     |     |
| ----------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[8] Chopra, S., Hadsell, R. and LeCun, Y. (2005). Learning a similarity metric discriminatively, with
application to face verification. In 2005 IEEE computer society conference on computer vision and
| pattern | recognition | (CVPR’05) |     | 1 539–546. |     | IEEE. |     |     |     |     |     |
| ------- | ----------- | --------- | --- | ---------- | --- | ----- | --- | --- | --- | --- | --- |
[9] Cotter, S. L., Dashti, M., Robinson, J. C. and Stuart, A. M. (2009). Bayesian inverse problems
for functions and applications to fluid mechanics. Inverse Problems 25 115008.
| Elizalde, | B., | Deshmukh, |     | S., Al | Ismail, | M.  | Wang, | H.      |               |                |     |
| --------- | --- | --------- | --- | ------ | ------- | --- | ----- | ------- | ------------- | -------------- | --- |
| [10]      |     |           |     |        |         | and |       | (2023). | Clap learning | audio concepts |     |
from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics,
| Speech | and Signal | Processing |     | (ICASSP) |     | 1–5. IEEE. |     |     |     |     |     |
| ------ | ---------- | ---------- | --- | -------- | --- | ---------- | --- | --- | --- | --- | --- |
[11] Fang, H., Xiong, P., Xu, L. and Chen, Y. (2021). Clip2video: Mastering video-text retrieval via
| image clip. | arXiv:2106.11097. |     |           |     |     |     |     |     |     |     |     |
| ----------- | ----------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
| Friedland,  |                   | S.  | Torokhti, | A.  |     |     |     |     |     |     |     |
[12] and (2007). Generalized rank-constrained matrix approximations. SIAM
| Journal | on Matrix | Analysis |     | and Applications |     | 29 656–659. |     |     |     |     |     |
| ------- | --------- | -------- | --- | ---------------- | --- | ----------- | --- | --- | --- | --- | --- |
| Gage,   | P.        |          |     |                  |     |             |     |     |     |     |     |
[13] (1994). A new algorithm for data compression. The C Users Journal 12 23–38.
[14] Giuseppe Carere, H. C. L. (2024). Optimal low-rank approximations of posteriors for linear Gaussian
| inverse | problems | on  | Hilbert | spaces. | arXiv:2411.01112. |     |     |     |     |     |     |
| ------- | -------- | --- | ------- | ------- | ----------------- | --- | --- | --- | --- | --- | --- |

|     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     |     | Learning |     |     |     | 34  |
| --- | --- | --- | --- | --------- | ------- | ---------------- | --- | --- | -------- | --- | --- | --- | --- |
[15] Grill, J.-B.,Strub, F.,Altché, F.,Tallec, C.,Richemond, P.,Buchatskaya, E.,Doersch, C.,
Avila Pires, B., Guo, Z., Gheshlaghi Azar, M. et al. (2020). Bootstrap your own latent-a new
approach to self-supervised learning. Advances in Neural Information Processing Systems 33 21271–
21284.
| HaoChen, | J.  | Z., Wei, | C., | Gaidon, |     | A.  | Ma, | T.      |          |            |     |                     |     |
| -------- | --- | -------- | --- | ------- | --- | --- | --- | ------- | -------- | ---------- | --- | ------------------- | --- |
| [16]     |     |          |     |         |     | and |     | (2021). | Provable | guarantees |     | for self-supervised |     |
deep learning with spectral contrastive loss. Advances in Neural Information Processing Systems 34
5000–5011.
| Ide, K., | Kuznetsov, |     | L.  | Jones, |     | C. K. |     |     |     |     |     |     |     |
| -------- | ---------- | --- | --- | ------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
[17] and (2002). Lagrangian data assimilation for point vortex
| systems. | Journal | of  | Turbulence | 3   | 053. |     |     |     |     |     |     |     |     |
| -------- | ------- | --- | ---------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z.
[18]
and Duerig, T. (2021). Scaling up visual and vision-language representation learning with noisy text
supervision. In International Conference on Machine Learning 4904–4916. PMLR.
| Joseph, | K. J., | Khan, | S., | Khan, | F.  | S.  | Balasubramanian, |     |     | V.  | N.      |         |      |
| ------- | ------ | ----- | --- | ----- | --- | --- | ---------------- | --- | --- | --- | ------- | ------- | ---- |
| [19]    |        |       |     |       |     | and |                  |     |     |     | (2021). | Towards | Open |
World Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
| Recognition | (CVPR) |     | 5830–5840. |     |     |     |     |     |     |     |     |     |     |
| ----------- | ------ | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., White-
[20]
head, S., Berg, A. C., Lo, W.-Y. et al. (2023). Segment anything. In Proceedings of the IEEE/CVF
| International | Conference |     | on  | Computer |     | Vision | 4015–4026. |     |     |     |     |     |     |
| ------------- | ---------- | --- | --- | -------- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- |
[21] Kuznetsov, L., Ide, K. and Jones, C. K. (2003). A method for assimilation of Lagrangian data.
| Monthly  | Weather      | Review  | 131         | 2247–2260. |        |       |          |            |         |                |     |                  |     |
| -------- | ------------ | ------- | ----------- | ---------- | ------ | ----- | -------- | ---------- | ------- | -------------- | --- | ---------------- | --- |
| LeCun,   | Y., Bottou,  |         | L.,         | Bengio,    | Y.     |       | Haffner, | P.         |         |                |     |                  |     |
| [22]     |              |         |             |            |        | and   |          |            | (1998). | Gradient-based |     | learning applied | to  |
| document | recognition. |         | Proceedings |            | of the | IEEE  | 86       | 2278–2324. |         |                |     |                  |     |
| Liu, H., | Li,          | C., Wu, | Q.          |            | Lee,   | Y. J. |          |            |         |                |     |                  |     |
[23] and (2024). Visual instruction tuning. Advances in Neural
| Information | Processing |     | Systems | 36. |     |     |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[24] Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N. and Li, T. (2022). Clip4clip: An empirical
study of clip for end to end video clip retrieval and captioning. Neurocomputing 508 293–304.
[25] Miech, A.,Alayrac, J.-B.,Smaira, L.,Laptev, I.,Sivic, J.andZisserman, A.(2020).End-to-end
learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF
| conference | on computer |     | vision | and | pattern | recognition |     | 9879–9889. |     |     |     |     |     |
| ---------- | ----------- | --- | ------ | --- | ------- | ----------- | --- | ---------- | --- | --- | --- | --- | --- |
[26] Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I.
Chen, M.
and (2021). Glide: Towards photorealistic image generation and editing with text-guided
| diffusion | models. | arXiv:2112.10741. |     |     |     |     |     |     |     |     |     |     |     |
| --------- | ------- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[27] Pardo, L. (2018). Statistical inference based on divergence measures. Chapman and Hall/CRC.
[28] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J. et al. (2021). Learning transferable visual models from natural
language supervision. In International Conference on Machine Learning 8748–8763. PMLR.
[29] Ramesh, A.,Dhariwal, P.,Nichol, A.,Chu, C.andChen, M.(2022).Hierarchicaltext-conditional
| image generation |     | with       | CLIP | latents. | arXiv:2204.06125 |     |            | 1   | 3.        |     |         |                 |     |
| ---------------- | --- | ---------- | ---- | -------- | ---------------- | --- | ---------- | --- | --------- | --- | ------- | --------------- | --- |
| Rombach,         | R., | Blattmann, |      | A.,      | Lorenz,          |     | D., Esser, |     | P. Ommer, |     | B.      |                 |     |
| [30]             |     |            |      |          |                  |     |            |     | and       |     | (2022). | High-resolution |     |
image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer
| vision and | pattern | recognition |     | 10684–10695. |     |     |     |     |     |     |     |     |     |
| ---------- | ------- | ----------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[31] Shi, L., Fan, J. and Yan, J. (2024). OT-CLIP: Understanding and Generalizing CLIP via Optimal
| Transport. | In International |     |     | Conference |     | on Machine |     | Learning. | PMLR. |     |     |     |     |
| ---------- | ---------------- | --- | --- | ---------- | --- | ---------- | --- | --------- | ----- | --- | --- | --- | --- |
Spantini, A., Solonen, A., Cui, T., Martin, J., Tenorio, L. Marzouk, Y.
| [32] |     |     |     |     |     |     |     |     | and |     |     | (2015). | Optimal |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- |
Low-rank Approximations of Bayesian Linear Inverse Problems. SIAM Journal on Scientific Computing
37 A2451-A2487.
[33] Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S. and Lucic, M. (2019). On mutual
| information  | maximization |     |     | for representation |     |     | learning. | arXiv:1907.13625. |     |     |     |     |     |
| ------------ | ------------ | --- | --- | ------------------ | --- | --- | --------- | ----------------- | --- | --- | --- | --- | --- |
| Von Luxburg, |              | U.  |     |                    |     |     |           |                   |     |     |     |     |     |
[34] (2007). A tutorial on spectral clustering. Statistics and computing 17 395–416.
[35] Wang, C., Chai, M., He, M., Chen, D. and Liao, J. (2022). CLIP-NeRF: Text-and-image driven
manipulation of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision
| and Pattern | Recognition |     | 3835–3844. |     |     |     |     |     |     |     |     |     |     |
| ----------- | ----------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[36] Wu, M., Zhuang, C., Mosse, M., Yamins, D. and Goodman, N. (2020). On mutual information in
contrastivelearningforvisualrepresentations.InInternational Conference on Learning Representations.

Baptista, Stuart, Tran/Contrastive Learning 35
[37] Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M. and Wu, Y. (2022). CoCa:
Contrastive Captioners are Image-Text Foundation Models. Transactions of Machine Learning Research
2022.
[38] Yue, Y., Lin, F., Yamada, K. D. and Zhang, Z. (2023). Hyperbolic contrastive learning.
arXiv:2302.01409.
[39] Zbontar, J., Jing, L., Misra, I., LeCun, Y. and Deny, S. (2021). Barlow twins: Self-supervised
learning via redundancy reduction. In International Conference on Machine Learning 12310–12320.
PMLR.
[40] Zhai, X., Mustafa, B., Kolesnikov, A. and Beyer, L. (2023). Sigmoid loss for language image pre-
training. In Proceedings of the IEEE/CVF International Conference on Computer Vision 11975–11986.
[41] Ziyin, L., Lubana, E. S., Ueda, M. and Tanaka, H. (2023). What shapes the loss landscape of
self-supervised learning? In The Eleventh International Conference on Learning Representations.
Appendix A: Proofs of Optimization Results
Proof of Theorem 3.2. Let ν and µ be measures, both absolutely continuous with respect to a
common reference measure λ. The Kullback-Leibler divergence between µ and ν is
(cid:20) (cid:21)
dµ/dλ
D (µ ν) = E log .
kl µ
|| dν/dλ
Using this identity with reference measures µ and µ , respectively, we obtain
u v
(cid:20) dµ ( v)(cid:21)
E (cid:2) D (µ ( v) ν ( v;θ) (cid:3) = E E log u|v ·| E E [logρ(u v;θ)],
v∼µv kl u|v ·| || u|v ·| v∼µv µu|v(·|v) dµ − v∼µv u∼µu|v(·|v) |
u
(cid:20) dµ ( u)(cid:21)
E (cid:2) D (µ ( u) ν ( u;θ) (cid:3) = E E log v|u ·| E E [logρ(v u;θ)].
u∼µu kl v|u ·| || v|u ·| u∼µu µv|u(·|u) dµ − u∼µu v∼µv|u(·|u) |
v
Under the finite-entropy assumption, the objective J can be written as
cond
J (θ) = 1 E (cid:2) D (µ ( v) ν ( v;θ) (cid:3) + 1 E (cid:2) D (µ ( u) ν ( u;θ)) (cid:3)
cond 2 v∼µv kl u|v ·| || u|v ·| 2 u∼µu kl v|u ·| || v|u ·|
1 1
= C E E [logρ(u v;θ)] E E [logρ(v u;θ)]
− 2 v∼µv u∼µu|v(·|v) | − 2 u∼µu v∼µv|u(·|u) |
= C +L (θ),
cond
where C is a constant that only depends on µ, and hence not on the parameters θ, given by
1 (cid:20) dµ ( v)(cid:21) 1 (cid:20) dµ ( u)(cid:21)
C := E E log u|v ·| + E E log v|u ·|
2 v∼µv µu|v(·|v) dµ 2 u∼µu µv|u(·|u) dµ
u v
= 1 E (cid:2) D (µ ( v) µ ) (cid:3) + 1 E (cid:2) D (µ ( u) µ ) (cid:3) .
2 v∼µv kl u|v ·| || u 2 u∼µu kl v|u ·| || v
From the assumption, C < . Thus, a minimizer of J is also a minimizer of L .
cond cond
∞
Proof of Theorem 3.10. For the lower bound, we use that x log(x) is a convex function on
(cid:55)→ −
(0, ) so that by Jensen’s inequality we have
∞
E (cid:2) logE exp (cid:0) g¯ (u;θ ),g¯ (v;θ ) /τ (cid:1)(cid:3) logE exp (cid:0) g¯ (u;θ ),g¯ (v;θ ) /τ (cid:1)
u∼µu
−
v∼µv
⟨
u u v v
⟩ ≥ −
(u,v)∼µu⊗µv
⟨
u u v v
⟩
E (cid:2) logE exp (cid:0) g¯ (u;θ ),g¯ (v;θ ) /τ (cid:1)(cid:3) logE exp (cid:0) g¯ (u;θ ),g¯ (v;θ ) /τ (cid:1) .
v∼µv
−
u∼µu
⟨
u u v v
⟩ ≥ −
(u,v)∼µu⊗µv
⟨
u u v v
⟩

|              |     |       |             |     | Baptista, |          | Stuart, Tran/Contrastive |     |       | Learning |     |     |     | 36  |
| ------------ | --- | ----- | ----------- | --- | --------- | -------- | ------------------------ | --- | ----- | -------- | --- | --- | --- | --- |
| Substituting |     | these | expressions |     | in        | the loss | function                 | L   | gives | us       |     |     |     |     |
cond
|     |     |      | E         |      |           |      |        |     | logE        |     |      |      |                  |     |
| --- | --- | ---- | --------- | ---- | --------- | ---- | ------ | --- | ----------- | --- | ---- | ---- | ---------------- | --- |
|     | L   | (θ)  |           |      | [ g¯ (u;θ | ),g¯ | (v;θ ) | /τ] |             |     | [ g¯ | (u;θ | ),g¯ (v;θ ) /τ]. |     |
|     | −   | cond | ≥ (u,v)∼µ |      | ⟨ u       | u    | v v    | ⟩ − | (u,v)∼µu⊗µv |     | ⟨ u  | u    | v v ⟩            |     |
|     |     |      | = L       | (θ), |           |      |        |     |             |     |      |      |                  |     |
joint
−
for all θ Rp. Multiplying both signs by a negative number gives us the final result.
∈
Proof of Theorem 4.2. For an empirical distribution, we have that the maximum is achieved at one
| of  | the elements |     | in the | dataset. | Moreover, |     |     |     |     |     |     |     |     |     |
| --- | ------------ | --- | ------ | -------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
argmaxρN(u
|     |     |     |     |     |     |     |     | v;θ) = | argmaxω  | (v). |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | -------- | ---- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | |   |        |          | i    |     |     |     |     |
|     |     |     |     |     |     | u∈U |     |        | ui:i∈[N] |      |     |     |     |     |
Given that the un-normalized weights are monotonic functions of the cosine similarity and the
normalization constant does not affect the minimum, we have the result.
| Appendix |     | B:  | Use | of Maximum |     | Mean | Discrepancy |     |     |     |     |     |     |     |
| -------- | --- | --- | --- | ---------- | --- | ---- | ----------- | --- | --- | --- | --- | --- | --- | --- |
Example B.1. One metric that is readily computable between empirical measures is the maximum
mean discrepancy (MMD). Assume we are given a space U and kernel function k: U U R
+
|     |     |     |     |     |     |     |     |     |     |     |     |     | ×   | →   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
measuring the closeness of points in U; for example we might use the Gaussian kernel or polynomial
| kernel. | The | MMD | between |      | measures   | π       | and π′ | supported  | on  | U is     |              |     |          |      |
| ------- | --- | --- | ------- | ---- | ---------- | ------- | ------ | ---------- | --- | -------- | ------------ | --- | -------- | ---- |
|         |     | D   | (π,π′)  | := E |            | k(x,x′) |        | 2E         |     | k(x,y)+E |              |     | k(y,y′). | (60) |
|         |     | mmd |         |      | (x,x′)∼ν⊗ν |         |        | (x,y)∼ν⊗ν′ |     |          | (y,y′)∼ν′⊗ν′ |     |          |      |
−
|     |     |     |     | 1 (cid:80)N |     | π′  | 1 (cid:80)N |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ----------- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
For instance, if π = δ and = δ , then the metric has the form
|     |     |     | N   | i=1 | xi  |     | N   | i=1 yi |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
|     |     |     |     | 1   |     |     |     | 1      |     |     |     | 1   |     |     |
(π,π′) (cid:88) k(xi,xj) (cid:88) k(xi,yj)+ (cid:88) k(yi,yj).
|     | D   |     | :=  |     |      |     | 2   |     |      |     |     |     |      |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- | --- | ---- | --- |
|     | mmd |     | N(N |     | 1)   |     | −   | N(N | 1)   |     | N(N |     | 1)   |     |
|     |     |     |     | −   | i̸=j |     |     | −   | i̸=j |     |     | −   | i̸=j |     |
Now set U = and choose a kernel k ; and then set U = and choose a kernel k . Using the
|     |     |     |     |     |     |     | u   |     |     |     |     |     | v   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | U   |     |     |     |     |     |     | V   |     |     |     |     |
two resulting MMDs for the divergence D in Optimization Problem 3.4 we have the objective function
(noting that the MMD distance will typically be defined with different kernels on the two different
| spaces |     | , ) given | as: |     |     |     |     |     |     |     |     |     |     |     |
| ------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|        | U   | V         |     |     |     |     |     |     |     |     |     |     |     |     |
λ
|     |     |          |      |     |     | uE  | (cid:2) |         |         |       | (cid:3) |     |         |     |
| --- | --- | -------- | ---- | --- | --- | --- | ------- | ------- | ------- | ----- | ------- | --- | ------- | --- |
|     |     | J        | (θ;λ | ,λ  | ) = |     | D       | (µ      | ( v)    | ν (   | v;θ))   |     |         |     |
|     |     | cond,mmd |      | u   | v   | 2   | v∼µv    | mmd u|v |         | u|v   |         |     |         |     |
|     |     |          |      |     |     |     |         |         | ·|      | || ·| |         |     |         |     |
|     |     |          |      |     |     |     |         | λ       | (cid:2) |       |         |     | (cid:3) |     |
|     |     |          |      |     |     |     | +       | vE      | D       | (µ    | ( u)    | ν ( | u;θ)) . |     |
|     |     |          |      |     |     |     |         | u∼µu    |         | mmd   | v|u     | v|u |         |     |
|     |     |          |      |     |     |     |         | 2       |         |       | ·|      | ||  | ·|      |     |
Only a subset of terms in this objective depend on the parameters θ. Thus to minimize this objective
with respect to θ, we may identify a loss function comprising only those terms that depend on θ. The
objective is a sum of two terms for matching the u v and v u conditionals:
|     |     |     |          |     |         |     |            | |         |     | |          |          |     |     |     |
| --- | --- | --- | -------- | --- | ------- | --- | ---------- | --------- | --- | ---------- | -------- | --- | --- | --- |
|     |     |     | J        |     | (θ;1,1) | =   | λ J        | (θ;1,0)+λ |     | J          | (θ;0,1). |     |     |     |
|     |     |     | cond,mmd |     |         |     | u cond,mmd |           |     | v cond,mmd |          |     |     |     |
We first identify the θ-dependent terms in the contribution to this objective defined by the u v
|
conditional. The term defined by the v u conditional is handled similarly, by symmetry. From the
|

|     |     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     |     | Learning |     |     |     | 37  |
| --- | --- | --- | --- | --- | --------- | ------- | ---------------- | --- | --- | -------- | --- | --- | --- | --- |
definition in (60), the expected MMD between the true and model u v conditionals is given by
|
|     |          |     |         | 1   | (cid:104) |                            |                              |     |     |        |        |     |     |     |
| --- | -------- | --- | ------- | --- | --------- | -------------------------- | ---------------------------- | --- | --- | ------ | ------ | --- | --- | --- |
|     |          |     |         | E   |           | E                          |                              |     |     | (u,u′) |        |     |     |     |
|     | J        |     | (θ;1,0) | =   |           |                            |                              |     | k   |        |        |     |     |     |
|     | cond,mmd |     |         | 2   | v∼µv      | (u,u′)∼µu|v(·|v)⊗µu|v(·|v) |                              |     | u   |        |        |     |     |     |
|     |          |     |         |     |           | 2E                         |                              |     |     | k      | (u,u′) |     |     |     |
|     |          |     |         |     |           |                            | (u,u′)∼µu|v(·|v)⊗νu|v(·|v;θ) |     |     | u      |        |     |     |     |
−
(cid:105)
|     |     |     |     |     |           |              |              | +E          |                                |        |      | k (u,u′) |     |     |
| --- | --- | --- | --- | --- | --------- | ------------ | ------------ | ----------- | ------------------------------ | ------ | ---- | -------- | --- | --- |
|     |     |     |     |     |           |              |              |             | (u,u′)∼νu|v(·|v;θ)⊗νu|v(·|v;θ) |        |      | u        |     |     |
|     |     |     |     | 1   | (cid:104) |              |              |             |                                |        |      |          |     |     |
|     |     |     |     | = E |           | E            |              | k (u,u′)r(u |                                | v)r(u′ | v)   |          |     |     |
|     |     |     |     |     | v∼µv      | (u,u′)∼µu⊗µu |              | u           |                                |        |      |          |     |     |
|     |     |     |     | 2   |           |              |              |             |                                | | |    |      |          |     |     |
|     |     |     |     |     |           | 2E           |              |             | (u,u′)r(u                      | v)ρ(u′ |      |          |     |     |
|     |     |     |     |     |           |              | (u,u′)∼µu⊗µu |             | k                              |        | v;θ) |          |     |     |
|     |     |     |     |     |           | −            |              |             | u                              | |      | |    |          |     |     |
(cid:105)
|     |     |     |     |     |     |     |     | +E  |              |     | (u,u′)ρ(u | v;θ)ρ(u′ |      |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --------- | -------- | ---- | --- |
|     |     |     |     |     |     |     |     |     |              |     | k         |          | v;θ) |     |
|     |     |     |     |     |     |     |     |     | (u,u′)∼µu⊗µu |     | u         | |        | |    |     |
1
|       |     |         |         | E   |                   |     |     | (u,u′)r(u | v)r(u′ |      |          |          |     |     |
| ----- | --- | ------- | ------- | --- | ----------------- | --- | --- | --------- | ------ | ---- | -------- | -------- | --- | --- |
|       |     |         |         | =   |                   |     | k   |           |        | v)+L |          | (θ;1,0), |     |     |
|       |     |         |         | 2   | (v,u,u′)∼µv⊗µu⊗µu |     |     | u         |        |      | cond,mmd |          |     |     |
|       |     |         |         |     |                   |     |     |           | |      | |    |          |          |     |     |
| where | the | loss is | defined | as  |                   |     |     |           |        |      |          |          |     |     |
1
|     |     | L        | (θ;1,0) | =   | E   |                   |     | k   | (u,u′)ρ(u       | v;θ)ρ(u′ | v;θ) |            |       |     |
| --- | --- | -------- | ------- | --- | --- | ----------------- | --- | --- | --------------- | -------- | ---- | ---------- | ----- | --- |
|     |     | cond,mmd |         |     |     | (v,u,u′)∼µv⊗µu⊗µu |     | u   |                 |          |      |            |       |     |
|     |     |          |         |     | −2  |                   |     |     |                 | |        | |    |            |       |     |
|     |     |          |         |     |     |                   |     |     | +E              |          | k    | (u,u′)ρ(u′ | v;θ). |     |
|     |     |          |         |     |     |                   |     |     | ((v,u),u′)∼µ⊗µu |          |      | u          |       |     |
|
| We  | then minimize |     |            |     |         |       |            |           |     |              |          |     |     |     |
| --- | ------------- | --- | ---------- | --- | ------- | ----- | ---------- | --------- | --- | ------------ | -------- | --- | --- | --- |
|     |               |     | L cond,mmd |     | (θ;1,1) | = λ u | L cond,mmd | (θ;1,0)+λ |     | v L cond,mmd | (θ;0,1). |     |     |     |
µN,µN,µN
This may be evaluated using only the empirical measures in (3) by noting that
|          |          |     |           |     |                  |             |     |                  |     | u      | v        |             |        |     |
| -------- | -------- | --- | --------- | --- | ---------------- | ----------- | --- | ---------------- | --- | ------ | -------- | ----------- | ------ | --- |
|          |          |     |           | 1   | (cid:88)(cid:88) |             |     |                  |     |        | (cid:88) |             |        |     |
|          | LN       |     |           |     |                  | (uj,uk)ρ(uj |     | vi;θ)ρ(uk        |     | vi;θ)+ |          | (ui,uk)ρ(uk | vi;θ), |     |
|          |          |     | (θ;1,0)   | =   |                  | k u         |     |                  |     |        | k        | u           |        |     |
|          | cond,mmd |     |           | 2   |                  |             |     | |                |     | |      |          |             | |      |     |
|          |          |     |           |     | i j̸=k           |             |     |                  |     |        | i̸=k     |             |        |     |
| together | with     | the | analogous |     | expression       | for         | the | v u conditional. |     |        |          |             |        |     |
|
Appendix C: Retrieval and Classification at the Population Level
| C.1. | Retrieval: |     | Population |     | Level |     |     |     |     |     |     |     |     |     |
| ---- | ---------- | --- | ---------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
For simplicity of exposition we assume that the the marginal of the data distribution over u
is absolutely continuous with respect to the Lebesgue measure: there exists potential function
R
| ϕ u | :   | such | that |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
U →
|     |     |     |     |     |     | µ (du) | =   | exp(ϕ | (u))du. |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------ | --- | ----- | ------- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     | u      |     |       | u       |     |     |     |     |     |
Then, the conditional measure for u v in (15) also has Lebesgue density
|
|     |     |     |       |      |          |          | (cid:0)  |      |      |         | (cid:1)  |          |     |     |
| --- | --- | --- | ----- | ---- | -------- | -------- | -------- | ---- | ---- | ------- | -------- | -------- | --- | --- |
|     |     |     |       |      | (cid:16) | exp      | g¯ (u;θ  | ),g¯ | (v;θ | ) /τ +ϕ | (u)      | (cid:17) |     |     |
|     |     |     | ν (du | v;θ) | =        |          | ⟨ u      | u    | v    | v ⟩     | u        | du.      |     |     |
|     |     |     | u|v   |      |          | (cid:82) |          |      |      |         |          |          |     |     |
|     |     |     |       | |    |          | exp(     | g¯ (u′;θ | ),g¯ | (v;θ | ) /τ +ϕ | (u′))du′ |          |     |     |
|     |     |     |       |      |          | U        | u        | u    | v v  |         | u        |          |     |     |
|     |     |     |       |      |          |          | ⟨        |      |      | ⟩       |          |          |     |     |
We may view this formulation in the context of a Bayesian inverse problem for u given v. The
preceding identity defines the posterior ν from the prior µ and likelihood proportional to ρ( v)
|     |     |     |     |     |     |     | u|v |     |     | u   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
·|
defined in (9a). To identify the input u that is most related to the given point v , it is
|     |     |     |     |     |     |     | ∈ U |     |     |     |     |     | ∈ V |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
natural to seek the point in that maximizes the posterior probability. That is,
U
|     |     |     |     |     |     |     |     | (cid:16) |     |     |     |     | (cid:17) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | -------- | --- |
u∗(v) argmaxν (u v;θ) = argmax g¯ (u;θ ),g¯ (v;θ ) /τ +ϕ (u) . (61)
|     |     |     |     |     | u|v |     |     |     | u   | u v | v   | u   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | ∈   |     |     | |   |     |     | ⟨   |     | ⟩   |     |     |     |
|     |     |     |     | u∈U |     |     | u∈U |     |     |     |     |     |     |     |
u∗
In the context of Bayesian inverse problems, is commonly referred to as the mode or maximum
| a-posteriori |     | (MAP) | point. |     |     |     |     |     |     |     |     |     |     |     |
| ------------ | --- | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

|                      |     |     |            | Baptista, |       | Stuart, | Tran/Contrastive |     | Learning |     |     | 38  |
| -------------------- | --- | --- | ---------- | --------- | ----- | ------- | ---------------- | --- | -------- | --- | --- | --- |
| C.2. Classification: |     |     | Population |           | Level |         |                  |     |          |     |     |     |
The contrastive learning problem defines the learned joint distribution ν, given by (13) and (14), as
change of measure ρ from reference measure µ µ . In the Bayesian interpretation we may view
|     |     |     |     |     |     |     | u ⊗ | v   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the resulting conditional distribution ν from (16) as being defined by the likelihood proportional
v|u
to ρ(v u;θ) and the prior µ . In classification we use the same likelihood, possibly with a modified
v
|
parameter θ found using fine-tuning, but develop algorithms that generalize to different priors.
To this end, we introduce a measure µ on and modify (13), (14) to obtain ν( ;θ), a
|     |     |     |     |     |     | (cid:98) |     |     |     |     |     | (cid:98) |
| --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | -------- |
|     |     |     |     |     |     |          | U   | × V |     |     |     | ·        |
probability measure for the joint random variable (u,v) , defined by
∈ U ×V
|     |     | ν(du,dv;θ) |          | =   | ρ(u,v;θ)µ |           | (du)µ     | (dv),  |          |     |     | (62a) |
| --- | --- | ---------- | -------- | --- | --------- | --------- | --------- | ------ | -------- | --- | --- | ----- |
|     |     | (cid:98)   |          |     |           | (cid:98)u | (cid:98)v |        |          |     |     |       |
|     |     |            |          |     | 1         | (cid:16)  |           |        | (cid:17) |     |     |       |
|     |     |            | ρ(u,v;θ) | =   | exp       | g¯ u (u;θ | u ),g¯    | v (v;θ | v ) /τ , |     |     | (62b) |
|     |     |            |          |     | Z         | ⟨         |           |        | ⟩        |     |     |       |
(cid:90)
|       |           |           |     | Z =           |     | exp(     | g¯ (u;θ | ),g¯ | (v;θ ) /τ)µ | (du)µ               | (dv), | (62c) |
| ----- | --------- | --------- | --- | ------------- | --- | -------- | ------- | ---- | ----------- | ------------------- | ----- | ----- |
|       |           |           |     |               |     |          | u       | u v  | v           | (cid:98)u (cid:98)v |       |       |
|       |           |           |     |               | U×V | ⟨        |         |      | ⟩           |                     |       |       |
| where | µ (du),µ  | (dv)      | are | the marginals |     | of µ.    |         |      |             |                     |       |       |
|       | (cid:98)u | (cid:98)v |     |               |     | (cid:98) |         |      |             |                     |       |       |
R
For expository purposes we assume that, for ϕ(cid:98)v : and dv denoting Lebesgue measure,
V →
|     |     |     |     |     |     |      |       | (cid:0)        | (cid:1) |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | ----- | -------------- | ------- | --- | --- | --- |
|     |     |     |     |     | µ   | (dv) | = exp | ϕ(cid:98)v (v) | dv.     |     |     |     |
(cid:98)v
The conditional measure for v u then also has a Lebesgue density and is given by
|
|     |     |             |          |     | (cid:32) |         |      |            |      |                      | (cid:33) |      |
| --- | --- | ----------- | -------- | --- | -------- | ------- | ---- | ---------- | ---- | -------------------- | -------- | ---- |
|     |     |             |          |     |          | (cid:0) |      |            |      | (cid:1)              |          |      |
|     |     |             |          |     | exp      | g¯      | (u;θ | ),g¯ (v;θ  | ) /τ | +ϕ(cid:98)v (v)      |          |      |
|     |     | ν           | (dv u;θ) | =   |          | ⟨       | u    | u v        | v ⟩  |                      | dv.      | (63) |
|     |     | (cid:98)v|u |          |     | (cid:82) | (cid:0) |      |            |      | (cid:1)              |          |      |
|     |     |             | |        |     | exp      | g¯      | (u;θ | ),g¯ (v′;θ | ) /τ | +ϕ(cid:98)v (v′) dv′ |          |      |
|     |     |             |          |     | V        | u       | u    | v          | v    |                      |          |      |
|     |     |             |          |     |          | ⟨       |      |            | ⟩    |                      |          |      |
For now we fix the parameters (θ ,θ ) at the solution found during contrastive learning based on
u v
measure µ. To assign one label v to an input u , it is natural to identify the mode that
|           |     |             |     |         | ∈ V  |         |      | ∈        | U   |     |          |     |
| --------- | --- | ----------- | --- | ------- | ---- | ------- | ---- | -------- | --- | --- | -------- | --- |
| maximizes | the | conditional |     | measure | over | labels. | That | is,      |     |     |          |     |
|           |     |             |     |         |      |         |      | (cid:16) |     |     | (cid:17) |     |
v∗(u)
argmaxν (v u;θ) = argmax g¯ (u;θ ),g¯ (v;θ ) /τ +ϕ(cid:98)v (v) . (64)
|     |     |     | ∈   | (cid:98)v|u | |   |     |     | ⟨ u | u   | v v ⟩ |     |     |
| --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | ----- | --- | --- |
|     |     |     | v∈V |             |     |     | v∈V |     |     |       |     |     |
In the Bayesian interpretation ν differs from ν only in the change of prior; the likelihood has
|     |     |     |     |     | (cid:98)v|u |     |     | v|u |     |     |     |     |
| --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
v∗(u)
been fixed. The point is the MAP estimator for the posterior ν .
(cid:98)v|u
Example C.1. For a set , one choice for the marginal distribution µ (cid:98)v is
C ⊆ V
|     |     |     |     |     |     | µ (dv)    | µ   | (dv)1 | .   |     |     |     |
| --- | --- | --- | --- | --- | --- | --------- | --- | ----- | --- | --- | --- | --- |
|     |     |     |     |     |     | (cid:98)v | v   | {v∈C} |     |     |     |     |
∝
Assuming that µ is absolutely continuous with respect to the Lebesgue measure so that there exists a
v
potential ϕ such that µ (dv) = exp(ϕ (v))dv, the optimization problem in (64) to identify the mode
|             | v   |         | v     |             | v      |          |      |           |      |          |     |     |
| ----------- | --- | ------- | ----- | ----------- | ------ | -------- | ---- | --------- | ---- | -------- | --- | --- |
| corresponds | to  | solving | the   | constrained |        | problem  |      |           |      |          |     |     |
|             |     |         |       |             |        | (cid:16) |      |           |      | (cid:17) |     |     |
|             |     |         | v∗(u) |             | argmax | g¯       | (u;θ | ),g¯ (v;θ | ) /τ | +ϕ (v) . |     |     |
|             |     |         |       |             |        |          | u    | u v       | v    | v        |     |     |
|             |     |         |       | ∈           | v∈C    | ⟨        |      |           | ⟩    |          |     |     |
During pretraining, the parameters (θ ,θ ) defining the encoders for both modalities are learned
u v
using data µ. To improve the classification accuracy of the model it may be advantageous, during a
fine-tuning phase, to adjust the conditional distribution whose mode defines the classifier so that
it better matches the dataset µ. We will use fine-tuning to find an improved prior and to modify
(cid:98)

Baptista, Stuart, Tran/Contrastive Learning 39
the parameter θ which defines the likelihood. To this end we first introduce a family of marginal
v
measures µ ( ;θ ) depending on parameters θ that are defined via their density with respect to µ :
(cid:101)v ϕ ϕ (cid:98)v
·
(cid:0) (cid:1) (cid:0) (cid:1)
µ
(cid:101)v
(dv;θ
ϕ
) = exp ϕ(cid:101)v (v;θ
ϕ
) µ
(cid:98)v
(dv) = exp ϕ(cid:101)v (v;θ
ϕ
)+ϕ(cid:98)v (v) dv.
Now define ϑ = (θ ,θ ). Using the same form for the change of measure as in (63), but using µ as
v ϕ (cid:101)v
reference measure rather than µ , we obtain
(cid:98)v
(cid:32) (cid:0) (cid:1) (cid:33)
exp g¯
u
(u;θ
u
),g¯
v
(v;θ
v
) /τ +ϕ(cid:101)v (v;θ
ϕ
)+ϕ(cid:98)v (v)
ν (dv u;ϑ) = ⟨ ⟩ dv. (65)
(cid:101)v|u | E
v′∼µ(cid:98)v
exp (cid:0)
⟨
g¯
u
(u;θ
u
),g¯
v
(v′;θ
v
)
⟩
/τ +ϕ(cid:101)v (v′;θ
ϕ
)+ϕ(cid:98)v (v′) (cid:1)
We consider the learning of ϑ by minimizing the following objective function for the one-sided
conditional distribution:
J (ϑ) = E D (µ ( u) ν ). (66)
fine u∼µ(cid:98)u kl (cid:98)v|u
·| ||
(cid:101)v|u
In solving this optimization problem, and adopting the Bayesian perspective, we choose the prior
(via optimal choice of parameter θ ) and the likelihood (via optimal choice of parameter θ ) so that
ϕ v
the resulting conditional distribution ν ( u;ϑ) is well-suited to classification tasks based on data
(cid:101)v|u
·|
from, or closely related to, µ. By Theorem 3.2 and Remark 3.3, the relevant optimization problem
(cid:98)
for fine-tuning parameters ϑ is
Optimization Problem C.2.
−
L
fine
(ϑ) = E
(u,v)∼µ(cid:98)
[
⟨
g¯
u
(u;θ
u
),g¯
v
(v;θ
v
)
⟩
/τ +ϕ(cid:101)v (v;θ
ϕ
)]
−
E
u∼µ(cid:98)u
logE
v′∼µ(cid:98)v
exp[
⟨
g¯
u
(u;θ
u
),g¯
v
(v′;θ
v
)
⟩
/τ +ϕ(cid:101)v (v′;θ
ϕ
)].
ϑ argmin L (ϑ).
fine fine
∈ ϑ∈Rp′
The resulting model with the optimal parameters ϑ is used to classify typical inputs u from
fine
the marginal distribution µ by finding the mode, analogously to (64):
(cid:98)u
v∗(u) argmaxν (v u;ϑ ). (67)
(cid:101)v|u fine
∈ |
v∈V
Appendix D: Gaussian Proofs and Remarks
Proof of Theorem 5.1. The loss function in (42) can be written as
1 1
L (A) = E u,Av + E logE exp u,Av + E logE exp u,Av
cond − (u,v)∼µ ⟨ ⟩ 2 u∼µu v∼µv ⟨ ⟩ 2 v∼µv u∼µu ⟨ ⟩
Using Lemma E.2 with B = 0 and c = Av or c = A⊤u we have that
(cid:18) (cid:19)
1
E exp u,Av = exp v⊤A⊤ Av , or
u∼µu
⟨ ⟩ 2 C
uu
(cid:18) (cid:19)
1
E exp u,Av = exp u⊤A A⊤u ,
v∼µv
⟨ ⟩ 2 C
vv
respectively. Substituting these forms in L results in the objective
cond
(cid:18) (cid:19) (cid:18) (cid:19)
1 1 1 1
L (A) = E Tr(Avu⊤)+ E u⊤A A⊤u + E v⊤A⊤ Av
cond − (u,v)∼µ 2 u∼µu 2 C vv 2 v∼µv 2 C uu
1 1
= E Tr(Avu⊤)+ E Tr(A⊤ Auu⊤)+ E Tr(A⊤ Avv⊤)
− (u,v)∼µ 2 u∼µu C vv 2 v∼µv C uu
= Tr(A )+Tr(A⊤ A ).
vu uu vv
− C C C

Baptista, Stuart, Tran/Contrastive Learning 40
By adding a constant that is not dependent on A to the loss function we have
L (A)+Tr( −1 −1 ) = Tr((A⊤ −1 )(A −1 ))
cond CuuC uv Cvv C vu C uu −Cvv C vu C vv −CuuC uv
= Tr( (A −1 −1)⊤ (A −1 −1))
C vv −CuuC uv Cvv C uu −CuuC uv Cvv
= 1/2(A A∗) 1/2 2, (68)
∥Cvv − Cuu ∥F
where we define A∗ := −1 −1. Given that minimizing L corresponds to minimizing (68), which
Cvv C vu Cuu
is bounded from below, we have that the minimum over all matrices A is given by A∗.
With the rank constraint, from [12, Theorem 2.1] we have that
A∗ = argmin 1/2(A A∗) 1/2 2,
r ∥Cvv − Cuu ∥F
A∈Ar
has a unique solution given by computing the SVD of 1/2A∗ 1/2 = −1/2 −1/2 and then
vv uu vv vu uu
C C C C C
pre-multiplying by the square roots of the marginal covariances to get A∗.
r
Proof of Corollary 5.2. If the parameter in the learnable model ν (39) is chosen to be A = A∗(r)
for any 0 < r min(n ,n ), the form of the approximate conditional distributions is given by:
u v
≤
ν (u v;A∗(r)) = ( 1/2( −1/2 −1/2) −1/2v, ) (69a)
u|v | N Cuu Cuu C uv Cvv r Cvv C uu
ν (v u;A∗(r)) = ( 1/2( −1/2 −1/2)⊤ −1/2u, ). (69b)
v|u | N Cvv Cuu C uv Cvv r Cuu C vv
Without the rank constraint, i.e., r = min(n ,n ), we have A∗(r) = A∗ and hence the conditional
u v
measures ν and ν are given by (45b) and (45a), respectively.
u|v v|u
Proof of Theorem 5.3. Using Lemma E.1 for the KL Divergence between the multivariate Gaussians
µ and ν in expectation over µ , the objective for A,B is given by
u|v u|v v
J (A,B;2,0) = E (cid:2) D (µ ( v) ν ( v)) (cid:3)
cond v∼µv kl u|v
·| ||
u|v
·|
= 1 (cid:2) Tr((B+ −1) ) d+log (B+ −1) (cid:3) +∆(A,B), (70)
2 Cuu C u|v − | Cuu C u|v |
where the last term is expressed as
∆(A,B) := E (B+ −1)−1Av −1v 2
v∼µv∥ Cuu −C uv Cvv ∥(B+Cu − u 1)−1
= Tr(((B+ −1)−1A −1)⊤(B+ −1)((B+ −1)−1A −1) )
Cuu −C uv Cvv Cuu Cuu −C uv Cvv C vv
(cid:13)(cid:16) (cid:17) (cid:13)2
= (cid:13) (B+ −1)−1/2A (B+ −1)1/2 −1 1/2(cid:13) .
(cid:13) Cuu − Cuu C uv Cvv Cvv (cid:13) F
Only the last term in (70) depends on A. Minimizing the last term over unconstrained matrices A
for each B results in
A∗ = (B+ −1) −1,
Cuu C uv Cvv
where the minimum satisfies ∆(A∗,B) = 0. Then, the minimizer of L (A∗,B;2,0) over uncon-
cond
strained matrices B is given by
B∗ = −1 −1 = −1 −1 −1.
Cu|v −Cuu CuuC uv Cv|uC vu Cuu
Minimizing the last term in (70) over the constrained set for each B gives us
r
A
A∗(r) = (B+ −1)1/2((B+ −1)1/2 −1/2) −1/2,
Cuu Cuu C uv Cvv r Cvv

Baptista, Stuart, Tran/Contrastive Learning 41
where the minimum of the third term is then given by
(cid:13)(cid:16) (cid:17) (cid:16) (cid:17)(cid:13)2
∆(A∗(r),B) = (cid:13) (B+ −1)1/2 −1/2 (B+ −1)1/2 −1/2 (cid:13) .
(cid:13) Cuu C uv Cvv r − Cuu C uv Cvv (cid:13) F
Substituting the minimum value of ∆(A∗(r),B) into the loss yields the objective in (50b) for finding
the optimal B∗(r).
Proof of Corollary 5.4. Iftheparametersinthelearnablemodelν in(46)arechosentobeA = A∗(r)
and B = B∗(r) from Theorem 5.3, the approximate conditional distribution ν in (48) is given by
u|v
(cid:16) (cid:17)
ν (u v;A∗(r),B∗(r)) = (B∗(r)+ )−1/2(cid:0) (B∗(r)+ )1/2 −1/2(cid:1) −1/2v,(B∗(r)+ −1)−1 ,
u|v | N C uu C uu C uv Cvv rCvv Cuu
where B∗(r) solves the optimization problem in (50b).
Without a rank constraint, i.e., r = n , we have B∗(r) = B∗ in (49b). Then, using the Sherman-
u
Woodbury matrix identity, the conditional covariance of u v for the learnable model is
|
(B∗+ −1)−1 = ( −1 −1 −1+ −1)−1 = −1 .
Cuu CuuC uv Cv|uC vu Cuu Cuu C uu −C uv Cvv C vu
Thus, the conditional distribution for u v is given by (45a), whose conditional expectations match
|
the conditional expectations for u v in (37a).
|
The one-sided objective L (A,B;2,0) does not depend on the matrix C = H⊤H appearing
cond
in (48), which defines the approximate conditional distribution for v u. For this reason, minimizing
|
the one-sided objective only matches the conditional distribution for u v.
|
Proof of Theorem 5.6. Using Lemma E.2 with z = (u,v) µ µ we have
u v
∼ ⊗
(cid:20) (cid:18) (cid:19)(cid:21)
1 1
E [exp(u⊤Av)] = E exp (u,v)⊤B(u,v) = ,
(u,v)∼µu⊗µv z∼N(0,Λ) 2 (cid:112) I ΛB
|
nu+nv
− |
where
(cid:20) (cid:21) (cid:20) (cid:21)
0 0 A
Λ = C uu , B = .
0 A⊤ 0
vv
C
Thus, the objective function in (52) is given by
L (A) = E [Tr(Avu⊤)]+logE [exp(u⊤Av)]
joint
−
(u,v)∼µ (u,v)∼µu⊗µv
1
= Tr(A ) log I A⊤ A
− C
vu
− 2 |
nv
−C
vv
C
uu
|
1
= Tr(( 1/2A 1/2)( −1/2 1/2)) log I 1/2( 1/2A⊤ 1/2)( 1/2A 1/2) −1/2 .
− Cuu Cvv Cvv C vu Cuu − 2 | nv −Cvv Cvv Cuu Cuu Cvv Cvv |
Let A(cid:98) :=
u
1/
u
2A
v
1
v
/2 Rnu×nv and Ω =
v
−
v
1/2
vu u
−
u
1/2 Rnv×nu. Then, an equivalent objective
C C ∈ C C C ∈
function that is maximized to identify A(cid:98) is given by
1
(cid:98)L
joint
(A(cid:98)) := Tr(A(cid:98)Ω)+
2
log
|
I
nv
−
A(cid:98) ⊤A(cid:98)
|
.
Let the matrix Ω have a singular value decomposition UΣV⊤. Each matrix A(cid:98) has a singular value
decomposition V(cid:98)DU(cid:98) ⊤. By the von Neumann Trace inequality (see [3]), we have
min(nu,nv)
(cid:88)
max Tr(V(cid:98)DU(cid:98) ⊤UΣVT) D
ii
Σ
ii
≤
U(cid:98),V(cid:98)
i=1

Baptista, Stuart, Tran/Contrastive Learning 42
where equality is attained at U(cid:98) = U and V(cid:98) = V. By the invariance of the log-determinant to the
singular vectors of A(cid:98) ⊤A(cid:98), the objective satisfies
min
(cid:88)
(nu,nv)
1
(cid:98)L
joint
(A(cid:98))
≤
D
ii
Σ
ii
+
2
log(1
−
D
i
2
i
),
i=1
which is a sum of concave functions for each singular value D . Thus, the objective is maximized at
ii
the solutions of the equation Σ D /(1 D2) = 0 for i = 1,...,min(n ,n ). Rearranging into a
ii − ii − ii u v
quadratic equation, the objective is maximized at the positive root of the equation
D2Σ +D Σ = 0,
ii ii ii − ii
whose solution is given by D = h(Σ ) for the function h in Definition 5.5.
ii ii
In the rank-constrained setting, the same result follows with D = 0 for i > r.
ii
Proof of Corollary 5.7. For any parameter A, the marginal distribution of the parameterized model
ν in (39) is given by
ν (du;A) = (0, + A( −1 A⊤ A)−1A⊤ ),
u N C uu C uu Cvv − C uv C uu
where the covariance follows from the Schur complement of the inverse covariance of ν for choices of
A so that −1 A⊤ A is invertible. In particular, for any parameter of the form
Cvv − C uv
A = −1/2UDV⊤ −1/2 (71)
Cuu Cvv
where U and V are unitary matrices and D is a diagonal matrix, the marginal distribution of ν is
u
ν (du;A) = (0, 1/2U(I D2)−1U⊤ 1/2), (72)
u N Cuu nu − Cuu
where the entries of D must satisfy D < 1 for the covariance to be well defined.
ii
| |
The optimal parameters A∗ in (43) and A∗ in (53) that minimize the conditional and joint
cond joint
loss functions, respectively, both have the form in (71). In particular, letting UΣV denote the SVD
of
−1/2 −1/2,
the optimal parameters are given by
uu uv vv
C C C
A∗ = −1/2UΣV⊤ −1/2
cond Cuu Cvv
A∗ = −1/2Uh(Σ)V⊤ −1/2.
joint Cuu Cvv
SubstitutingD = ΣandD = h(Σ)in(72)yieldsthemarginaldistributionsin(55a)and(55b)forthe
parameters minimizing the conditional and joint losses, respectively. Moreover, we note that D = 0
or equivalently A = 0 yields the marginal distribution of the data distribution µ (du) = (0, ).
u uu
N C
Lastly, using the property 0 h(σ) < σ for the function h in Definition 5.5, we have that
≤
I (I h(Σ)2)−1 (I Σ2)−1. Thus, it follows that
≺ − ≺ −
= 1/2UU⊤ 1/2 1/2U(I h(Σ)2)−1U⊤ 1/2 1/2U(I Σ2)−1U⊤ 1/2.
C uu Cuu Cuu ≺ Cuu nu − Cuu ≺ Cuu nu − Cuu
That is, the marginal covariance corresponding to A∗ is strictly closer to the true marginal
joint
covariance of µ than the marginal covariance corresponding to A∗ in the cone of positive definite
u cond
matrices.

|     |     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     |     | Learning |     |     |     | 43  |
| --- | --- | --- | --- | --- | --------- | ------- | ---------------- | --- | --- | -------- | --- | --- | --- | --- |
Remark D.1. The loss function (42) can also be derived by noting that KL divergence between
two multivariate Gaussians with the same covariance, which we denote by J cond (A), is given by
squared and weighted L2 norms of the errors in the mean for both the u v and v u conditionals; see
|       |      |       |         |     |     |     |     |     |     |          | |   |     | |          |     |
| ----- | ---- | ----- | ------- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | ---------- | --- |
| Lemma | E.1. | Then, | we have |     |     |     |     |     |     |          |     |     |            |     |
|       |      |       | 1       |     |     |     |     |     | 1   | (cid:12) |     |     | (cid:12) 2 |     |
J (A) = E (cid:12) Av − 1v (cid:12) 2 + E (cid:12) A⊤u − 1u (cid:12)
cond v∼µv (cid:12) uu uv (cid:12) u∼µu (cid:12) vv vu (cid:12)
|     |     |     | 4   |     | C   | −C  | Cv v | Cuu | 4   | C   | −C  | Cu  | u Cvv |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | ----- | --- |
1
|     |     |     |     | (cid:0)   |         | −1)⊤   | −1(  |      |     | −1)    | (cid:1) |         |     |     |
| --- | --- | --- | --- | --------- | ------- | ------ | ---- | ---- | --- | ------ | ------- | ------- | --- | --- |
|     |     |     | =   | Tr (      | A       |        |      | A    |     |        |         |         |     |     |
|     |     |     | 4   |           | uu      | uv Cvv | Cuu  | uu   | uv  | Cvv    | vv      |         |     |     |
|     |     |     |     | C         | −C      |        |      | C    | −C  | C      |         |         |     |     |
|     |     |     |     | 1 (cid:0) |         |        |      |      |     |        |         | (cid:1) |     |     |
|     |     |     | +   | Tr        | ( A⊤    |        | −1)⊤ | −1(  | A⊤  |        | −1)     |         |     |     |
|     |     |     |     |           | vv      | vu     | Cuu  | Cvv  | vv  | vu Cuu | uu      |         |     |     |
|     |     |     |     | 4         | C       | −C     |      | C    |     | −C     | C       |         |     |     |
|     |     |     | 1   |           |         |        | 1    |      |     |        |         |         |     |     |
|     |     |     | =   | Tr( −1    | −1      | )+     | L    | (A), |     |        |         |         |     |     |
|     |     |     |     | Cvv       | vu CuuC | uv     | cond |      |     |        |         |         |     |     |
|     |     |     | 2   |           | C       |        | 2    |      |     |        |         |         |     |     |
where L cond (A) corresponds to the loss function in (42), which has the equivalent form
|     |     |     | L    | (A) | := Tr(A⊤ |     | A ) | Tr(A |     | ) Tr(A⊤ |     | ).  |     |     |
| --- | --- | --- | ---- | --- | -------- | --- | --- | ---- | --- | ------- | --- | --- | --- | --- |
|     |     |     | cond |     |          | uu  | vv  |      | uv  |         | vu  |     |     |     |
|     |     |     |      |     |          | C   | C   | −    | C   | −       | C   |     |     |     |
Similarly, the generalized loss function in Optimization Problem 3.4 for the KL divergence between
R
the u v and v u conditionals given weighting parameters λ ,λ has the form
| |   |     | |    |      |        |      |           |     |     | u      | v ∈ + |         |     |           |     |
| --- | --- | ---- | ---- | ------ | ---- | --------- | --- | --- | ------ | ----- | ------- | --- | --------- | --- |
|     |     |      |      |        | λ +λ | (cid:104) |     |     |        |       |         |     | (cid:105) |     |
|     |     | L    | (A;λ | ,λ ) = | u    | v Tr(A⊤   |     | A   | ) Tr(A |       | ) Tr(A⊤ |     | ) .       |     |
|     |     | cond | u    | v      |      |           | uu  | vv  |        | uv    |         |     | vu        |     |
|     |     |      |      |        | 2    |           | C   | C   | −      | C     | −       | C   |           |     |
Using general weighting λ u and λ v only results in a scaling of the loss arising in Theorem 5.1 when
λ = λ = 1. Thus, for any values λ ,λ , L (A;λ ,λ ) has the minimizer A∗ in (43) over all
| u   | v   |     |     |     |     | u v | cond | u   | v   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
matrices of size n n and the minimizer A∗ in (44) over rank-r matrices.
|     |     | u   | v   |     |     |     | r   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | ×   |     |     |     |     |     |     |     |     |     |     | ⋄   |
Remark D.2. In the empirical setting defined by (3), the loss for matrix A is given by
|     |     |     | LN   | (A) = | Tr(A⊤ | A(cid:98)vv | ) Tr(A(cid:98)uv |     | )   | Tr(A⊤ | ),         | where |     |     |
| --- | --- | --- | ---- | ----- | ----- | ----------- | ---------------- | --- | --- | ----- | ---------- | ----- | --- | --- |
|     |     |     |      |       |       | (cid:98)uu  |                  |     |     |       | (cid:98)vu |       |     |     |
|     |     |     | cond |       |       | C C         | −                | C   | −   |       | C          |       |     |     |
N
1 (cid:88)
|     |     |     |     | (cid:98):= |     | (ui,vi) | (ui,vi). |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---------- | --- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | C          | N   |         | ⊗        |     |     |     |     |     |     |     |
i=1
This has the same form as the population-level loss with the covariances replaced by their empirical
counterparts. Moreover, assuming that the product of empirical covariances − 1 − 1 is invertible,
|     |     |     |     |     |     |     |     |     |     |     |     | (cid:98) u u | (cid:98)uv (cid:98) v v |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ----------------------- | --- |
|     |     |     |     |     |     |     |     |     |     |     |     | C C          | C                       |     |
then the solution that minimizes LN is also given by Theorem 5.1 with the covariances replaced by
cond
| their empirical |     | counterparts. |     |     |     |     |     |     |     |     |     |     |     |     |
| --------------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
⋄
Remark D.3. After identifying A,B, the original encoders G,H are identifiable without the rank
constraint assuming that n n . That is, G Rnu×nu can be computed from a square root of
|     |     |     |     | u   | v   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | ≤   |     |     | ∈   |     |     |     |     |     |     |     |
G⊤G = B∗ given that B∗ is strictly positive definite. Moreover, given that G is invertible, we can
|     |     |     | G⊤H | A∗  |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
solve the equation = for H. We note that this procedure does not define a unique solution
| for G,H | unless | we  | seek | the positive-definite |     |     | square | root | of B∗. |     |     |     |     |     |
| ------- | ------ | --- | ---- | --------------------- | --- | --- | ------ | ---- | ------ | --- | --- | --- | --- | --- |
⋄
| Appendix |     | E: Useful |     | Identities |      |     |     |     |     |     |     |     |     |     |
| -------- | --- | --------- | --- | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|          |     |           |     |            | Rd×d |     |     |     |     |     |     |     |     | Rd  |
Lemma E.1 ([27]). Let C ,C be symmetric positive definite matrices and m ,m be
|     |     |     |     | 1   | 2 ∈ |     |     |     |     |     |     |     | 1 2 ∈ |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
mean vectors. The Kullback-Leibler divergence from (m ,C ) to (m ,C ) is
|     |     |         |     |      |      |           |          |     | 2     | 2   | 1     | 1   |          |     |
| --- | --- | ------- | --- | ---- | ---- | --------- | -------- | --- | ----- | --- | ----- | --- | -------- | --- |
|     |     |         |     |      |      |           |          | N   |       | N   |       |     |          |     |
|     |     |         |     |      |      |           | (cid:20) |     |       |     |       |     | (cid:21) |     |
|     |     | (cid:0) |     |      |      | (cid:1) 1 |          |     |       | C   |       |     |          |     |
|     | D   | (m      | ,C  | ) (m | ,C ) | =         | Tr(C−1C  | )   | d+log | |   | 2 | + | m   | m 2 .    |     |
|     | kl  |         | 1 1 |      | 2 2  |           | 2        | 1   |       |     |       | 2   | 1 ∥C2    |     |
|     |     | N       |     | ||N  |      | 2         |          |     | −     | C   | ∥     | −   |          |     |
|     |     |         |     |      |      |           |          |     |       | |   | 1 |   |     |          |     |

|     |     |     |     |     | Baptista, | Stuart, | Tran/Contrastive |     |     | Learning |     |     |     | 44  |     |
| --- | --- | --- | --- | --- | --------- | ------- | ---------------- | --- | --- | -------- | --- | --- | --- | --- | --- |
Rd
Lemma E.2. Let z follow a multivariate Gaussian distribution z ν = (m,Λ). Then, for
|        |          |            | Rd×d ∈ |                  |             | Rd,  |         |               |     |     | ∼ N          |     |        |     |          |
| ------ | -------- | ---------- | ------ | ---------------- | ----------- | ---- | ------- | ------------- | --- | --- | ------------ | --- | ------ | --- | -------- |
| any    | matrix   | B          |        | and vector       |             | c    | we have |               |     |     |              |     |        |     |          |
|        |          |            | ∈      |                  |             | ∈    |         |               |     |     |              |     |        |     |          |
|        | (cid:20) | (cid:18)   |        | (cid:19)(cid:21) |             |      |         | (cid:18)      |     |     |              |     |        |     | (cid:19) |
|        |          | 1          |        |                  |             | 1    |         | 1             |     |     |              |     |        |     |          |
| E      |          | z⊤Bz+c⊤z   |        |                  |             |      |         | (c+Λ−1m)⊤(Λ−1 |     |     | B)−1(c+Λ−1m) |     | m⊤Λ−1m |     |          |
|        | z∼ν exp  |            |        |                  | = (cid:112) |      | exp     |               |     |     |              |     |        |     | .        |
|        |          | 2          |        |                  |             | I    | ΛB      | 2             |     |     | −            |     | −      |     |          |
|        |          |            |        |                  |             | | −  | |       |               |     |     |              |     |        |     |          |
| Proof. |          | Completing | the    | square           | we          | have |         |               |     |     |              |     |        |     |          |
(cid:20) (cid:18) 1 (cid:19)(cid:21) (cid:90) 1 (cid:18) 1 (cid:19) (cid:18) 1 (cid:19)
| E   |     | z⊤Bz+c⊤z |     |     |     |           |     |     |     | m)⊤Λ−1(z |        | z⊤Bz+c⊤z |     |     |     |
| --- | --- | -------- | --- | --- | --- | --------- | --- | --- | --- | -------- | ------ | -------- | --- | --- | --- |
|     | exp |          |     |     | =   |           |     | exp | (z  |          | m) exp |          |     |     | dz  |
|     | z∼ν | 2        |     |     |     | (cid:112) |     |     | −2  |          |        | 2        |     |     |     |
|     |     |          |     |     |     | (2π)d     | Λ   |     |     | −        | −      |          |     |     |     |
| |
|     |     |     |     |     | (cid:90) |     |     | (cid:18) |     |          |      | (cid:19) |     |     |     |
| --- | --- | --- | --- | --- | -------- | --- | --- | -------- | --- | -------- | ---- | -------- | --- | --- | --- |
|     |     |     |     |     |          |     | 1   |          | 1   |          |      |          |     |     |     |
|     |     |     |     |     | =        |     |     | exp      | (z  | m∗)⊤(Λ−1 | B)(z | m∗)      |     |     |     |
(cid:112)
|     |     |     |     |     |     | (2π)d | Λ   |     | −2  | −   | − − |     | ×   |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| |
|     |     |     |     |     |     | (cid:18) | 1         |     |      |        | (cid:19) |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | --------- | --- | ---- | ------ | -------- | --- | --- | --- | --- |
|     |     |     |     |     |     |          | (m∗)⊤(Λ−1 |     | B)m∗ | m⊤Λ−1m |          |     |     |     |     |
|     |     |     |     |     |     | exp      |           |     |      |        | dz,      |     |     |     |     |
|     |     |     |     |     |     |          | 2         |     | −    | −      |          |     |     |     |     |
where m∗ = (Λ−1 B)−1(c+Λ−1m). Computing the normalizing constant in closed form and using
|     |          |     | −                      |     |     |     |     | Λ−1 |     |      |     |     |     |     |     |
| --- | -------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
| the | property | for | the matrix-determinant |     |     |     | Λ = | 1/  | we  | have |     |     |     |     |     |
|     |          |     |                        |     |     |     | | | | |   | |   |      |     |     |     |     |     |
(cid:115)
|              |     | (cid:20) | (cid:18) 1 |        | (cid:19)(cid:21) |        | Λ−1   |         | (cid:18) | 1         |      |        | (cid:19) |     |     |
| ------------ | --- | -------- | ---------- | ------ | ---------------- | ------ | ----- | ------- | -------- | --------- | ---- | ------ | -------- | --- | --- |
|              | E   |          | z⊤Bz+c⊤z   |        |                  |        |       |         |          | (m∗)⊤(Λ−1 | B)m∗ | m⊤Λ−1m |          |     |     |
|              |     | exp      |            |        |                  | =      | |     | | exp   |          |           |      |        | .        |     |     |
|              |     | z∼ν      | 2          |        |                  |        | Λ−1   | B       |          | 2         | − −  |        |          |     |     |
|              |     |          |            |        |                  |        | | −   | |       |          |           |      |        |          |     |     |
| Substituting |     | the      | form       | for m∗ | gives            | us the | final | result. |          |           |      |        |          |     |     |