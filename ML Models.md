The first categorisation of ML models are based on how it learns. 
==Model Taxonomy for ML== Every model sits inside one of 4 Learning Paradigms. The Paradigm decides what shape of Data the model is even allowed to touch. Within a Paradigm, models are grouped by Architecture/Technique family, and every model gets a metaphor, its Math-DNA 
([[Math]]), and its Data-fit ([[Data]]). 

***Hyperparameters are omitted for propriety causes.*** 

==**Hence every model has: 1 Paradigm, 1 metaphor, N math codes, and a Data-code signature.**==

## Learning Paradigms

|Code|Paradigm|Definition|Data Signature Used|
|---|---|---|---|
|SL|Supervised Learning|Learns a mapping from input → known output using fully labeled examples|A11 (Labeled) required; any A2x relational structure; any A3x modality|
|USL|Unsupervised Learning|Finds structure, groupings, or compressed representations with no target labels at all|A12 (Unlabeled) required; any A2x; any A3x|
|SSL|Semi/Self-Supervised Learning|Uses a small labeled slice plus a large unlabeled pool, or manufactures its own labels from the data (pretext tasks)|A14 (Semi-labeled) or A15 (Self-labeled); any A2x; any A3x|
|RL|Reinforcement Learning|Learns by acting in an environment and adjusting behavior based on reward/penalty feedback rather than fixed labels|A15 (Self-labeled, reward-as-signal); A22 (Sequential/temporal) almost always; A31/A32 typical state-action encodings|

---

## Domain: Linear Models [LM]

| Code | Model                      | Metaphor                                                                            | Math Codes        | Data Codes  | Paradigm |
| ---- | -------------------------- | ----------------------------------------------------------------------------------- | ----------------- | ----------- | -------- |
| LM1  | Linear Regression          | A ruler drawn through scattered dots, chasing the average trend                     | LA1+LA12+CA8+PS4  | A11-A21-A31 | SL       |
| LM2  | Logistic Regression        | A bouncer at the door, weighing a checklist to decide yes/no                        | LA1+CA10+CA22+PS3 | A11-A21-A32 | SL       |
| LM3  | Ridge Regression           | Linear Regression wearing a seatbelt : trades a bit of bias for a lot less variance | LA1+LA4+CA8       | A11-A21-A31 | SL       |
| LM4  | Lasso Regression           | A strict editor that deletes whole variables it decides are dead weight             | LA1+LA3+CA8       | A11-A21-A42 | SL       |
| LM5  | Elastic Net                | The diplomat between Ridge and Lasso : shrinks some, deletes some                   | CA23+LA3+LA4      | A11-A21-A42 | SL       |
| LM6  | Polynomial Regression      | Linear Regression that finally learned to draw curves instead of straight lines     | LA1+CA8+CA27      | A11-A21-A31 | SL       |
| LM7  | Bayesian Linear Regression | Linear Regression that refuses to give one answer, only a range of plausible ones   | PS1+PS3+PS2       | A11-A21-A31 | SL       |

## Domain: Tree-Based Models [TR]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|TR1|Decision Tree|A flowchart of yes/no questions : 20 Questions with data|IT1+IT2|A11-A21-A32|SL|
|TR2|Random Forest|Democracy : hundreds of independent trees vote, majority rules|IT1+IT2+PS20|A11-A21-A38|SL|
|TR3|Extra Trees|Democracy with even more randomness in who gets nominated as a candidate split|IT1+IT2|A11-A21-A38|SL|
|TR4|Isolation Forest|The odd one out in a police lineup : outliers get isolated in the fewest questions|IT1+PS4|A12-A21-A31|USL|

## Domain: Gradient Boosting / Ensemble Models [EN]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|EN1|AdaBoost|A classroom where the teacher keeps re-testing whoever failed last time, harder|CA8+PS3|A11-A21-A32|SL|
|EN2|Gradient Boosting (GBM)|A relay team : each runner exists only to correct the last runner's error|CA2+CA3+CA8|A11-A21-A31|SL|
|EN3|XGBoost|The CCP-govt model : centralized, top-down correction with heavy penalties (regularization) against excess|CA3+CA23+CA6+LA4|A11-A21-A42|SL|
|EN4|LightGBM|XGBoost's leaner, faster cousin : grows leaf-wise like a startup, not level-by-level like a bureaucracy|CA3+IT2|A11-A21-A42|SL|
|EN5|CatBoost|The specialist bureaucrat, purpose-built to process categorical paperwork without translating it first|CA3+PS1|A11-A21-A32|SL|
|EN6|Stacking|A panel of expert models whose separate opinions get combined by a judge model trained to weigh them|CA8+LA1|A11-A21-A38|SL|
|EN7|Voting Ensemble|A jury that just averages raw votes : no judge, no training, majority or mean wins|PS4|A11-A21-A38|SL|

## Domain: Support Vector Models [SV]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|SV1|SVM (linear)|Drawing the widest possible highway between two neighborhoods|LA1+CA14+CA20|A11-A21-A55|SL|
|SV2|Kernel SVM (RBF/Poly)|Warping the map into higher dimensions until a straight highway finally becomes possible|LA22+LA23+CA14+CA20|A11-A21-A56|SL|
|SV3|One-Class SVM|A bouncer who's only ever seen regulars, and flags anyone who doesn't fit the usual mold|LA22+CA14|A12-A21-A31|USL|

## Domain: Probabilistic & Graphical Models [PR]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|PR1|Naive Bayes|A stereotyping detective : treats every clue as independent evidence|PS1+PS9+PS2|A11-A21-A32|SL|
|PR2|Gaussian Mixture Model|A committee of bell curves, splitting a crowd into overlapping cliques|PS2+LA10+LA21|A12-A21-A31|USL|
|PR3|Hidden Markov Model|A weather guesser who only sees umbrellas, never the actual sky|RL5+PS9+PS1|A11-A22-A34|SL|
|PR4|Bayesian Network|A family tree of cause and effect, where each node only listens to its direct parents|PS1+PS9+PS10|A11-A24-A32|SL|
|PR5|Conditional Random Field|HMM's smarter cousin : labels the whole sequence at once instead of guessing one step at a time|PS9+CA9+IT3|A11-A22-A34|SL|

## Domain: Instance-Based Models [IB]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|IB1|k-Nearest Neighbors|You are the average of the 5 friends standing closest to you|LA9+LA19+LA20|A11-A21-A31|SL|
|IB2|K-Medoids|K-Means's more grounded cousin : the center must be a real member, not an average ghost point|LA9+LA19|A12-A21-A31|USL|

## Domain: Clustering Models [CL]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|CL1|K-Means|Towns forming naturally around whichever well happens to be closest|LA9+PS4|A12-A21-A31|USL|
|CL2|DBSCAN|A party where you only count as "in the group" if enough people are standing close to you|LA9|A12-A21-A31|USL|
|CL3|Hierarchical (Agglomerative) Clustering|A family reunion : everyone starts alone, closest pairs merge until it's one big tree|LA9+LA19|A12-A21-A31|USL|
|CL4|Spectral Clustering|Cutting a social network's graph along its weakest bridges|GT1+GT2+LA5|A12-A24-A37|USL|

## Domain: Dimensionality Reduction Models [DR]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|DR1|PCA|Photographing a 3D sculpture from the one angle that hides the least|LA5+LA6+LA10|A12-A21-A42|USL|
|DR2|t-SNE|A seating chart that keeps friends close and strangers far, ignoring the room's real shape|IT4+PS2|A12-A21-A42|USL|
|DR3|UMAP|t-SNE's faster cousin : preserves each neighborhood's shape, not just its position|GT1+IT4|A12-A21-A42|USL|
|DR4|LDA (Linear Discriminant Analysis)|PCA that plays favorites : uses the labels to pick directions that best separate classes|LA5+LA10+PS4|A11-A21-A42|SL|
|DR5|Autoencoder|Learning to compress a shout into a whisper, then rebuild the shout from memory|CA3+CA8+LA2|A15-A23-A38|SSL|

## Domain: Neural Network Models [NN]

| Code | Model                     | Metaphor                                                                                                                | Math Codes          | Data Codes  | Paradigm |
| ---- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------- | -------- |
| NN1  | Perceptron                | A single neuron voting yes/no off a weighted petition                                                                   | CA26+LA1            | A11-A21-A31 | SL       |
| NN2  | MLP (Feedforward)         | An assembly line of perceptrons, each station refining the last                                                         | CA4+CA12+CA3        | A11-A21-A38 | SL       |
| NN3  | CNN                       | A photographer's loupe sliding over an image, hunting for edges and patterns                                            | CA7+CA12+CA24       | A11-A23-A35 | SL       |
| NN4  | RNN                       | A reader of a sentence who can only remember the last few words                                                         | CA4+CA15            | A11-A22-A34 | SL       |
| NN5  | LSTM                      | The same reader, now carrying a notebook to remember what matters long-term                                             | CA15+LA7            | A11-A22-A34 | SL       |
| NN6  | GRU                       | LSTM's leaner sibling : same notebook idea, fewer gates to manage                                                       | CA15+LA7            | A11-A22-A34 | SL       |
| NN7  | Transformer               | A conference room where every word can talk to every other word at once                                                 | AT1+AT2+AT3+AT4+AT5 | A15-A22-A34 | SSL      |
| NN8  | ResNet                    | A relay race where any runner can hand the baton straight to someone further down the line                              | AT5+CA7+CA12        | A11-A23-A35 | SL       |
| NN9  | U-Net                     | An hourglass that squeezes an image to its essence then rebuilds it, keeping a direct line back to the original details | CA7+AT5+CA12        | A11-A23-A35 | SL       |
| NN10 | Siamese Network           | Twins raised separately, then compared side by side to see how alike they really are                                    | LA8+CA8             | A11-A21-A38 | SL       |
| NN11 | Self-Organizing Map (SOM) | Neurons arranged on a grid, competing for territory shaped like the data itself                                         | LA9+PS4             | A12-A21-A31 | USL      |
| NN12 | GAN                       | A forger and a detective locked in an arms race                                                                         | GM1+CA9             | A12-A23-A35 | USL      |
| NN13 | VAE                       | Compressing a face into a cloud of possibility, then sampling a new face from that cloud                                | IT4+CA3+LA6         | A12-A21-A35 | USL      |
| NN14 | Diffusion Model           | Sculpting a statue by slowly scraping noise off a block of static                                                       | DF1+DF2+DF3         | A15-A23-A35 | SSL      |

## Domain: NLP-Specific Architectures [NLP]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|NLP1|Word2Vec|Judging a word by the company it keeps|LA8+CA10|A15-A22-A34|SSL|
|NLP2|BERT|Reading a sentence with both eyes open, forward and backward at once, to fill in a blank in the middle|AT1+AT2+AT4+CA9|A15-A22-A34|SSL|
|NLP3|GPT|A storyteller who only ever looks backward, predicting the next word from everything said so far|AT1+AT2+AT3+CA9|A15-A22-A34|SSL|

## Domain: Graph Models [GR]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|GR1|Graph Neural Network (GNN)|Gossip passed around a friend circle until everyone knows everyone's business|GT1+GT2+LA2|A11-A24-A37|SL|
|GR2|Graph Convolutional Network (GCN)|CNN's rule ("look at your neighbors") applied to a social network instead of a pixel grid|GT1+CA7|A11-A24-A37|SL|
|GR3|Graph Attention Network (GAT)|Gossip where you don't weigh every friend's opinion equally|AT1+GT1|A11-A24-A37|SL|

## Domain: Reinforcement Learning Models [RLM]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|RLM1|Q-Learning|A trial-and-error gambler keeping a personal cheat-sheet of which bets pay off|RL1+RL2+RL5|A15-A22-A31|RL|
|RLM2|Deep Q-Network (DQN)|Q-Learning's cheat-sheet, replaced by a neural net once the sheet got too big to write down|RL1+RL2+CA3|A15-A22-A35|RL|
|RLM3|Policy Gradient / PPO|An actor adjusting tone and delivery based on the applause after every show|RL3+RL4|A15-A22-A31|RL|
|RLM4|Actor-Critic|An actor performing while a critic in the front row scores every scene in real time|RL3+RL4+RL1|A15-A22-A31|RL|
|RLM5|Monte Carlo Tree Search (MCTS)|Playing out thousands of imaginary games in your head before making one real move|RL1+RL3+PS20|A15-A22-A32|RL|

## Domain: Survival Models [SA-M]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|SA-M1|Cox Proportional Hazards|An actuary betting on when the other shoe drops, based on your risk profile|SA1+SA2+PS3|A11-A21-A31|SL|
|SA-M2|Kaplan-Meier Estimator|A survivorship scoreboard, updated every time someone drops out of the race|SA1+SA3|A11-A21-A31|SL|

## Domain: Time Series Models [TSM]

|Code|Model|Metaphor|Math Codes|Data Codes|Paradigm|
|---|---|---|---|---|---|
|TSM1|ARIMA|Predicting tomorrow's weather mostly from yesterday's, with a memory of past mistakes|TS1+TS2+TS3+TS4|A11-A22-A31|SL|
|TSM2|Exponential Smoothing (Holt-Winters)|A rumor that fades the further back it came from, but never fully disappears|TS5|A11-A22-A31|SL|
|TSM3|Prophet|ARIMA's user-friendly cousin, built to shrug off holidays and business calendars|TS1+TS5+CA8|A11-A22-A31|SL|
 
