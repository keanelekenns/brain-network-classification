## Review Synopsis

### Overall merit
4. Accept

### Reviewer expertise &mdash; hidden from auhors

I am knowledgeable on this topic

### Brief summary of the paper

This paper repurposes algorithms used to find locally optimal dense subgraphs and applies them to find discriminating connections in two classes of brain networks. By creating summary graphs for each class, where edge weights represent the percentage of subjects containing that edge in their brain network, the authors were able to find discriminating subgraphs called contrast subgraphs by searching for dense subgraphs within a difference network; this difference network is obtained by subtracting one summary graph from another. Once contrast subgraphs were identified, they could be used to classify new brain networks based on the number of edges present in each subgraph. The paper compares this classification technique to four other methods, and produces positive results in terms of accuracy and explainability.

### Contributions of the paper

- The authors introduce the novel problem of finding contrast subgraphs in two classes of networks (i.e. a graph that is dense in one network and not the other) defined over the same set of vertices (in this case, brain networks with the same brain regions).
- The authors apply their method to a dataset of brain networks consisting of individuals affected by Autism Spectrum Disorder and those who are not. They indicate that the results of the analysis match with existing domain literature, and they note the simplicity and high explainability of their method through the analysis.
- The authors compare their method to other classification techniques in terms of accuracy.

### What are the strengths of this paper?

- The paper is highly understandable and uses simple and clear language.
- The paper makes effective use of figures and graphs to convey understanding to the audience.
- The contributions of the paper are clearly stated, and examples are used to explain the problem statement.
- The paper indicates techniques that were attempted, but ended up being ineffective.
- The paper is clear about parameter tuning when comparing their method to other techniques.
- The paper references a github repository containing some of the programs for the study.

### What are the weaknesses of this paper?

- The included github repository does not contain the code used to test the classifiers as mentioned in section 5.2.
- The paper is not clear about the technique used to determine the decision boundary for each dataset as seen in figures 1, 5, 6, 7, and 8.
- The paper makes a critical typo when describing a rule for figure 5.
- The paper does not give any explanation for how algorithms 3 and 4 work (especially SDP).

## Comments to the authors based on review criteria
Review criteria: Significance, Soundness, Novelty, Verifiability & transparency, Presentation

### Significance

I believe the focus on finding an explainable method is a good precedent to set for future research in this area, and the proposed novel method is a good starting point for future research to explore. However, I think more justification as to why an interpretable method is needed would be useful. Also, it would be good to have a more clear explanation of the significance of this paper.

### Soundness
   
Using accuracy alone as a metric for the classification techniques can be misleading; metrics such as precision, recall, and f1-score could give a better picture of where the technique stands among other methods and would require little additional work. However, it was good to see so much work put into implementing other methods and considering the neuroscience literature.

### Novelty

The paper does not reference this research (https://snap.stanford.edu/gnnexplainer/) that was accomplished prior to its publication. It has a very similar goal of explainability. The paper does not create new algorithms itself, but repurposes the algorithms of previous works to accomplish a new goal. However, this repurposing is useful, and is helpful to inform researchers of what has already been accomplished.

### Verifiability & transparency

This paper does a very good job of this for the most part. It describes the location and processing of the datasets, provides a repository with the code used to implement the described algorithms, and indicates which specific implementations were used for comparison between techniques. However, the programs and resources for verifying the accuracy of the developed classifier were left out of the github repository, and furthermore, the process of determining a decision boundary for each dataset was not explained, though the boundary is roughly shown in the figures presented.

### Presentation

Very good work in this area. Aside from the critical typo while explaining the rule for figure 5, and a small number of minor typos in the rest of the paper, I have nothing to critique.

### Questions for authors

1. How could readers follow the progress of your efforts to create new datasets for Bipolar Disorder
and Schizophrenia?
2. I have not been able to find any code related to evaluating the accuracy of the classification methods used in the paper (i.e. classifying test data as asd or td based on number of edges in each contrast subgraph or L1 norm of induced subgraphs). Would you be able to identify or provide any materials you used for this evaluation?
3. I have seen code referencing the ego scan method in Cadena et. al’s paper, but it looks like you did not use that in your paper. Was this something you had hoped to implement but did not have time for?
4. Out of curiosity, may I ask the reasoning behind using Matlab’s SDP solver rather than options in python such as http://cvxopt.org/?