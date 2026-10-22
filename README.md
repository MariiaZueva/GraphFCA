# Neural FCA: GraphSAGE over Classical Concept Lattices

A big homework to merge **GraphSAGE** with **Formal Concept Analysis**, inspired by the CLSAGE architecture from Shao et al. (2023) — but redesigned around a classical concept lattice with concept selection.

## Description

There is a need for interpretability in the AI field. It is usually hard to interpret the performance of graph neural networks. But graph structure alone is not the only source of information in a dataset: clusters of objects sharing attributes carry *conceptual* information that a plain graph never sees.

Shao et al. (2023) propose *graph neural networks induced by concept lattices*, where message passing over the original graph is combined with message passing over an adjacency matrix derived from a **generalized one-sided iceberg concept lattice**. That lattice handles multi-valued, ordered attributes, but it is harder to interpret than a classical lattice.

In this project you take only the **GraphSAGE (inductive)** branch of that architecture and replace the generalized one-sided iceberg lattice with a **classical concept lattice** built on binarized attributes. To keep the lattice small and meaningful, you must implement a **concept selection** step (support, stability, purity, interestingness measures) instead of the iceberg threshold. The vertices of the concept-based graph are related to sets of similar objects with similarity given by their common attributes, so they are easily interpretable. The edges between vertices are interpretable in terms of concept generality (bottom-up) or conditional probability (top-down).
### To-do list

1. Choose a dataset (Kaggle, etc.), define the target attribute, binarize data and describe the scaling (binarization) strategy for the dataset features. 
   (Ideal dataset: openly available, with various data types, with hundreds of rows, and *without* a natural graph structure — so that the cosine-similarity branch is actually needed.)

2. Perform classification using standard ML tools (at least 4), explain your choice. For example:
   - K Nearest Neighbor (kNN)
   - Naive Bayes
   - Logistic Regression
   - SVM
   - Decision Tree
   - Random Forest
   - XGBoost

3. Describe the prediction quality measure best suited for the dataset (e.g. accuracy, F1 score, or any quality measure best suited for the dataset), fit and test the network on your task.

4. Implement the **CLSAGE-style architecture** (see *Architecture* below) with the following mandatory changes:
   - build a **classical concept lattice** `B(K)` from the binarized context `K = (G, M, I)` instead of a generalized one-sided iceberg lattice;
   - implement a **concept selection** step: instead of the iceberg support threshold `p`, rank and keep the best concepts using at least one interestingness measure (support, stability, purity, or one from Kuznetsov & Makhalova 2018);
   - convert the selected concept set (together with the covering relation of the lattice) into an adjacency matrix, following Algorithm 2 of the paper but restricted to classical concepts;
   - keep the **pruning by edge density** (`β`) and the **parameter sharing** between the two branches.

5. Try to improve the basic baseline in different ways:
   - a better scaling/binarization algorithm for the original data;
   - different concept selection strategies (top-n by support, stability index, purity, etc.);
   - your own modification of the architecture (e.g. concatenation, a third branch, concept-level readout, etc.).

6. Submit a report with a comparison of all models, both standard and developed by you. Analyse your results.

### Grading system (out of 10)

- Performing steps 1–5 grants a maximum of 1/10.
- Improving the performance grants a maximum of 2/10.
- Proper report with analysis of the results grants a maximum of 3/10.
- Improving the architecture grants a maximum of 1/10.
- Presenting the work grants a maximum of 3/10.
- (p.s.!!!) Not uploading the report before the deadline / cheating will result in a 0 mark.


---


## Typical mistakes

- Using the test set to build the concept lattice (leakage).
- Enumerating the full lattice on large `M` without any concept-selection step (it blows up as `2^|M|`).
- Forgetting that the concept-lattice adjacency is **multi-valued**, and training GraphSAGE with its default boolean adjacency without probabilistic sampling.
- Reporting accuracy only, and never comparing against GraphSAGE on `A_cos` alone.
- Fixing `k` and `β` arbitrarily without a sensitivity analysis.
- Showing the full lattice when it is unreadable (use lattice fragments).
- Calling the concept list "interpretable" without reporting intent sizes, support, and purity.
- Training on the same perturbed/constructed graph for all seeds without reporting variance.

---

## What counts as an improvement (grading hints)

- **Steps 1–5 done correctly** (1/10): pipeline reproduces the CLSAGE architecture with a classical lattice + concept selection, and baselines are trained.
- **Performance improvement** (2/10): concept selection and/or β-tuning beats plain GraphSAGE on `A_cos`, *with* variance reported over ≥ 5 seeds.
- **Report with analysis** (3/10): comparison table for all models, ablations, sensitivity to `k` and `β`, discussion of interpretability vs. accuracy.
- **Architecture improvement** (1/10): e.g. concept-level readout, a third branch, per-concept weights — anything beyond the vanilla Shao et al. fusion.
- **Presentation** (3/10): clear explanation of the method, the key ablations, and at least one visualised lattice fragment used for prediction.

---

### References

1. Mingwen Shao, Zhiyong Hu, Weizhi Wu, Huan Liu. "Graph neural networks induced by concept lattices for classification." *International Journal of Approximate Reasoning* 154 (2023) 262–276. — the source of the CLSAGE architecture.
2. Kuznetsov, Sergei O., Nurtas Makhazhanov, and Maxim Ushakov. "On neural network architecture based on concept lattices." *Foundations of Intelligent Systems: 23rd International Symposium, ISMIS 2017*, Springer, 2017.
3. Kuznetsov, S. O., & Makhalova, T. (2018). On interestingness measures of formal concepts. *Information Sciences*, 442, 202–219.
4. Hamilton, W. L., Ying, R., Leskovec, J. "Inductive representation learning on large graphs." *NIPS 2017*.
5. https://github.com/EgorDudyrev/FCApy
6. https://github.com/smartFCA/caspailleur
7. https://github.com/smartFCA/paspailleur
