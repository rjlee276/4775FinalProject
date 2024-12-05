# CS4775 Final Project:

## Characterization and Evolution of DNA Motifs in Mammalian Transcription Factor Binding Sites

### Group Members:

- Ash Pagedemarry (ap828)
- Ryan Lee (rjl275)
- Jack Chen (jc2742)
- Mohammed Islam (mti5)
- Noam Canter (nec53)

---

### **Motivation**

We seek to analyze the evolutionary conservation and divergence of DNA motifs associated with transcription factor binding sites across multiple mammalian species. By reimplementing and benchmarking motif clustering algorithms—such as hierarchical agglomerative clustering, K-medoids clustering, and Hidden Markov Models—we aim to cluster DNA motifs obtained from public databases like JASPAR.

Our objective is to compare the performance of these algorithms in identifying conserved motifs and reducing redundancy in motif identification. Through a comparative analysis, we seek to understand how evolutionary processes influence motif conservation and transcription factor functionality across different mammals.

The findings may provide insights into gene regulation mechanisms and enhance the effectiveness of computational methods in genomics research.

---

### **Installation and Setup**

1. **Install conda** (or confirm that it's already installed on your system).
2. Use conda to install Python and the required libraries into a new virtual environment (to keep it isolated from other projects).
3. Activate your new Python environment.

#### **Commands to Set Up the Environment**

```bash
conda update conda
conda config --add channels defaults
conda config --add channels conda-forge
conda config --add channels bioconda
conda create --name bioinfo --file env_setup/conda_packages.txt
```

#### **Run the Clustering**

```bash
python main.py
```
