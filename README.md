
# ShIeLD: Spatially-enhanced Immune Landscape Decoding in Hepatocellular Carcinoma (HCC) using Graph Attention Networks and CODEX Data
![GAT.pdf](https://github.com/V-Sehra/ShIeLD/files/14113477/GAT.pdf)

ShIeLD employs Graph Attention Networks (GAT) to decode complex immune landscapes in HCC from high-dimensional spatial molecular imaging data, providing insights into tissue microenvironments. This approach enables the interpretation of disease-associated cell type interactions, highlighted by the identification of interactions between MAITs and macrophages, without additional supervision. The project demonstrates the potential of attention-based graph models in advancing our understanding of HCC.

## Installation

To set up ShIeLD, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/V-Sehra/ShIeLD.git
   ```
2. Navigate to the ShIeLD directory:
   ```
   cd ShIeLD
   ```
3. Install Poetry if not already installed:
   ```
   conda install poetry -c conda-forge
   ```
4. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Workflow

### Data Preparation

1. Download the CODEX images dataset as described in the paper (link to dataset publication).
2. Place the raw CSV file containing CODEX images information under `../data/raw_data.csv` or specify a custom path using `-path_to_raw_csv` flag in `data_preprocessing.py`.

To preprocess the data and generate input graphs, run:
```
python data_preprocessing.py
```

### Model Training

To train the ShIeLD model, execute:
```
python run_training.py
```
The script saves the trained model automatically.

### Extracting Cell Phenotype Attention Scores

After training, evaluate the model's attention on cell phenotypes by running:
```
python run_attention_evaluation.py
```
This generates a dictionary with phenotype-to-phenotype attention scores accessible via `[normed_p2p]` for each input graph.


## Citation

If you use ShIeLD in your research, please cite our paper. (Bibtex TBA)

## License

ShIeLD is open-sourced under the AGPL-3.0 license. See the LICENSE file for more details.

