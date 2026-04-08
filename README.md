# SHIELD: **S**patially-en**H**anced Immun**E** **L**andscape **D**ecoding  
*Weakly supervised graph attention networks for interpretable cell-cell interaction analysis*

<img width="1575" height="1487" alt="only_Colobar_Shield" src="https://github.com/user-attachments/assets/77e42ea4-3756-4a01-b539-ec203d9c9984" />



**SHIELD** is a framework that leverages *Graph Attention Networks (GATs)* to infer and interpret disease-associated cell-cell interactions from spatial omics data. It highlights which immune interactions are most predictive of disease outcomes, by summarizing attention scores at the cell-phenotype level — enabling biologically meaningful insights.

We applied SHIELD to multiplexed tissue imaging data in two key disease contexts:

- **Hepatocellular carcinoma (HCC)**: SHIELD identified differential immune interactions between tumor and adjacent healthy tissue, including a subtle but significant interaction between **MAIT cells** and **macrophages**, supporting previous findings on tumor progression.
- **Diabetes**: Using imaging mass cytometry from diabetic pancreas tissue, SHIELD revealed dynamic immune-β cell interactions, uncovering stage-specific communication patterns between **helper T cells**, **cytotoxic T cells**, and **β cells**.
- 	**Colorectal cancer (CRC)**: SHIELD uncovered a suppressive interaction between **CD8⁺ T cells** and **CD11b⁺CD68⁺ macrophages** enriched in non-responders. This finding refines previous observations of immune-suppressive neighborhood coupling in CRC and highlights a potential mechanism of immunotherapy resistance.

In both cases, SHIELD recovered experimentally validated interactions and demonstrated its strength in extracting interpretable and spatially-aware immune landscape features.

---

## 🔧 Installation

To set up SHIELD locally, follow these steps:

1. Clone the repository:
   
```
git clone https://github.com/V-Sehra/ShIeLD.git
```
3.	Navigate to the project directory:
   
   ```
   cd ShIeLD
   ```
3.	Install Poetry if not already installed:
   
   ```
   conda install poetry -c conda-forge
   ```

5.	Install project dependencies:

   ```
   poetry install
   ```

## 🧭 Workflow Overview

SHIELD is built to process immune spatial data in a reproducible and modular way. Example inputs are provided in the example/ folder, including dictionary-based data specs that describe paths, phenotypes, and output locations.

⸻

## 📂 Data Preparation

To construct spatial graphs from your input data:
   ```
   python create_graphs.py
   ```

## 🧪 Model Training

To perform a hyperparameter search:
   ```
   python train_model.py
   ```

•	Runs a search using your specified config.
•	Saves training and validation metrics in CSV format for all models.

To analyze the hyperparameter search and retrain the best model:
   ```
   python evaluate_hyperearch.py
   ```

•	Generates plots to evaluate the search results.

•	Saves the best configuration as best_config_dict.

•	Retrains the best model on the full training set.

•	Saves final train/test performance.


## 🔍 Interaction Scoring

To calculate cell-to-cell interaction scores using the trained model:
   ```
   python create_attention_score_interaction.py
   ```

•	Loads the trained model.

•	Computes phenotype-to-phenotype attention scores.

•	Saves both plots and raw scores for downstream interpretation.



 ## 📖 Citation

If you use SHIELD in your research, please cite our paper:


Citation: TBA (pre accept as Cell Patterns)


## 🪪 License

SHIELD is released under the AGPL-3.0 License. See the LICENSE file for more details.
