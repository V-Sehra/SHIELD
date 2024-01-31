# ShIeLD: Spatially-Enhanced Immune Landscape Decoding in HCC using Graph Attention Networks and CODEX Data

![GAT.pdf](https://github.com/V-Sehra/ShIeLD/files/14113477/GAT.pdf)

High dimensional spatial molecular imaging data is transforming our understanding of tissue microenvironment, crucial in complex diseases like hepatocellular carcinoma (HCC). However, inferring spatial cell type interaction in disease contexts directly from imaging data remains challenging. Recent advancements in deep learning, specifically graph neural networks, have been employed to integrate spatial and cellular information for immune response analysis in HCC. These networks effectively encode spatial arrangements, enhancing our understanding of cellular interactions and spatial organization of tissue .

Results: We introduce ShIeLD (Spatially-enHnced Immune Landscape Decoding): ShIeLD utilizes Graph Attention Networks (GAT) to provide interpretable predictions. Specifically ShIeLD infers putative disease- associated cell type interactions by means of summarizing the model’s learned attention scores at the cell type level. We report an application of the model to spatial imaging study of liver tissue from hepatocellular liver cancer patients and focus on the differential immune cell interactions across healthy and tumor tissue. We identify a recently discovered subtle interaction between mucosa associated invariant T cells (MAITs) and macrophages without additional supervision.This approach underscores the potential of attention-based graph models in enhancing the understanding of complex diseases like liver cancer.

## Installation

Activate a virtual environment, e.g. using conda. Install dependencies:

```
git clone git@github.com:V-Sehra/ShIeLD.git
cd ShIeLD
```
