# DIST-VERSON1

## Dual-Input Spatio-Temporal Transformer Model: Predicting the Efficacy of Neoadjuvant Chemotherapy in Breast Cancer Based on DCE-MRI Images

**Abstract**
To address the challenges of predicting the effectiveness of neoadjuvant chemotherapy (NACT) for breast cancer, this paper presents a Dual-Input Spatio-Temporal Transformer (DIST) model utilizing DCE-MRI images. The model integrates three core embedding modules, which significantly improve prediction accuracy and help avoid unnecessary treatments. Firstly, the enhanced Tokens-to-Token Patch Embedding (EiT2T) module maintains and amplifies structural features such as edges and textures through multi-scale feature extraction and a combined local-global attention mechanism, improving lesion detection in intricate images. Secondly, the Spatio-Temporal Embedding (ST) module merges dynamic positional and temporal embeddings within each token, enabling the model to accuractely capture and analyze pathological changes over time. Finally, the Adaptive Feature Fusion and Classification (AFFC) module combines features from various time points and adaptively models their differences, boosting the model's ability to discriminate. Using both pre-chemotherapy and post-first-cycle chemotherapy imaging, the proposed DIST model achieves an AUC of 0.924 and an accuracy of 86.9% on a proprietary breast cancer dataset, outperforming the best existing DiT model by 4.1% and 2.9%, respectively.
These outcomes, particularly with multi-time-point DCE-MRI data, confirm our model's superior performance and practical viability.


## Data
We use a private DCE-MRI dataset and the public TCIA I-SPY2 dataset to validate the proposed DIST model. Due to privacy concerns, only a portion of the private dataset is shown as example data to illustrate its characteristics. If you require access to the full dataset, please contact the corresponding author via email.
TCIA I-SPY2 dataset can be obtained from the [website](https://www.cancerimagingarchive.net/collection/ispy2/).


## Main modules
We have open-sourced the **core components** of **DIST (v1)**:
- **EiT2T (Enhanced image-to-token)**: multi-scale conv fusion + local/global self-attention  
- **ST (Spatial–Temporal embedding)**: dynamic position embedding + time embedding  
- **AFFC (Adaptive Feature Fusion & Classification)**: adaptive weighting of before/after features for classification  
- **Backbone:** built upon a Transformer encoder with configurable depth, heads, and MLP blocks

> The complete training pipeline, dataset adapters (e.g. DICOM loader), and pretrained weights are being collated and will be released soon.

