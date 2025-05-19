# CNN-LSTM-Transformer_Hybrid_DNA_Classification (DeepHybrid)

**Authors: Sunil Kumar, Ajit Bhandari, Aryan Gupta, Chanddan H, Biswajit Bhowmik**

This project implements a hybrid CNN-LSTM-Transformer model for classifying viral DNA sequences. The approach incorporates adaptive K-mer encoding for optimal feature extraction and model performance. This README provides detailed instructions on the dataset, included source files, and how to use them.

---

## Included Files:
1. **merge.csv**: The dataset containing over 9000 DNA samples used for training and evaluation.
2. **CNN_model.ipynb**: Implementation of a standalone CNN-based model.
3. **LSTM_model.ipynb**: Implementation of a standalone LSTM-based model.
4. **Transformer_model.ipynb**: Implementation of a standalone Transformer-based model.
5. **Hybrid_Onehot.ipynb**: Hybrid model using One-Hot encoding for feature representation.
6. **Hybrid_model.ipynb**: Final hybrid CNN-LSTM-Transformer model integrating adaptive K-mer encoding.

---

## Dataset Details:
- **File**: `merge.csv`
- **Description**: The dataset contains viral DNA sequences from a range of pathogens, making it suitable for evaluating the robustness and scalability of deep learning models.
- **Virus Types and Sample Distribution**:

| Virus Type       | Number of Sequences |
|------------------|---------------------|
| Influenza        | 2250               |
| Dengue           | 2250               |
| SARS-CoV-2       | 2250               |
| MERS             | 1908               |
| SARS-CoV-1       | 250                |
| Ebola            | 250                |

- **Format**:
  - Column 1: DNA Sequence
  - Column 2: Class Label (categorical labels for classification tasks).

- **Encoding Method**: 
  - Adaptive K-mer encoding:
    - Sequence length < 500: K-mer size = 3
    - 500 ≤ Sequence length < 1000: K-mer size = 4
    - Sequence length ≥ 1000: K-mer size = 6

---

## How to Run:
### Step 1: Install Prerequisites
- Ensure you have Python installed along with the following libraries:
  - TensorFlow
  - Keras
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
- Install libraries using:
  ```bash
  pip install tensorflow keras numpy pandas matplotlib scikit-learn
  ```

### Step 2: Update Dataset Path
- Replace `<dataset_path>` in the notebooks with the full path to `merge.csv`.

### Step 3: Select and Run a Model
1. Open the desired notebook using Jupyter Notebook or JupyterLab:
   - For the **main hybrid model**, use `Hybrid_model.ipynb`.
   - For standalone architectures, open the respective files:
     - `CNN_model.ipynb`
     - `LSTM_model.ipynb`
     - `Transformer_model.ipynb`
   - For an alternative approach with One-Hot encoding, open `Hybrid_Onehot.ipynb`.

2. Execute all cells in the notebook to:
   - Load the dataset.
   - Preprocess data using adaptive K-mer encoding.
   - Train and evaluate the model.
   - Generate classification metrics and visualizations.

### Step 4: View Results
- Outputs include metrics like accuracy, precision, recall, F1-score, and visualizations:
  - Confusion matrix.
  - ROC and Precision-Recall (PR) curves.

---

## Model Highlights:
1. **Hybrid Model**:
   - Combines CNN, LSTM, and Transformer architectures.
   - Captures local (CNN), sequential (LSTM), and global (Transformer) dependencies in DNA sequences.
   - Robust performance across diverse datasets and configurations.

2. **Standalone Models**:
   - Individual CNN, LSTM, and Transformer models are included for comparative evaluation.

3. **One-Hot Encoding Variant**:
   - Uses traditional One-Hot encoding as an alternative to K-mer encoding for feature representation.

---

## Results Summary:
- **Performance on Dataset**:
  - The hybrid model achieves:
    - Accuracy > 99%
    - Precision > 99%
    - Recall > 99%
    - F1-score > 99%
  - Tested on various train-test splits (80-20, 70-30, 60-40, 50-50) and dataset sizes.

- **Scalability**:
  - Maintains high accuracy across increasing dataset sizes (10%, 25%, 50%, 75%, 100% of total samples).

- **Key Strengths**:
  - Adaptive K-mer encoding dynamically adjusts feature extraction based on sequence length.
  - Handles imbalanced and variable-length DNA data effectively.

---

## Notes:
- Ensure the dataset `merge.csv` is placed in the same directory as the notebooks or update the dataset path accordingly.
- For optimal performance, a system with GPU acceleration is recommended.

---
