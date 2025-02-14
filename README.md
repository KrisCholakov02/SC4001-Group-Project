# SC4001 Neural Networks and Deep Learning - Group Project

## Project Title: **Speech and Text Emotion Recognition Using Deep Learning**

### Group Members:
- **Cholakov Kristiyan Kamenov** (U2123543B)
- **Mishra Pradyumn** (U2123912E)
- **Denzyl David Peh** (U2122190F)

---

## 📌 Project Overview

In the rapidly advancing fields of **Artificial Intelligence (AI)** and **Virtual/Augmented Reality**, emotion recognition is crucial for improving human-computer interaction. Traditional models often rely on either **text-based** or **speech-based** emotion detection, failing to capture the full emotional context.

This project **bridges the gap** by integrating **text emotion recognition** with **speech-based emotion recognition** using deep learning techniques. Our **fusion model** combines the strengths of **Convolutional Neural Networks (CNNs)** for text processing and **Multi-Layer Perceptrons (MLPs)** for speech-based feature extraction, achieving improved accuracy in emotion classification.

---

## 🚀 Key Contributions & Novelty

### 🌟 **Fusion of Text and Audio-Based Emotion Recognition**
Unlike existing models that focus on a single modality (text or speech), our system **combines both**, leveraging OpenAI’s **Whisper model** for speech-to-text conversion. This dual-input approach ensures a more **accurate** and **context-aware** emotion classification.

### 🏆 **Improved Emotion Recognition Accuracy**
By integrating **audio feature extraction** (using **Librosa** and **openSMILE**) with **text analysis**, our fusion model **outperforms** standalone text and audio models. The final model achieves **71.87% validation accuracy**, higher than CNN (**60.04%**) and MLP (**68.91%**) alone.

### 🔍 **Comprehensive Model Selection & Evaluation**
- **CNN** was selected for text-based classification after comparisons with GRU, Bi-GRU, and LSTM models.
- **MLP** was found to be the most effective for audio-based emotion classification.
- The **fusion model** was designed to dynamically weight predictions from both models, **optimizing performance**.

### 🗣 **Robust Speech-to-Text Processing**
- **OpenAI's Whisper model** was chosen for its superior performance over wav2vec 2.0, especially in noisy environments.
- Evaluated across multiple datasets, achieving **WER = 7.5%** with the large Whisper model.

---

## 📂 Project Structure

- `data/` - Contains the dataset used for the project.
- `whisper_evaluation/` - Contains the code for Whisper Evaluation.
- `iemocapTrans.csv` - Contains data from the IEMOCAP dataset ([Dataset Link](https://www.kaggle.com/datasets/mouadriali/iemocap-transcriptions-english-french)).
- `text_models_analysis.ipynb` - Jupyter notebook for analyzing the text-based emotion recognition models.
- `data.ipynb` - Jupyter notebook for data preprocessing.
- `fused_model_main.ipynb` - Jupyter notebook implementing the final **fusion model**.
- `hyperparameter_*.ipynb` - Jupyter notebooks for **hyperparameter tuning** of different models.
- `requirements.txt` - Contains dependencies required to run the project.
- `*.pt` - Contains trained models.

---

## 📊 Model Architecture

### 📝 **Text Emotion Recognition (CNN)**
- **Embedding Layer** for text vectorization.
- **Multiple convolutional layers** with varying kernel sizes.
- **Max pooling & dropout layers** for improved generalization.
- **Softmax classifier** for predicting emotion classes.

### 🔊 **Speech Emotion Recognition (MLP)**
- **Feature extraction** using **Librosa** (MFCC, Mel-Spectrogram) & **openSMILE**.
- **MLP model** with two hidden layers (256 and 128 neurons).
- **ReLU activation** and dropout for regularization.
- **Softmax classifier** for predicting emotion classes.

### 🔗 **Fusion Model**
- Merges predictions from CNN (text) and MLP (audio).
- Uses a **fully connected layer** to refine the final prediction.
- Achieves **higher accuracy than individual models**.

---

## 📥 Running the Code

### 🔧 **Setup Instructions**
1. **Download** the required datasets:
   - [IEMOCAP dataset](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease)
   - [Processed training & testing data](https://entuedu-my.sharepoint.com/:u:/g/personal/kristiya001_e_ntu_edu_sg/EfZGKzUHznJHmHPrR8G96sABZUSOtPDlKB7eUCAxgENLWA?e=Vc8bmP)

2. **Extract** the datasets:
   - Place the `IEMOCAP` dataset in the `../` directory (or update the path in `data.ipynb`).
   - Extract other datasets into the `data/` folder.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks in the following order**:
   - `data.ipynb` - Preprocesses the dataset.
   - `text_models_analysis.ipynb` - Runs CNN-based text emotion recognition.
   - `hyperparameter_bs_tuning.ipynb`
   - `hyperparameter_embd_dim.ipynb`
   - `hyperparameter_filter_sizes.ipynb`
   - `hyperparameter_num_neurons_tuning.ipynb`
   - `hyperparameter_numfilters.ipynb`
   - `fused_model_main.ipynb` - Runs the final **fusion model**.
   - `whisper_evaluation/whisper_evaluation.ipynb` *(Optional for Whisper Evaluation)*

---

## 📈 Results & Performance

| **Model**   | **Validation Accuracy** |
|------------|--------------------|
| **CNN (Text-Based)**  | 60.04% |
| **MLP (Audio-Based)**  | 68.91% |
| **Fusion Model (Text + Audio)**  | **71.87%** |

- **Fusion model outperforms standalone models** by effectively combining textual and audio features.
- **Performance bottleneck**: Small dataset size (~4500 data points) limits generalization.
- **Future work**: Dataset augmentation & fine-tuning preprocessing techniques.

---

## 📌 Future Improvements

- **Data Augmentation**: Increase dataset size for better generalization.
- **Transformer-based models**: Explore **BERT** for text and **wav2vec** for audio.
- **Emotion granularity**: Predict nuanced emotional states rather than broad categories.

---

## 📚 References

1. [Interactive Emotional Dyadic Motion Capture (IEMOCAP)](https://sail.usc.edu/iemocap/)
2. OpenAI's [Whisper Speech-to-Text Model](https://arxiv.org/abs/2212.04356)
3. Librosa: [Audio Feature Extraction](https://librosa.org)
4. OpenSMILE: [Speech Feature Extraction](https://www.audeering.com/opensmile/)

---

## 🔗 Additional Links

- [Training Data](https://entuedu-my.sharepoint.com/:u:/g/personal/kristiya001_e_ntu_edu_sg/EfZGKzUHznJHmHPrR8G96sABZUSOtPDlKB7eUCAxgENLWA?e=Vc8bmP)
- [IEMOCAP Dataset](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease)
- [Whisper Evaluation Data](https://www.openslr.org/12)

---

## 🛠️ Acknowledgments

We would like to thank **Nanyang Technological University (NTU)** for providing the resources to conduct this research, as well as OpenAI for their **Whisper** model, which played a crucial role in improving our text-based emotion recognition pipeline.

---

This project represents a **step forward in multimodal emotion recognition**, merging speech and text analysis to enhance **AI’s understanding of human emotions**. 🚀
