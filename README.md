# SC4001 Neural Networks and Deep Learning - Group Project
### Group Members:
- Mishra Pradyumn (U2123912E)
- Denzyl David Peh (U2122190F)
- Cholakov Kristiyan Kamenov (U2123543B)

## Project Structure:
- `data/` - Contains the dataset used for the project.
- `whisper_evaluation/` - Contains the code for the Whisper Evaluation.
- `iemocapTrans.csv` - Contains data from the IEMOCAP dataset. Link: https://www.kaggle.com/datasets/mouadriali/iemocap-transcriptions-english-french
- `text_models_analysis.ipynb` - Jupyter notebook containing the code for analyzing the text models.
- `data.ipynb` - Jupyter notebook containing the code for data preprocessing.
- `fused_model_main.ipynb` - Jupyter notebook containing the code for the fused model, the proposed model from this project.
- `hyperparameter_*.ipynb` - Jupyter notebooks containing the code for hyperparameter tuning of the models.

## Running the code:
1. Download the data used for training and testing the models from the following link: https://entuedu-my.sharepoint.com/:u:/g/personal/kristiya001_e_ntu_edu_sg/EfZGKzUHznJHmHPrR8G96sABZUSOtPDlKB7eUCAxgENLWA?e=Vc8bmP
2. Extract the contents of the zip file to the `data/` directory.
3. Download the IEMOCAP dataset from the following link: https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease
4. Place the IEMOCAP you just downloaded in `../` directory, or change the path in the `data.ipynb` notebook.
5. `pip install -r requirements.txt` to install the required dependencies.
6. Run the notebooks in the following order:
    - `data.ipynb`
    - `text_models_analysis.ipynb`
    - `hyperparameter_bs_tuning.ipynb`
    - `hyperparameter_embd_dim.ipynb`
    - `hyperparameter_filter_sizes.ipynb`
    - `hyperparameter_num_neurons_tuning.ipynb`
    - `hyperparameter_numfilters.ipynb`
    - `fused_model_main.ipynb`
    - `whisper_evaluation/whisper_evaluation.ipynb` (optional for Whisper Evaluation)