# Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction
Code for the paper "Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction".

## Key Features
- Black-box hallucination detection: tracking of token embedding trajectories.
- Resource efficient: single-sample and retrieval-free classification, bypassing expensive sampling/RAG loops.
- Extended DMD and lifting: captures high-order nonlinear manifold dynamics through polynomial observable lifting.


## Repository Structure
```
├── main.m                         # main script
└── utils/                         
    ├── load_training_data.m       # for loading embeddings of LLM responses
    ├── compute_svd_basis.m        # for Singular Value Decomposition basis projection space
    ├── lifting_function.m         # for state lifting
    ├── fit_dynamical_systems.m    # for linear regression Koopman DMD solver
    └── evaluate.m                 # for threshold-sweeping and evaluation on test data
```
## Configuration and adaptation
When adapting this code to a new evaluation dataset or a different LLM/embedding architecture, modify the data path `DATA_DIR` and the parameter `EMB_DIM` in the `main.m` file [here](https://anonymous.4open.science/r/hallucination-detection-DS-EC07/main.m#L10-L11).
When handling custom datasets with longer or shorter text sequences, adjust the sequence lengths `tokens_length_threshold` [here](https://anonymous.4open.science/r/hallucination-detection-DS-EC07/main.m#L17) to evaluate our method over various LLM responses' lengths.

## Running the code
Raw data for embedding tensors can be downloaded from [this link](https://www.data-research.store). Running the main file on the embeddings of the models `Jina-v5-Nano`, `Qwen3`, `F2LLM`, `Mistral`, and `Llama-3` will display the subfigures in figure 5 of the paper.
