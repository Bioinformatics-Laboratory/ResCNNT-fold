# ResCNNT-fold
ResCNNT-fold: Combining Residual Convolutional Neural Network and Transformer for Protein Fold Recognition from Language Model Embeddings

In this study, we introduce a predictor, ResCNNT-fold, for protein fold recognition and employ the LE dataset for testing purpose. ResCNNT-fold leverages a pre-trained language model to obtain embedding representations for protein sequences, which are then processed by the ResCNNT feature extractor, a combination of residual convolutional neural network and Transformer, to derive fold-specific features. Subsequently, the query protein is paired with each protein whose structure is known in the template dataset. For each pair, the similarity score of their fold-specific features is calculated. Ultimately, the query protein is identified as the fold type of the template protein in the pair with the highest similarity score.

## Requirements
pip install numpy 
pip install pandas 
pip install sklearn 
pip install pytorch 
pip install h5py 
pip install matplotlib 

## Usage
To realize protein fold recognition through ResCNNT-fold, run the following code.

### Step 1: Obtain the datasets.
The training dataset and the benchmark dataset LE can be obtained from the “data” folder.
### Step 2: Obtain the language model embeddings.
Using the ProSE protein language model from the “language model” folder allows the generation of language model embeddings for each protein sequence.
### Step 3: Construct the feature extractor based on deep model named ResCNNT.
Using the language model embeddings generated from the training dataset as input, running the train_nn.py in the “train_model” folder can construct the ResCNNT feature extractor. 
### Step 4: Realize protein fold recognition.
Utilizing the feature extractor generated in step 3, feature extraction is performed on a given query protein sequence. Using the cosine similarity metric function from the “classificationORrecognition” folder, similarity scores between the query protein sequence and each protein in the template protein dataset are calculated. Finally, the query protein is assigned to the fold type of the protein with the highest similarity score.






