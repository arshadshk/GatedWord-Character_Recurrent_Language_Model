# Gated_Word-Character_Recurrent_Language_Model
(pytorch)

This repo is implementation of LSTM model from the paper [Gated Word-Character Recurrent Language Model](https://arxiv.org/pdf/1606.01700.pdf). The main use case as described in the paper is to utilize both the word and character level inputs.  
Model can be used in 3 models[1] -   
* Word and character -
This model simply concatenates the vector representations of a word constructed from the character input and the word input to get the final representation of a word.  
X(word) = [X(char) ; X(word)]
* Gated word and character, Fixed value -
This model uses globally constant gating value to combine the word representation created by the character inputs and and the word inputs.  
X(word) = (1-g) * X(word) + g * X(char)  
Where g is a value between 0 and 1.  
In the paper the value of g is choosen from - {0.25, 0.5, 0.75}.  
* Gated Word and  Character adaptive - This model uses adaptive gating values to combine vector representation of word constructed from character inputs.     

![Model Architecture](https://github.com/arshadshk/GatedWord-Character_Recurrent_Language_Model/blob/master/model_architecture.JPG)  

The gate can be efficiently trained so that the
model can find a good balance between the word level and character-level inputs.gate that adaptively finds the optimal mixture of the character-level and wordlevel inputs. The gate creates the final vector representation of a word by combining two distinct representations of the word. The character-level inputs are converted into vector representations of words using a bidirectional LSTM. The word-level inputs are projected into another high-dimensional space by a word lookup table. The final vector representations of words are used in the LSTM language model which predicts the next word given all the preceding words. Our model with the gating mechanism effectively utilizes the character-level inputs for rare and out-ofvocabulary words and outperforms word-level language models on several English corpora.[1]


References:-  
[1] Paper {miyamoto2016gated,  
    title={Gated Word-Character Recurrent Language Model},  
    author={Yasumasa Miyamoto and Kyunghyun Cho},  
    year={2016},
    link={https://arxiv.org/abs/1606.01700v2}