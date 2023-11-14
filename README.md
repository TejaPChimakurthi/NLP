## NLP Learning 

The advent of transformers marked a revolutionary moment in the field of Natural Language Processing (NLP), elevating the utilization of Language Model (LLM). GPT, in particular, has propelled NLP to new heights, representing the pinnacle of transformative advancements.

To delve into these cutting-edge developments, it is essential to grasp the fundamentals of NLP. The journey begins with tokenization, stemming, and handling stop words. In NLP use cases, where the input is comprised of words, machines necessitate a translation of this textual information into numerical data. Various methods such as Bag of Words, Term Frequency-Inverse Document Frequency (TF-IDF), Word2Vec, AvgWord2Vec, and embedded layers (in deep learning) facilitate this conversion.

However, traditional methods present drawbacks such as sparsity in matrices and the failure to capture semantic meaning. Addressing these challenges, Word2Vec and AvgWord2Vec manage to overcome issues related to semantic meaning and sparsity. Yet, when applying machine learning models, the crucial consideration of context is often overlooked.

To capture the context, we turn to deep learning models. Models like Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) RNN, Gated Recurrent Unit (GRU) RNN, Bidirectional LSTM RNN, encoders, and decoders, as well as Transformers, come into play. These models excel in comprehending the context of preceding words, allowing them to predict future words accurately.

Here's a step-by-step guide for those embarking on their NLP learning journey:

Step 1: Tokenization, stop words, stemming, lemmatization (Libraries: NLTK, spaCy)

Step 2: Bag of words, TF-IDF (Libraries: scikit-learn; BOF: CountVectorizer, TF-IDF: TfidfVectorizer)

Step 3: Word2Vec, AvgWord2Vec (Libraries: gensim)

Step 4: Practical implementation using the above techniques.

Transitioning to deep learning, it's crucial to grasp the basics of its components:

Basics of neural networks (input layer, hidden layer, and output layer)
Forward propagation
Backward propagation
Activation function
Loss function
Optimizers
Step 5: Recurrent Neural Network (Theory and practical implementation)

Step 6: LSTM RNN and GRU RNN (Theory and practical implementation)

Step 7: Bidirectional LSTM RNN (Theory and practical implementation)

Step 8: Encoders and Decoders (Theory and practical implementation)

Step 9: Transformers (Theory and practical implementation)

For Steps 5-9, PyTorch library is recommended, and for pre-trained transformers, the Hugging Face library is invaluable.

Embark on this structured learning path to master the intricacies of NLP, from foundational concepts to advanced deep learning models. Happy learning!

The NLP_ML.ipynb file contains the pratical implementation all machine learning topics from step 1-3. 

The NLP_DL.ipynb file contains pratical implementation of Simple RNN, LSTM RNN (Uni and Bi directional), GRU RNN (Uni and Bi directional)

encoder_decoder_seq2seq_basic contains Encoder and decoder implementation using PYTORCH. I recommand to follow the bentrevett github repo pytorch-seq2seq for sequence2sequence models.

I learned this from NLP live session from Krish naik youtube channel. Its have amazing videos for NLP beginners. I recommed you watch these videos if you are new to NLP. THANK YOU KRISH.

https://www.youtube.com/watch?v=w3coRFpyddQ&list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn <br>

Additional blogs and github resources:

https://colah.github.io/  <br>
https://jalammar.github.io/illustrated-transformer/  <br>
https://github.com/krishnaik06/The-Grand-Complete-Data-Science-Materials/tree/main  <br>
https://github.com/rasbt/machine-learning-book/tree/main  <br>

https://github.com/bentrevett/pytorch-seq2seq/tree/master <br>
