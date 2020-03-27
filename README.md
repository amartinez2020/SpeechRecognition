


  Long Short-Term Memory Networks for
Classification of English Vowel Phonemes


Alejandro Roman Martinez
Dartmouth College
Hanover, New Hampshire
alejandro.r.martinez.20@darmouth.edu
https://github.com/amartinez2020/SpeechRecognition




Abstract
Long short-term memory networks (LSTMs) have taken over the field of speech recognition through the introduction of constant error carrousels. These carrousels allow the network to learn long-term relationships while also mitigating the risks of vanishing gradients. When applied to audio signals, LSTMs are able to learn valuable dependencies between frequency coefficients and forget invaluable information, making them ideal networks for speech recognition. In this study, I chose to apply these networks to the challenge of English vowel phoneme classification. When tested on the University of Western Michigan Vowel Database, the network achieved an accuracy of 91.37% on the validation set. With the promising results obtained, I anticipate that this network can be applied to vowel phoneme classification for other less prominent languages in efforts of recovery.
1	Introduction
Phoneme classification is a field of Computational linguistics research that focuses on the capacity to accurately classify phonemes. This field is an important aspect in understanding a given language’s phonetical structure. Much of previous work on phoneme classification is done through the use of Support Vector Machines (SVMs) (Amami et al, 2012; Li et al, 2005). While support vector machines provide simplicity to a network, in an effort to avoid overfitting and achieve worthy generalization, they are not optimal when dealing with sequential audio data. In this study I present the use of Long short-term memory networks (LSTMs) on such data, due to their exceptional ability to learn valuable long-term dependencies and forget invaluable information (Hochreiter & Schmidhuber, 1997; Graves et al 2005). 
I specifically focus on English vowel phonemes in this study as a preliminary experimentation. As the chosen language in this study seems somewhat mundane (in the age of widely available English speech recognition programs) the true purpose of this study is to provide the grounds for these networks to generalize to less prominent languages that face the risk of extinction. As we can provide these languages with computational linguistic resources, we can hopefully prolong their life in this era of information technology. Therefore, the results reflect future performance on these more imperative use cases. 
	This paper is structured as follows: In section 2 the methodology of the experiment is described including the dataset, feature extraction, the architecture of the network,  and its training strategy; the results of the network’s performance are presented in section 3;  in section 4 I discuss the results; and in section 5 I conclude by considering future anticipations and applications for this study. 

2	Methodology
Data: The dataset I utilized was obtained from the University of Western Michigan Vowel Database:
https://homepages.wmich.edu/~hillenbr/voweldata.html

The dataset was available online and comprises of audio recordings of every English vowel phoneme from 150 men, women, and children (see figure 1 for English vowel phonemes).
 

Fig. 1: English Vowel Phonemes


Feature Extraction: After obtaining the data, I transformed each audio file into Mel Frequency Cepstral Coefficients (MFCC). I chose to utilize MFCCs because of their inherent ability to accurately model human auditory perception. As humans have trouble distinguishing between high frequency sounds, MFCCs model this quality by performing a log transformation on frequency data (see figure 2 for MFCC spectrogram). The MFCC feature set contains a total of 12 MFCC parameters. The sampling rate was set to the audio files default of 16kHz. 
Because the network I utilized was supervised I needed to provide training labels for the inputs. These labels were formatted as one hot encoded vectors that corresponded to the appropriate vowel phonemes. 


  

Fig. 2: Mel Frequency Cepstral Coefficients on an Audio Signal

Network Architecture: As mentioned above, I chose to utilize an LSTM network to process each audio file (see figure 3 for LSTM architecture). The model was adapted from Keras, an open source machine learning library. As the goal was to classify an audio signal into one of 12 English vowel phonemes, I chose to implement an LSTM layer, which took as input a sequence of MFCCs, followed by a fully connected dense layer that output a 12-dimensional vector. The LSTM layer contained 100 memory units while the dense layer utilized a sigmoid activation function.
The compilation of the model utilized a binary cross entropy loss function and a stochastic gradient optimizer because of the nature of the one hot encoded training vector.  

 

Fig. 3: Long short-term memory network architecture

Training: I trained the network with a batch size of 32 for a total of 15 epochs. Because of the small size of the dataset, the network did not need many training iterations in order to reach convergence. 

3	Results
Utilizing Keras’ model metric of ‘accuracy’, I obtained a loss of 0.3365 and an accuracy of 91.39% on the training set and an accuracy of 91.27% on the validation set. 
Although there is room to improve, the high degree of accuracy in the validation set indicates that the model was able to avoid overfitting. 

4	Discussion
As discussed in the introduction, the purpose of this study is to provide optimality to phoneme classification with LSTMs. In comparison with a study done by Amami et al, 2012, who utilized SVMs for Phoneme Recognition, the  LSTM model achieved a superior accuracy (91.37% vs. 52%). These results confirm the notion that LSTMs are optimal for classification on sequential audio data due to their ability to remember valuable long-term dependencies and forget invaluable information. 

5	Conclusions
The future for this study is twofold: increase the dataset and apply the network to less prominent languages at risk for extinction. As referenced in the methodology section, the dataset did not need many training iterations to reach convergence. This is both beneficial and limiting. It is beneficial because it can achieve similar accuracy on languages that are at risk and cannot afford to produce large datasets due to a shortage of speakers. On the other hand, it is limiting because the early stopping prevents the network from obtaining a better accuracy rating that could be achieved by training on a larger dataset. This step for improvement must be dormant until we can find large datasets for at risk languages. 

Acknowledgments
I would like to thank Rolando Coto-Solano for providing me with helpful guidance on network architecture as well as general mentorship.

References 
Amami, Rimah, et al. “Phoneme Recognition Using Support Vector Machine and Different Features Representations.” Advances in Intelligent and Soft Computing Distributed Computing and Artificial Intelligence, 2012, pp. 587–595., doi:10.1007/978-3-642-28765-7_71.
Graves, Alex, and Jürgen Schmidhuber. “Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures.” Neural Networks, vol. 18, no. 5-6, 2005, pp. 602–610., doi:10.1016/j.neunet.2005.06.042.
Li, Fuhai, et al. “MFCC and SVM Based Recognition of Chinese Vowels.” Computational Intelligence and Security Lecture Notes in Computer Science, 2005, pp. 812–819., doi:10.1007/11596981_118.
Maklin, Cory. “LSTM Recurrent Neural Network Keras Example.” Medium, Towards Data Science, 21 July 2019, towardsdatascience.com/machine-learning-recurrent-neural-networks-and-long-short-term-memory-lstm-python-keras-example-86001ceaaebc.
Yoon, Seunghyun, et al. “Multimodal Speech Emotion Recognition Using Audio and Text.” 2018 IEEE Spoken Language Technology Workshop (SLT), 2018, doi:10.1109/slt.2018.8639583.
Zhang, Yuanyuan, et al. “Attention Based Fully Convolutional Network for Speech Emotion Recognition.” 2018 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2018, doi:10.23919/apsipa.2018.8659587.
Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.
Homepages at WMU, homepages.wmich.edu/~hillenbr/voweldata.html.



























