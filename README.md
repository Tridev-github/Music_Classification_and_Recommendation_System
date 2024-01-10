# Music_Classification_and_Recommendation_System
ABSTRACT
In recent years, online music streaming services have rapidly increased. Searching for new songs
which fall under user‘s similar tastes has become a challenging issue. Music classification and
recommendation systems have gained significant attention in recent years due to the abundance
of digital music content and the growing demand for personalized music recommendations. In
this project, we develop a music genre classification model using a convolutional neural network
which takes user input songs and outputs its genre. The proposed music classification and
recommendation system aims to provide accurate genre classification and personalized music
recommendations to enhance the music listening experience for users. The system's effectiveness
and accuracy are evaluated through extensive experiments and evaluations using benchmark
music datasets, and the results demonstrate the system's potential for providing accurate genre
classification and personalized music recommendations. We also developed a personalized
music recommendation system using Euclidean Distance, Cosine Similarity and Correlation
Distance.
6
4-CHAPTERS
4.1 Introduction
Music has become an integral part of modern society, and with the proliferation of digital music
content, the need for effective music classification and recommendation systems has grown
exponentially. Music classification involves assigning genres or labels to music based on its
audio features, while music recommendation aims to provide personalized music suggestions to
users based on their preferences, listening history, and demographic information. Music
classification and recommendation systems have numerous applications, including music
streaming platforms, online music marketplaces, and personalized music recommendation
services, and can significantly enhance the music listening experience for users.
Music classification involves the process of assigning genres or labels to music based on its
audio features, such as rhythm, melody, timbre, and instrumentation. Traditional music
classification methods rely on manual genre labeling, which can be subjective, inconsistent, and
time-consuming. However, with the advancements in machine learning, digital signal processing,
and data analytics, automated music classification techniques have become increasingly popular,
allowing for more accurate and efficient genre classification. Traditional music classification
methods rely on manual genre labeling, which can be subjective and inconsistent. However, with
the advancements in machine learning and data analytics, automated music classification
techniques have emerged, leveraging audio features extracted from music signals to classify
music into different genres. Additionally, collaborative filtering techniques, which utilize user
listening history and preferences, have been widely used in music recommendation systems to
generate personalized music suggestions. On the other hand, music recommendation aims to
provide users with personalized music suggestions based on their preferences, listening history,
and demographic information. Collaborative filtering techniques, which analyze user behavior
data, such as listening history and ratings, have been widely used in music recommendation
systems to generate personalized music suggestions. By leveraging user preferences and
behaviors, music recommendation systems can provide users with relevant and enjoyable music
recommendations, enhancing their music listening experience and promoting music discovery.
Online music streaming services such as Spotify, iTunes, Saavn, etc. all provide users with
millions of songs from a wide variety of artists, genres, and decades to choose from, along with
complex recommendation algorithms to suggest new music tailored to fit the tastes of any user.
For example, Spotify provides users a web API to extract audio features and metadata like
popularity, tempo, loudness, keys and the year of release to create music recommendation
systems based on both collaborative and content-based filtering. In this project we build two
models:
7
1) Music Genre Classification System
2) Music Recommendation System
Music Genre Classification System
Genre classification is the base of any strong music recommendation system which helps
recommend songs based on the core style or category of songs, namely its genre. We develop a
model which can take an input song from the user, extract its features and output the genre under
which it falls under with a high degree of accuracy. Music Recommendation System Searching
for songs we like manually is a difficult and time-consuming process. This also requires having
access to a huge dataset of songs and finding songs which match a given taste. 8 The goal of any
good music recommendation system is to provide users with a selection of songs matching their
tastes (based on their previously heard history of songs). Recommendation systems automate
finding songs by analyzing trends, popular genres, and artists.
4.2 LITERATURE REVIEW
Literature Survey 1:
A survey of music recommendations systems and future perspectives
Author: Marcus Pearce, Simon Dixon, Yading Song
Source: ResearchGate
Content: This paper surveys a general framework and state-of-art approaches in recommending
music. Two popular algorithms: collaborative filtering (CF) and content-based model (CBM),
have been found to perform well. Three essential components of a music recommender system—
user modeling, item profiling, and match algorithms—are covered in this article. Four possible
concerns with the user experience and six suggestion models are outlined.
Literature Survey 2:
Research on Music Content Recognition and Recommendation Technology
Based on Deep Learning
Author: Gao Yang, Muhammad Arif
8
Source: ResearchGate
Content: This research aims to create a better music algorithm that incorporates user data for
deep learning, a candidate matrix compression technique for suggestion improvement, accuracy,
recall rate, and other metrics as evaluation criteria. With the use of user data for deep learning, a
potential matrix compression approach for bettering recommendation quality, accuracy, recall
rate, and other metrics as evaluation criteria, this research intends to develop music algorithms.
Literature Survey 3:
A Preliminary Study on a Recommender System for the Million Songs Dataset
Challenge
Author: Fabio Aiolli
Source: ResearchGate
Content: In this paper, the preliminary study was conducted on the Million Songs Dataset
(MSD) challenge. The task of the competition was to suggest a set of songs to a user given half
of its listening history and complete listening history of other 1 million people. The early
research we did on the Million Songs Dataset (MSD) challenge is discussed in this publication.
The goal of the competition was to choose a group of music to recommend to a user based on
their listening history and the listening histories of 1 million other users. The paper focused more
on 9 defining similarity functions; effect of ―locality‖ of collaborative scoring function;
aggregating multiple ranking strategies.
Literature Survey 4:
Hybrid Music Recommendation using K-Means Clustering
Author: Gurpreet Singh, Vishal Kashyap, Jaskirat Singh, Bishal,Pratham Verma
Source: ResearchGate
Content: The demand for effective music recommendation systems is increasing as the digital
music market grows. Collaborative filtering is a well-liked technique that has been around for a
long and is often employed. Collaborative filtering is a technique, although it is quite slow. KMeans clustering is applied on the user-features modeled using the user's listening history and
MFCC features of the songs to address this. This paper uses content-based filtering with K-mean
clustering algorithm for music recommendation system which provides effective and relevant
content to be suggested.
9
Literature Survey 5:
A music recommendation system based on music data grouping and user
interests
Author: Hungchen Chen, Arbee L.P. Chen
Source: Semantic Scholar
Content: With the expansion of the World Wide Web, a lot of music data is now accessible
online. It becomes required to provide a suggestion service in addition to looking for expected
music items for customers. In this study, we develop the Music Recommendation System
(MRS), which offers a customized music recommendation service. On the basis of the users'
favorite music groups, content-based, collaborative, and statistics-based recommendation
approaches are offered.
4.3 Overview of the Proposed System
1. Data Collection: The system will collect a large music dataset, including audio features
and metadata, from various sources such as online music platforms, music databases, or
APIs. The metadata may include information such as artist, album, release year, and user
listening history.
2. Feature Extraction: Audio features, such as rhythm, melody, timbre, and instrumentation,
will be extracted from the music signals using digital signal processing techniques. These
features will be used as input to the classification and recommendation algorithms.
3. Music Classification: Convolutional Neural Networks (CNN) will be used for music
genre classification. The extracted audio features will be fed into the CNN, which will
learn to automatically classify music into different genres based on the patterns it
captures from the audio features. The CNN will be trained on a labeled dataset of music
genres to optimize its classification accuracy.
4. Music Recommendation: For music recommendation, cosine similarity and Euclidean
similarity will be used. Collaborative filtering techniques will be employed to analyze
user listening history, preferences, and demographic information to create user profiles.
Similarity measures such as cosine similarity and Euclidean similarity will be used to
find the similarity between the audio features of the user's liked music and other music in
the dataset. Music that is most similar to the user's liked music will be recommended.
10
5. Real-time User Feedback: The system will incorporate real-time user feedback, such as
user ratings, likes, and skips, to continuously improve its recommendation accuracy. This
feedback will be used to update the user profiles and adjust the recommendation
algorithms accordingly.
6. Adaptation and Personalization: The system will continuously adapt and personalize its
recommendations based on user interactions and feedback. As the user's preferences and
listening history evolve, the system will update the user profiles and refine its
recommendations to better match the user's preferences over time.
7. User Interface: The system will have a user-friendly interface, such as a web or mobile
application, that allows users to interact with the system, browse and search for music,
view recommended music, provide feedback, and customize their music preferences.
8. Evaluation and Optimization: The system will be evaluated using various performance
metrics, such as classification accuracy, recommendation accuracy, and user satisfaction.
The system will be optimized based on the evaluation results to enhance its performance
and provide the best possible music classification and recommendation experience for
users.
ARCHITECTURE DIAGRAM:
11
ER DIAGRAM/ UML DIAGRAM:
12
Dataset
GTZAN Dataset - The dataset consists of 1000 audio tracks each 30 seconds long.
It contains 10 genres, each represented by 100 tracks. The tracks are all 22,050 Hz
Mono 16-bit audio files in .wav format.
4.4 Proposed System Analysis and Design
Recommendation systems generally use collaborative-based filtering or content-based filtering to
classify data tuples to given labels or make novel labels from scratch with no prior knowledge
given. For ‗Genre classification‘ we are using convolutional neural network (CNN or convent).
Convolutional Neural Networks
A convolutional neural network (CNN or convnet) is a machine learning subset. It is one of
several types of artificial neural networks used for various applications and data types. A CNN is
a type of network architecture for deep learning algorithms that is specifically used for image
recognition and pixel data processing tasks. There are other types of neural networks in deep
learning, but CNNs are the network architecture of choice for identifying and recognizing
objects. As a result, they are ideal for computer vision (CV) tasks and applications requiring
object recognition, such as self-driving cars and facial recognition. Deep learning algorithms rely
heavily on artificial neural networks (ANNs). A recurrent neural network (RNN) is one type of
ANN that takes sequential or time series data as input. It is appropriate for natural language
processing (NLP), language translation, speech recognition, and image captioning applications.
Another type of neural network that can uncover important information in both time series and
image data is CNN.
13
A deep learning CNN is composed of three layers: convolutional, pooling, and fully connected
(FC). The first layer is the convolutional layer, and the last layer is the FC layer.
Convolutional layer: The majority of computations take place in the convolutional layer, which
is the foundation of a CNN. A second convolutional layer can be added after the first.
Convolution involves a kernel or filter within this layer moving across the image's receptive
fields, checking for the presence of a feature.
Pooling layer: The pooling layer, like the convolutional layer, sweeps a kernel or filter across
the input image. However, in contrast to the convolutional layer, the pooling layer reduces the
number of parameters in the input while also causing some information loss. On the plus side,
this layer reduces complexity and improves CNN's efficiency.
Fully connected layer: The FC layer in the CNN is where image classification occurs based on
the features extracted in the previous layers. Fully connected in this context means that all of the
inputs or nodes from one layer are linked to every activation unit or node in the next layer.
As it would result in an overly dense network, all of the layers in the CNN are not fully
connected. It would also increase losses, reduce output quality, and be computationally
expensive. We used python to create various graphs such as, Raw_Audio_Waveplot,
Audio_Spectrogram, Audio_Spectral_Rolloff_Plot, Zero_Crossing_Rate Plot, etc , to classify the
genre of our music input file.
● Audio_Spectrogram - A spectrogram is a visual representation of the spectrum of
frequencies of a signal as it varies with time. Spectrograms are used extensively in the
fields of music, linguistics, sonar, radar, speech processing , seismology, and others. We
are using them to compare the genres wavelength, tempo
● Zero-Crossing Rate: The zero-crossing rate (ZCR) is the rate at which a signal transitions
from positive to zero to negative or negative to zero to positive.
For ‗Music Recommendation‘ we are using Cosine similarity, Euclidean Distance and
Correlation distance.
Cosine Similarity - Cosine similarity is the cosine of the angle between two ndimensional
vectors in an n-dimensional feature space. It is the dot product of the two vectors divided by the
product of the magnitude of two vectors.
To understand it better, let us take an example of two items, item 1 and item 2, and two features
‗x_1‘ and ‗x_2‘ , which define an item. The plot below represents item 1 and item 2 as vectors in
a feature space. The lesser the angle between vectors, the more the cosine similarity.
14
Euclidean Distance – In mathematics, the Euclidean distance between two points in Euclidean
space is the length of a line segment between the two points. It can be calculated from the
Cartesian coordinates of the points using the Pythagorean theorem
Correlation distance – In statistics and in probability theory, distance correlation or distance
covariance is a measure of dependence between two paired random vectors of arbitrary, not
necessarily equal, dimension. The population distance correlation coefficient is zero if and only
if the random vectors are independent. Thus, distance correlation measures both linear and
nonlinear association between two random variables or random vectors. Distance correlation can
be used to perform a statistical test of dependence with a permutation test. One first computes the
distance correlation (involving the recentering of Euclidean distance matrices) between two
random vectors, and then compares this value to the distance correlations of many shuffles of the
data. The distance correlation of two random variables is obtained by dividing their distance
covariance by the product of their distance standard deviations. The distance correlation is the
square root of the formula given below.
Deep learning, specifically Convolutional Neural Networks (CNNs), plays a crucial role in music
classification and recommendation systems. CNNs are a type of neural network architecture that
excel at processing data with grid-like structures, such as images and audio signals, making them
well-suited for music analysis tasks. Here are some key aspects of deep learning and CNNs
related to the topic of music classification and recommendation:
15
1. Feature Learning: CNNs are capable of automatically learning relevant features from raw
audio signals, such as rhythm, melody, timbre, and instrumentation, without relying on
handcrafted features. This allows the model to capture intricate patterns and
representations from the music data, leading to improved classification accuracy.
2. Hierarchical Feature Extraction: CNNs utilize multiple layers of convolutional and
pooling operations to hierarchically extract features of increasing complexity. The lower
layers capture low-level features, such as edges and textures, while the higher layers
capture high-level features, such as semantic information. This hierarchical feature
extraction enables the model to learn meaningful representations of music data at
different levels of abstraction, leading to better classification performance.
3. Parameter Sharing and Local Receptive Fields: CNNs use parameter sharing, where the
same set of weights is used for multiple locations in the input, and local receptive fields,
which limit the model's focus to a small region of the input at a time. This makes CNNs
computationally efficient, as they reduce the number of parameters and operations
compared to fully connected networks, making them well-suited for processing large
audio datasets.
4. Transfer Learning: Transfer learning is a technique where a pre-trained CNN, trained on
a large dataset, is used as a starting point for training a new model on a smaller dataset.
This allows leveraging the learned features from the pre-trained model, which may have
learned general audio features from a large dataset, to improve the performance of the
music classification model with limited data. Transfer learning can significantly speed up
the training process and improve the classification accuracy.
5. Interpretability: CNNs, being complex black-box models, may lack interpretability in
terms of understanding the reasoning behind their predictions. However, there are
techniques such as visualization of learned features and attention mechanisms that can
provide insights into how the model is processing the music data and making decisions,
which can aid in understanding the model's behavior and building trust with users.
6. Model Optimization: Deep learning models, including CNNs, require careful tuning of
hyperparameters, model architectures, and regularization techniques to optimize their
performance. Techniques such as dropout, batch normalization, and learning rate
scheduling can be employed to enhance the model's generalization, robustness, and
convergence properties.
Overall, deep learning and CNNs are powerful tools for music classification and
recommendation tasks, enabling the system to automatically learn relevant features,
hierarchically process music data, adapt to user feedback, and provide personalized
recommendations. However, careful considerations must be made in terms of data quality, model
optimization, interpretability, and user privacy to ensure the effectiveness, reliability, and user
acceptance of the system.
16
4.5 Implementation
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
4.6 - Conclusions and future work
The use of deep learning and Convolutional Neural Networks (CNNs) in music classification and
recommendation systems holds great promise. CNNs excel at automatically learning relevant
features from raw audio signals, hierarchically extracting meaningful representations, and
adapting to real-time user feedback. They enable the system to provide accurate and personalized
music recommendations, enhancing the user experience. In the end we successfully predict the
genre of our music and also suggest recommendations based around our input audio file. The
results obtained were effectively visualized with the help of graphs.
Some potential areas for future work:
1. Incorporating Attention Mechanisms: Attention mechanisms can enhance the
interpretability of the CNN model by allowing it to focus on different parts of the input
audio signal with varying levels of importance. This can improve the model's ability to
capture relevant features and improve its classification and recommendation
performance.
2. Exploring Different CNN Architectures: CNNs come in various architectures, such as
residual networks (ResNets), densely connected networks (DenseNets), and recurrent
convolutional networks (RCNs). Exploring different architectures and their combinations
may lead to better performance in music classification and recommendation tasks.
3. Multi-modal Fusion: Music data often includes multiple modalities, such as audio, lyrics,
metadata, and user behavior. Incorporating multi-modal fusion techniques, such as audiolyrics fusion, audio-metadata fusion, or audio-user behavior fusion, can enhance the
system's ability to capture diverse information and provide more accurate and
personalized recommendations.
4. Incorporating User Context: Capturing user context, such as the user's listening history,
preferences, and contextual information (time, location, etc.), can improve the system's
ability to provide relevant recommendations. Utilizing contextual information can
enhance the system's understanding of user preferences and help in making more
personalized and context-aware recommendations.
5. Evaluating Robustness and Generalization: Robustness and generalization of the model
are critical factors for real-world deployment. Evaluating the model's performance under
different conditions, such as different music genres, audio quality, and user behaviors,
can provide insights into its reliability and generalization capabilities. Robustness testing,
adversarial training, and domain adaptation techniques can be explored to enhance the
model's performance in real-world scenarios.
6. User Interaction and Feedback: Incorporating more user interaction and feedback
mechanisms, such as explicit user feedback (ratings, reviews), implicit user feedback
(user behavior, play counts), and real-time user feedback, can enhance the system's
ability to adapt and improve recommendations over time. Utilizing user feedback for
model updates and retraining can lead to more accurate and personalized
recommendations.
38
7. Privacy and Ethical Considerations: Ensuring user privacy and addressing ethical
considerations, such as data privacy, bias, fairness, and transparency, are important in the
development and deployment of music classification and recommendation systems.
Employing techniques such as federated learning, privacy-preserving machine learning,
and fairness-aware machine learning can address these concerns and ensure responsible
and ethical use of the system.
39
5- REFERENCES
[1] Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional Recurrent Neural
Networks for Music Classification: Philosophy and Experiments. Proceedings of the 18th
International Society for Music Information Retrieval Conference (ISMIR), 89-96.
[2] Van den Oord, A., Dieleman, S., & Schrauwen, B. (2013). Deep Content-Based Music
Recommendation: Using Convolutional Neural Networks to Predict Song Similarity.
Proceedings of the 26th International Conference on Neural Information Processing Systems
(NIPS), 2643-2651.
[3] Dieleman, S., & Schrauwen, B. (2014). End-to-end Learning for Music Audio. IEEE
Transactions on Audio, Speech, and Language Processing, 22(12), 1859-1869.
[4] Lee, K., Lee, K. J., & Lee, S. (2018). Music Genre Classification Using Convolutional
Neural Networks. Proceedings of the 19th International Society for Music Information
Retrieval Conference (ISMIR), 443-449.
[5] Wang, Z., & Zhang, C. (2019). Deep Learning for Music Classification: A
Comprehensive Review. ACM Computing Surveys, 52(4), 1-36.
[6] Wang, S., Wang, Y., & Yang, Y. (2019). Music Recommendation Using Convolutional
Neural Networks for Audio Representations Learning. Proceedings of the 20th International
Society for Music Information Retrieval Conference (ISMIR), 625-632.
[7] Sigtia, S., Benetos, E., Boulanger-Lewandowski, N., & Dixon, S. (2014). An End-to-End
Neural Network for Polyphonic Music Transcription. IEEE Transactions on Audio, Speech,
and Language Processing, 22(12), 2146-2158.
[8] Pons, J., Nieto, O., Prockup, M., & Schmidt, E. M. (2017). End-to-End Learning for
Music Audio Tagging at Scale. Proceedings of the 18th International Society for Music
Information Retrieval Conference (ISMIR), 546-553.
[9] Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Transfer Learning for Music
Classification and Regression Tasks. IEEE Transactions on Audio, Speech, and Language
Processing, 26(11), 2041-2053.
[10] Liang, Y., Yang, D., Yang, Y., & Zhou, L. (2020). MUFASA: Multimodal Fusion
Architecture with Semantic Alignment for Music Recommendation. Proceedings of the 21st
International Society for Music Information Retrieval Conference (ISMIR), 442-449.
[11] Dittmar, C., & Müller, M. (2018). Deep Audio Style Embeddings: Learning Latent
Representations of Music for Style-Based Music Recommendation. Proceedings of the 19th
International Society for Music Information Retrieval Conference (ISMIR), 620-626.
40
[12] Kim, K., & Lee, J. H. (2019). Music Recommendation Based on Deep Learning of
Variable-Length Sequences. Proceedings of the 20th International Society for Music
Information Retrieval Conference (ISMIR), 134-141
