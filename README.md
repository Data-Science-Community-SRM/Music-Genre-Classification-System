<p align="center">
<a href="https://dscommunity.in">
	<img src="https://github.com/Data-Science-Community-SRM/template/blob/master/Header.png?raw=true" width=80%/>
</a>
	<h2 align="center"> Genrify - The Music App </h2>
	<h3 align="center"> Hear it. Genrify it. </h3>
<h4 align="center"> In this work, the objective is to classify the audio data into specific genres from GTZAN dataset which contain about 10 genres. We have build a Convolutional Neural Network model using the tensorflow library to classify the 10 genres.  <h4>
</p>

---
[![DOCS](https://img.shields.io/badge/Documentation-see%20docs-green?style=flat-square&logo=appveyor)](INSERT_LINK_FOR_DOCS_HERE) 
  [![UI ](https://img.shields.io/badge/User%20Interface-Link%20to%20UI-orange?style=flat-square&logo=appveyor)](INSERT_UI_LINK_HERE)

## Introduction:
-  The idea behind this project is to see how to handle sound files in python, compute sound and audio features from them, run Deep learning algorithms on them, and predict the genre using an audio signal as its input. So we considered 2 datasets, one is the FMA dataset and the other is the GTZAN dataset.

<b>The FMA Dataset</b>
	<p> The FMA dataset is based on the music contributed by various, mostly indie artists to the Free Music Archive. The smallest variant of this dataset (‘fma-small’) which was about 9 GiB uncompressed and with about 8K tracks. The FMA dataset is robust as, it is actually representative of contemporary music, at least in terms of the recording quality (44.1 kHz stereo) and is generally very high quality, originally meant for end user consumption. Hence, it was chosen to be ideal for training a model for music genre classification. </p>

<b>The GTZAN Dataset </b>
	<p> GTZAN dataset was used that contains 1000 music clips of time duration 30 second with 22050 Hz sampling frequency. There are in all 10 different genres like blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae and rock. Each genre has 100 audio files. An audio read with the sampling rate of 22050 Hz. After that, it split the audio of 30 seconds durations into 3 seconds durations of audio clips. The 10 genres are as follows:
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock
</p>

## Data Preprocessing steps: 
- [ ]  Using Librosa library and displaying the raw audio files.
- [ ]  Plotting the Spectrograms and Mel-Spectrograms for better understanding of the audio files.
- [ ]  Splitting the data into training and testing sets.
- [ ]  Feature extraction and scaling of the features for easier model construction

<br>
	
## MFCC's are derived as follows: 
- [ ]  Fourier transform of a signal is taken.
- [ ]  The powers of the spectrum obtained above are mapped onto the mel scale, using triangular overlapping windows.
- [ ]  The logs of the powers at each of the mel frequencies are taken.
- [ ]  The discrete cosine transform of the list of mel log powers are taken, as if it were a signal.
- [ ]  The MFCCs are the amplitudes of the resulting spectrum.

## Model Construction:

### GTZAN Model:
#### Deep Learning Model : 
<p> After pre-processing the dataset, we come to the part where we use concepts of Convolutional Neural Network to build and train a model that classifies the music genre. Using the Convolutional Neural Network Model which made use of features such as MFCC's,spectral centroids, extracted features are in features3sec.csv. </p>

#### For the CNN model:
	
- All of the hidden layers are using the RELU activation function and the output layer uses the softmax function. The loss is calculated using the sparse_categorical_crossentropy function.Dropout is used to prevent overfitting.
- The model is compiled using the optimizer and the sparse_categorical_crossentropy loss function will be optimized which is suitable for multi-class classification.We are monitoring the classification accuracy metric since we have the same number of examples in each of the 10 classes.
- The model accuracy can be increased by further increasing the epochs but after a certain period, we may achieve a threshold, so the value should be determined accordingly.
- The final trained model resulted in an accuracy around 92% on the dataset with 6693 .wav files.

## Project architecture:
The basic project architechture is given below:
<p align="center">
	
<img src ="https://github.com/Data-Science-Community-SRM/Music-Genre-Classification-System/blob/master/Images/model.jpg" style="height:600px" align="centre">
</p>	
<br>

## Instructions to run

* Pre-requisites:
	<br><br>
  Installing Pre-requsites using environment.yml	
	
	```bash
	conda env create -f environment.yml 
	```
	```bash
      conda activate {Environment_Name} 
	```
  Installing Pre-requistes using requirements.txt
	```bash
	pip install -r /path/to/requirements.txt
	```

* Directions to run Flask Server 
	```bash
 	python app.py 
	```


## Contributors

<table>
<tr align="center">


<td>

Kruthi M

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Music-Genre-Classification-System/blob/master/Images/k.jpeg"  height="120" alt="Kruthi M">
</p>
<p align="center">
<a href = "https://github.com/Kruthim1304"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/kruthi-m-49b300200/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td>

Shashwat Ganesh

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Music-Genre-Classification-System/blob/master/Images/s.jpeg"  height="120" alt="Shashwat Ganesh">
</p>
<p align="center">
<a href = "https://github.com/kknives"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/sga0xc33d/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

	
	
	
<td>

Anushree Bajaj

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Music-Genre-Classification-System/blob/master/Images/a.jpg"  height="120" alt="Anushree Bajaj">
</p>
<p align="center">
<a href = "https://github.com/condescendo"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/anushree-bajaj-7486b71b9/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
	


<td>

Jahnavi Darbhamulla

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Hand-Written-Digit-Classification-Recognition/blob/master/Doc%20images/j.jpg" height="120" alt="Jahnavi Darbhamulla">
</p>
<p align="center">
<a href = "https://github.com/JahnaviDarbhamulla"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/jahnavi-darbhamulla-0a4280201/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
	
	
	
<td>

Rayaan Faiz

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Music-Genre-Classification-System/blob/master/Images/r.jpg"  height="120" alt="Rayaan Faiz">
</p>
<p align="center">
<a href = "https://github.com/RayaanFaiz14"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/rayaan-faiz-1746261b6/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

</tr>
  </table>
  
## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

<p align="center">
	Made with :heart: by <a href="https://dscommunity.in">DS Community SRM</a>
</p>

