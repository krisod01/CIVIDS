# CIVIDS
CIVIDS aims to provide a framework for collaborative intrusion detection system for the automotive domain
## Collaborative In-vehicle Intrusion Detection Systems

This repo consists of all files related to the masters thesis conducted by Daniel Aryan and Kristoffer Söderberg, during the Spring of 2021, at Chalmers University of Technology.

The thesis was conducted in the Software Engineering and Technology mastersprogram under the Computer Science and Engineering (CSE) Department.

The supervisor for this thesis was Rodi Jolak. 
The examiner of this thesis was Christian Berger.

## Structure of this Repository

 * `Implementation` contains all files related to the implementation of the CIVIDS framework.
 * `Implementation/Data` Contains the dataset files used during the implementation. 
 * `Implementation/Other` Contains both the resources used to train the ML models, as well as the scripts used 	to generate simulation runs 
 * `Implementation/Simulation` Contains the python files used to simulate a virtual CAN-network and to collect the experimental results 
 * `Implementation/Validation` contains the Results acquired from the simulation runs as well as the script-files used to validate the results and obtain F1-scores
 * `Report` contains both the LaTex project as well as the full masters thesis in PDF format.
 
## Utilized resources
* `Dataset` Hyunjae Kang; Byung Il Kwak; Young Hun Lee; Haneol Lee; Hwejae Lee;Huy Kang Kim. “Car Hacking: Attack & Defense Challenge 2020 Dataset”.In: (2021).doi:10.21227/qvr7- n418.url:https://dx.doi.org/10.21227/qvr7-n418
* `python-can`  [hardbyte](https://github.com/hardbyte/python-can) 
* `python-can-isotp`  [pylessard](https://github.com/pylessard/python-can-isotp) 
* `python-can-remote` [christiansandberg](https://github.com/christiansandberg/python-can-remote)

