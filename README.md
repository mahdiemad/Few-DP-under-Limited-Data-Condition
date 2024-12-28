# Differential Fault Diagnosis in Power Transformers under Limited Data Condition via Few-Shot Learning 
the code repository for the paper "Differential Fault Diagnosis in Power Transformers under Limited Data Condition via Few-Shot Learning "

[Paper Link in archive](https://ieeexplore.ieee.org/document/10192453)

## citation ( to be updated)
```
@ARTICLE{10192453,
  author={Emadaleslami, Mahdi; Moradzadeh,Arash; and Azzouz, Maher},
  journal={IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS}, 
  title={Differential Fault Diagnosis in Power Transformers under Limited Data Condition via Few-Shot Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={-}

```

## Env requirement
Python 3.8.8 and the following versioned packages were used for All codes:
- Keras >= 2.10.0
- matplotlib >= 3.3.4
- seaborn >= 0.11.1
  
open main _Code.ipynb by **Jupyter Notebook**  for better experience.
## Structure

- **PSCAD**:  PSCAD Simulations for generating Fault data according to Table II of the paper.
- **Complete_Data.xlsx**:  all raw current signals from the PSCAD simulation as one file for further use with the pandas library
- **Cycle_Data.xlsx**:  Extracted cyclic data from the Complete_Data.xlsx as one file for further use with the pandas library
- **main _Code.ipynb**: the main file for usage
- **models.py**: Define the Siamese Network model and other functions.
- **siamese_1D.py**: Define the init of Siamese Network input data, training, and testing functions.
- **models_contrastive.py**: The Siamese Network model with contrastive loss function.
- **siamese_1D.py**: Define the init of Siamese Network model with contrastive loss function input data, training, and testing functions.
- **Pictures**:  code for Confusion Matrix & T-SNE

## Intrudoction

Power transformers play a critical role in the efficient distri-bution of electrical energy, ensuring reliable power supply across various sectors. These critical equipments are con-tinuously exposed to thermal and electrical faults, which could lead to sudden and catastrophic failure, resulting in widespread outages and significant financial losses. To avoid severe consequences of transformer failures, constant monitoring and protection are paramount. a reliable differential protection method must address several challenges.First, power transformers cannot operate in faulty states for extended periods due to the risk of catastrophic failure. Moreover, many internal faults develop gradually, making it difficult to gather sufficient data during the degradation process. On the other hand, accelerated aging tests are costly, leading to data imbalance where certain fault types, such as turn-to-turn (T2T) or Inter-Winding (IW) faults, are underrepresented. Second, the scarcity of certain faults, which may take years to detect, complicates data collection and requires methods that perform effectively with limited data, ensur-ing robust fault detectionin real-world scenarios.
Third, a major challenge is developing fault detection models that are not limited to specific transformer types or operational conditions. These models must handle new, unseen faults and adapt across various transformer types without cumbersome retraining, enabling widespread adop-tion in the power grid. Fourth, noise from substations, often reducing the Signal-to-Noise Ratio (SNR) to around 27 dB, further complicates fault detection by obscuring fault sig-nals. Fifth, differential protection must balance compu-tational time and accuracy to provide timely responses and prevent damage. Finally, fault detection methods must be low in complexity and cost, considering hardware compat-ibility, computational resources, and ease of integration into existing systems. In this study, a novel AI-based model termed as few-shot learning is proposed to diagnose various internal and external faults in real-world power transformers. The fulcrums of the proposed Few-DP, as depicted in Fig.2, are the following layers: data preparation, feature embedding, distance unit, and application.

![Fig.2.](https://github.com/mahdiemad/Static-Eccentricity-Fault-Location-Diagnosis-in-Resolvers-with-Few-Shot-Learning/assets/57590076/a7c28eb8-36d0-461b-9f87-a5030474c491)
#### Fig. 2. Overview of the proposed Few-DP.      

Also, The pseudo-code is as follows:

![psedo](https://github.com/mahdiemad/Static-Eccentricity-Fault-Location-Diagnosis-in-Resolvers-with-Few-Shot-Learning/assets/57590076/f5c4c220-1236-4935-a6bc-58d974a62c4c)
 

All our models and PSCAD simulations in this study are open source. PSCAD simulations are used to generate fault data and gathered in the Complete_Data.xlsx; It is suggested to use Complete_Data.xlsx for python codes.