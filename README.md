# Microbial Biomarkers of BV Recurrence: Identifying Predictive Taxa
This project uses a patient dataset described below to (1) create a microbiome temporal network analysis for BV positive patients to track how microbial communities change over time in response to antibiotic and detect co-occurring microbial groups and (2) develop a model using random forests to predict bacterial vaginosis recurrences. The insights unveiled in the first aim would aid in deeper understanding of the microbial dynamics and interactions during antibiotic treatment over time to guide development of treatment regimens including novel antibiotics, personalized vaginal microbiome transplants, or species-specific prebiotic or probiotic agents. Meanwhile, the second aim would identify at-risk patients for early intervention avoiding the overuse of antibiotics and reducing the duration of discomforting symptoms associated with BV. Overall, prevention of BV could decrease HIV susceptibility and mitigate other related pathologies.

## Datasets
The datasets for the patient data and taxonomy for the OTU reads are stored in the datasets folder.

The patient data is an edited dataset composed of subject diagnostic information and vaginal microbiota of 132 HIV positive Tanzanian women (including 39 who received metronidazole treatment for BV) with added columns for BV status and Recurrence Type. A BV status of 1 indicates positive and 0 indicated negative for BV based on a nugent score of 7 or greater. The recurrence types are defined as 
- **No Initial Response**: BV positive for all follow-ups,
- **Immediate Recurrence**: BV positive after one negative follow-up,
- **Delayed Recurrence**: BV positive after two or more negative follow-ups,

No recurrence is labeled as
- **Successful Response**: BV negative for all follow-ups after first visit.

The taxonomic classifacation dataset is the original provided by Hummelen et al. (2010). 

## Project Files

In code folder:

- BVnetworks.py -- Python script to run in terminal to create the networks and extract network features.
- BVrfc.py -- Python script to run in terminal that creates the random forest classifier and evaluate performance.

In datasets folder: 

- diagnosis_data.csv -- Edited dataset from Hummelen et al. (2010) containing initial and follow-up tests (Amsel score, Nugent score, pH, OTUs, etc.) with added features BV status (1 = positive, 0 = negative) and Recurrence Type (Resistant, Immediate, Delayed, Successful). 
- classification.csv -- Original taxonomical classification for identified OTUs from Hummelen et al. (2010). 

In figures folder:

- BVnetworks_example_result.png -- Example graphs created from running BVnetworks.py
- BVrfc_example_result.png -- Example graphs created from running BVrfc.py

## Packages 
pandas version: 2.2.2
matplotlib version: 3.10.0

## Installation 
You can run the script from the terminal by following the steps below. 
1. Create and name folder in a directory of your choice (ie. Downloads > tutorial)
2. Download diagnosis_data.csv, classification.csv, BVnetworks.py and BVrfc.py and store in the folder created in step 1.
3. Open terminal and navigate into the folder created in step 1, which should have the files from step 2. (ie. cd Downloads/tutorial)
4. Enter in the command line: python3 BVnetworks.py or BVrfc.py
5. Compare to expected output file.
   

## Dataset Source
Hummelen, R., Fernandes, A. D., Macklaim, J. M., Dickson, R. J., Changalucha, J., Gloor, G. B., & Reid, G. (2010). Deep Sequencing of the Vaginal Microbiota of Women with HIV. PLOS ONE, 5(8), e12078. https://doi.org/10.1371/journal.pone.0012078 


