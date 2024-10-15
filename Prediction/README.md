# Sepsis Prediction
> This is the final project of the Comp90089 - Group 8. We use the MIMIC-III data to explore sepsis prediction.


### Prerequisite
- Set up python environment by 
```
pip install -r requirements.txt
```
- Scala 2.11

### Data Preprocess
1. Download MIMIC-III database in PostgreSQL on local 
2. Run the commands in the README.md in ./Sofa_Scoring to get pivoted vital data and infection_time (Johnson, Alistair EW, David J. Stone, Leo A. Celi, and Tom J. Pollard. "The MIMIC Code Repository: enabling reproducibility in critical care research." Journal of the American Medical Informatics Association (2017): ocx084    https://doi.org/10.5281/zenodo.821872)
3. Run the codes in the concept folder of ./Sofa_Scoring/mimic-iii repository to get pivoted SOFA score (https://doi.org/10.5281/zenodo.821872)
4. Run the codes in './Prediction/timeline' and generate SOFA timeline information
5. Run ```load_sepsis.py``` in './Prediction/src/data_preprocess' to retrieve ICU stays with sepsis and corresponding onset time
6. Run ```python data_preprocess.py``` in './src/data_preprocess' to get labeled pivoted vital data ready for model training

After data preprocess, the processed data are in the './data/sepsis/train', './data/sepsis/validation' and './data/sepsis/test'.


### Prediction models
1. Run ```convert_data.py``` in './src/transform' to construct the features sequence data for prediction models
2. Run ```start_modeling.py``` in './src/ML' to run the machine learning models
3. Run ```start_modelling.py``` in './src/LSTM' to run the deep learning models

