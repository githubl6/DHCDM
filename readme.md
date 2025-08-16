# Exploring Comprehensive High-Order Relationships: Dual-Hypergraphs Cognitive Diagnosis for Intelligent Education Systems



## Project Structure

Here is the file structure for the project:
```
DHCD/
├── DOA.py
├── homogeneity.py
├── main.py
├── data/
│   ├── assist17/
│   │   ├── a17TotalData.csv
│   │   ├── config.json
│   │   ├── q.csv
│   │   └── h.csv
│   ├── Math1/
│   │   ├── Math1TotalData.csv
│   │   ├── config.json
│   │   ├── q.csv
│   │   └── h.csv     
│   ├── Math2/
│   │   ├── Math2TotalData.csv
│   │   ├── config.json
│   │   ├── q.csv
│   │   └── h.csv 
└── README.md
```
### data annotation
`data.csv` consists of response logs.<br>
`config.json` records all necessary settings of dataset like the number of students.<br>
`q.csv` contains the relevant between questions and knowledge attributes.<br>
`h.csv` indicates whether the student has attempted the exercise.
## Quick Start
We provide Math2 as sample datasets to validate the DHCD. You can reproduce the results by directly running `main.py`, i.e.

`python main.py`


## Required Libraries and Versions

The following libraries and versions are required for this project:

```bash
pytorch==1.13.0+cu0.11
scikit-learn==1.1.2
pandas==1.3.2
scipy==1.9.1
numpy==1.21.2