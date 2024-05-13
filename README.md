# 645-mini-project
Mini project for the class COMPSCI 645 at UMass Amherst

Implements the paper: Explore-by-Example: An Automatic Query Steering Framework for Interactive Data Exploration
Which can be found here: https://dl.acm.org/doi/pdf/10.1145/2588555.2610523

## Installation
### Dataset setup
1. Create folder 'data' in the main repository
2. Download the dataset from https://reverie.cs.umass.edu/courses/645/s2024/projects.html
3. Extract the dataset using gzip and place it into the 'data' folder
4. Create file called 'database.ini', this will handle connections to the Postgres DB

Sample databse.ini
```
[postgresql]
host=localhost
database=sdss
user=admin
password=admin
```

### Packages required
1. scikit-learn
2. Numpy
3. Pandas
4. Psycopg2


## Running the experiment

1. To run AIDE, use
 
python main.py

2. To run the random exploration baseline, use

python main_baseline.py
