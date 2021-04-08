# aACP
Approval policies for modifications to machine learning‐based software as a medical device: A study of bio‐creep, Biometrics, 2020.
https://onlinelibrary.wiley.com/doi/10.1111/biom.13379
http://arxiv.org/abs/1912.12413.

## Installation
Create a [python virtualenv](https://virtualenv.pypa.io/en/latest/) using the required packages in requirements.txt.
Create a scratch directory in this folder.
Also install the required R packages `optparse` and `ldbounds`.

To run the simulation scripts, install [SCons](https://scons.org/pages/download.html).
The simulation scripts live in the folders prefixed with "simulation."
Then you can run each simulation using `scons <simulation_folder_name>`.

