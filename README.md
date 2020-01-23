# aACP
Automatic Algorithm Change Protocols for regulating AI/ML-based Software as a Medical Device

Feng, Jean, Scott Emerson, and Noah Simon. 2019. “Approval Policies for Modifications to Machine Learning-Based Software as a Medical Device: A Study of Bio-Creep.” arXiv [stat.ML]. arXiv. http://arxiv.org/abs/1912.12413.

## Installation
Create a [python virtualenv](https://virtualenv.pypa.io/en/latest/) using the required packages in requirements.txt.
Create a scratch directory in this folder.
Also install the required R packages `optparse` and `ldbounds`.

To run the simulation scripts, install [SCons](https://scons.org/pages/download.html).
The simulation scripts live in the folders prefixed with "simulation."
Then you can run each simulation using `scons <simulation_folder_name>`.

