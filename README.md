# Data Science University Course 

## Course Overview
This Data Science course provides a comprehensive introduction to the field, covering various data processing, analysis, and visualization techniques using R within Jupyter Notebooks. The course is designed for university students and includes practical projects and datasets to enhance learning.

## Repository Structure

### datasets/
This directory contains the datasets used throughout the course. Each file is related to specific exercises and projects covered in the course.


- **germination_csv.csv**: Data related to seed germination studies.
- **hair_eye_color_csv.csv**: Dataset containing information on hair and eye color.
- **Hypothesis_csv1.csv**: Data used for hypothesis testing.
- **knn1_csv.csv**: Dataset for K-Nearest Neighbors (KNN) algorithm.
- **pollutant_csv.csv**: Dataset on pollutant levels.
- **Toy_sales_csv.csv**: Sales data for toy products.
- **travelled_abroad_csv.csv**: Data on individuals who have traveled abroad.
- **wbc_csv.csv**: White blood cell count data.

### projects/
This directory contains Jupyter notebooks for various projects and exercises. These notebooks utilize R for data analysis and visualization.

- **00.ipynb**: Introduction and course overview notebook.
- **01.ipynb**: Initial data exploration and basic data analysis.
- **02.color.ipynb**: Analysis related to the hair and eye color dataset.
- **02.germination.ipynb**: Analysis of the germination dataset.
- **02.ipynb**: Additional exercises and projects.
- **02.pollutant.ipynb**: Analysis of pollutant data.
- **03.ipynb**: Advanced data analysis techniques.
- **03.boxplot.ipynb**: Creating and analyzing box plots.
- **03.scatterplot.ipynb**: Creating and analyzing scatter plots.
- **04.ipynb**: Further advanced analysis and visualization techniques.
- **05.ipynb**: Continuation of advanced topics.
- **06.ipynb**: Specific project work.
- **07.ipynb**: Additional project exercises.
- **08.ipynb**: Further project exercises.
- **Lab1.ipynb**: Laboratory exercises.


### Other files
- **.gitignore**: Specifies files to be ignored by Git.
- **LICENSE**: License information for the course materials.
- **README.md**: This readme file.

## Getting Started

1. **Clone the Repository**: Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/arya2004/data-science.git
   ```

2. **Install Dependencies**: Ensure you have Jupyter and the necessary R kernel installed. You can install them using:
   ```bash
   # Install Jupyter
   pip install jupyter
   
   # Install R and the IRKernel
   R
   install.packages('IRkernel')
   IRkernel::installspec(user = FALSE)
   ```

3. **Open Jupyter Notebooks**: Navigate to the `projects` directory and open the desired `.ipynb` file using Jupyter Notebook:
   ```bash
   jupyter notebook 00.ipynb
   ```

## Course Workflow
1. **Start with 00.ipynb**: Begin with the introductory notebook to get an overview of the course.
2. **Follow the Notebooks in Sequence**: Proceed through the notebooks in numerical order. Each notebook builds on the concepts and skills from the previous ones.
3. **Use the Datasets**: Apply the datasets in the `datasets/` directory to complete the exercises and projects in each notebook.



## License
This project is licensed under the terms of the [MIT license](LICENSE).

## Contact
For any questions or concerns, please open an issue on GitHub or contact the course instructor.

