import pandas as pd
import textwrap
import scipy.stats as stats

def print_stats(
        sample_mean: float, 
        sample_size: int, 
        population_mean: float, 
        population_sd: float, 
        alpha_1: float, 
        alpha_2: float,
        test_type: str = "left-tailed"
        ) -> None:
    """
    Print basic statistics for the dataset
    Args:
        sample_mean (float): Sample mean
        sample_size (int): Sample size
        population_mean (float): Population mean
        population_sd (float): Population standard deviation
        alpha_1 (float): Significance level 1
        alpha_2 (float): Significance level 2
        test_type (str): Type of hypothesis test ("left-tailed", "right-tailed", "two-tailed")
    Returns:
        None
    """
    alpha_levels = [alpha_1, alpha_2]
    z_score = (sample_mean - population_mean) / (population_sd / (sample_size ** 0.5))
    std_error = population_sd / (sample_size ** 0.5)

    if test_type == "left-tailed":
        p_value = stats.norm.cdf(z_score)
    elif test_type == "right-tailed":
        p_value = stats.norm.sf(z_score)
    else:  # two-tailed
        p_value = 2 * stats.norm.cdf(-abs(z_score))

    print(f"Std Err: {std_error:.2f}")
    print(f"Sample Mean: {sample_mean:.2f}")
    print(f"Test Statistic (z-score): {z_score:.2f}")
    print(f"p-value: {p_value:.4f}")

    for alpha in alpha_levels:
        print(f"At {alpha:.2f} significance level:")
        if(p_value < alpha):
            print("Reject the null hypothesis.")
        else:
            print("Don't reject the null hypothesis.")

    print("\n")

# Task descriptions
task_text = \
"Suppose the manufacturer claims that the mean lifetime of a ball bearing is 10000 hours. \
The auditing team stated that the mean lifetime is less than what is claimed. On the basis of a \
randomly chosen sample of 50 ball bearings as given in the dataset, at 0.05 significance level, \
can we reject the claim of the manufacturer? What will be your interpretation if the significance \
level is made as 0.01? Consider the data set titled “Hypothesis_csv1.csv”."

task_text_2 = \
"The nutrition label on a bag of potato chips says that a one ounce (28 gram) serving of \
potato chips has 130 calories and contains ten grams of fat, with three grams of saturated fat. \
A random sample of 35 bags yielded a sample mean of 134 calories with a standard deviation of 17 \
calories. Is there evidence that the nutrition label does not provide an accurate measure of \
calories in the bags of potato chips? We have verified the independence, sample size, and skew \
conditions are satisfied. Take alpha as 5% and 1%."

# Print the task description
print("Hypothesis Testing", end="\n\n")
print(textwrap.fill(task_text, width=100), end="\n\n")
print(textwrap.fill(task_text_2, width=100), end="\n\n")

# Load the dataset
df = pd.read_csv("../datasets/Hypothesis_csv1.csv")

# Display the contents of the dataset
print(df.head(n = 50), end="\n\n")

# Calculate necessary statistics from the dataset
sample_mean_df = df['Life_Hrs'].mean()
sample_size_df = df['Life_Hrs'].count()
population_mean_df = 10000
population_sd_df = df['Life_Hrs'].std()

# Print statistics and hypothesis test results
print_stats(sample_mean_df, sample_size_df, population_mean_df, population_sd_df, 0.05, 0.01, "left-tailed")

print_stats(134, 35, 130, 17, 0.05, 0.01, "two-tailed")
