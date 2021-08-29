# Neural Network Charity Analysis

[Module 3]() is to use Python 

## Analysis

From the CSV file containing more than 34,000 organizations that received funding form Alphabet Soup over the years, the following metadata about each organization have been captured

* **EIN** and **NAME** — Identification columns
* **APPLICATION_TYPE** — Alphabet Soup application type
* ** AFFILIATION** — Affiliated sector of industry
* **CLASSIFICATION** — Government organization classification
* **USE_CASE** — Use case for funding
* **ORGANIZATION** — Organization type
* **STATUS** — Active status
* **INCOME_AMT** — Income classification
* **SPECIAL_CONSIDERATIONS** — Special consideration for application
* **ASK_AMT** — Funding amount requested
* **IS_SUCCESSFUL** — Was the money used effectively

### Deliverable 1: Preprocessing Data for a Neural Network Model (30 points)

Starting the Analysis from the DataFrame, `application_df`, below,

![DataFrame Info](img/mltechptwy-mod-3-dev-1-info.png)

#### Drop unnecessary columns

Since the **EIN** and the **NAME** are indentifying columns, they will not provide generalizable patterns and thus and not effective for machine learning. Thus, they were dropped from `application_df`

```python
# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(columns=['EIN', 'NAME'])
```

#### Addressing categorical columns

The prior image also highlights the fact that many of the columns contain categorical data (`object`). The evaluate this preprocessing step, the amount of unique values associated within each column is evaluated:

```python
# Determine the number of unique values in each column.
application_df.nunique()
```

Then a density plot per categorical column is plotted to determine the cutoff point to bin all uncommon unique values

![](img/mltechptwy-mod-3-dev-1-density-plot.png)

This process is repeated for all (`object`) columns with a unique categorical count of greater than 10. Which means APPLICATION_TYPE and CLASSIFICATION.

#### One Hot Encoding

```python
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
# application_cat_lt_10 = set(s[s].index) - set(application_cat_gr_10)
encode_df = pd.DataFrame(enc.fit_transform(application_df[application_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(application_cat)
encode_df.head()
```

```python
# Merge one-hot encoded features and drop the originals
application_df = application_df.merge(encode_df, left_index=True, right_index=True)
application_df = application_df.drop(application_cat, 1)
application_df.head()
```

### Deliverable 2: Compile, Train, and Evaluate the Model (20 points)

### Deliverable 3: Optimize the Model (20 points)

## Results

## Summary
