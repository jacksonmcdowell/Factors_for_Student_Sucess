---
title: "Final Project"
author: "Jackson McDowell"
format:
  html:
    toc: true
    toc-location: left
    theme: vaporcd
    highlight-style: breeze
    self-contained: true
---

CMD + SHIFT + I or CTRL + SHIFT + I will help you create code chunks quickly.

```{python}
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
student_performance = pd.read_csv("C:/Users/JacksonMcDowell/Downloads/archive (1).zip")
student_performance.columns
student_performance.describe(include='all')
student_performance.shape
```

To start of, I imported the libraries that I knew I would need in order to run the statistical models that I was hoping to run (logistical and linear regressions). I then read in my dataset, which I then decided to look at the columns to decide what variables I would want to keen in my interest on. The dataset had many variables, and the main focus was to see what had the biggest effects on student's exam scores. I then used describe to quickly get an overview of the dataset and understand how the variables were distributed. After this I also wanted to look at the amount of data that was in the dataset with shape. I realize that not all of these steps are needed, but I just wanted to be sure I got familiar with the data. 

```{python}
student_performance["above_average_students"] = (student_performance["Exam_Score"]>=student_performance["Exam_Score"].median())
student_performance["above_average_students"].head()
student_performance['above_average_students'] = student_performance['above_average_students'].apply(lambda x: 1 if x else 0)
student_performance['above_average_students'].head()
top_logit = smf.logit('above_average_students ~ Previous_Scores + Hours_Studied + Attendance + C(Access_to_Resources) + C(Extracurricular_Activities) + Sleep_Hours + C(Family_Income) + Tutoring_Sessions + C(Parental_Involvement)', student_performance).fit()
top_logit.summary()
```

This section of code creates a new variable that labels whether a student performed above average. First, I compared each student’s exam score to the median exam score in the dataset. If a student scored at or above the median, they were marked as True; if not, they were False. Then I converted these True/False values into 1s and 0s so they could be used in a logistic regression model. I also was sure to check the data with .head(), just so that I saw for myself that everything was working well. After that, I ran a logistic regression predicting which students were above average using factors such as hours studied, attendance, resource access, family income, extracurricular activity, sleep, tutoring, and parental involvement. The top_logit.summary() output then showed which variables were statistically significant in determining whether a student ends up above average or not. This was very important in my process of seeing what variables helped the most with student performance. 

```{python}
linear_fit = smf.ols(
    'Exam_Score ~ Previous_Scores + Hours_Studied + Attendance + C(Access_to_Resources) + C(Extracurricular_Activities) \
     + Sleep_Hours + C(Family_Income) + Tutoring_Sessions \
     + C(Parental_Involvement)',
    data = student_performance).fit()
linear_fit.summary()
sns.regplot(
    data = student_performance,
    x = "Hours_Studied",
    y = "Exam_Score",
    scatter_kws={"alpha":0.3},
    line_kws={"linewidth":3}
)
plt.title("Hours Studied and Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()
```

This code runs a multiple linear regression to predict a student’s exam score based on academic behaviors and support factors. The purpose of this model is to measure how strongly different student habits influence performance and determine which ones contribute most to higher scores. After fitting the model, I generated a regression plot to visualize one of the key relationships Hours Studied vs. Exam Score, which clearly shows that students who study more tend to score higher. I thought this would help show the general idea in my presentation. I ran this analysis because the main goal of my project is to identify what drives student success, and this model allows me to quantify which factors have the greatest impact.
```{python}
student_performance["predicted_score"] = linear_fit.predict()
student_performance[["Exam_Score","predicted_score"]].head()
```

This code uses the linear regression model to generate predicted exam scores for each student. The first line creates a new column called predicted_score using the model’s fitted values, and the second line displays the first five actual scores next to their predicted values. This comparison helps evaluate how well the model estimates performance and shows how closely predictions align with real student outcomes.
```{python}
print(linear_fit.params.sort_values(ascending=False))
```

This line sorts the regression coefficients from highest to lowest. In simple terms, it lists which factors had the strongest positive effect on exam score at the top, and which had the strongest negative impact at the bottom. This makes it easy to see which variables mattered most in predicting student performance and helps answer the core goal of the project
