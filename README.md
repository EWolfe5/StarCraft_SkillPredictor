# StarCraft SkillPredictor
This repository contains the code and resources for a project focused on classifying StarCraft players based on various gameplay attributes. The classification model aims to predict the skill level or league of a player using machine learning techniques.

## **Overview**
This project aimed to analyze and predict the skill level of players in a video game based on various features. The work was divided into two main sections: Exploratory Data Analysis (EDA), which included feature engineering and data preprocessing, and Modeling.

In the EDA phase, the dataset was analyzed and preprocessed. The 'GameID' feature, which served no purpose in prediction or analysis, was dropped. Missing data and outliers were handled appropriately. The relationships between different features and the target variable were explored using plots and statistical analysis.

The EDA revealed that certain features showed an upward trend as the player's skill level increased. Other features exhibited different patterns or no discernible patterns. The correlation between features was also examined to identify potential predictive variables and multicollinearity.

Based on the EDA, feature selection was performed using mutual information gain and chi-square analysis. The top 7 features were selected for further analysis, while the rest were removed.

In the Modeling phase, various machine learning models, including Logistic Regression, Random Forest, Support Vector Machines (SVM), XGBoost, and a Neural Network were trained and evaluated. The metrics used for evaluation were weighted precision, recall, and F1-score, considering the imbalanced nature of the target variable.

The initial performance of the models showed room for improvement. The Random Forest and Logistic Regression models exhibited slightly better performance compared to the others. However, overfitting was observed in some models, indicating a need for hyperparameter optimization and further improvement.

Hyperparameter optimization was performed using cross-validation techniques. However, the optimized models still showed suboptimal performance on the test set, suggesting the need for additional adjustments.

Overall, the analysis revealed that the models struggled to generalize the key features of the dataset and did not perform satisfactorily. Despite efforts to optimize the models, they were not able to accurately predict the skill level of players.

It is important to note that the dataset imbalance and potential overfitting issues could have contributed to the models' performance. Different techniques, such as feature engineering, ensemble methods, or exploring other advanced models, could be considered to enhance the predictive power of the models.

Despite the limitations and suboptimal results, the project provides valuable insights into the dataset and serves as a foundation for further exploration and improvement in predicting players' skill levels in the video game.

## **Hypothetical**
*Hypothetical: after seeing your work, your stakeholders come to you and say that they can collect more data, but want your guidance before starting. How would you advise them based on your EDA and model results?*

After carefully analyzing the existing data and considering the results of our exploratory data analysis (EDA) and model findings, I would like to provide guidance on collecting additional data to further enhance our player classification system. The following recommendations aim to improve the accuracy and effectiveness of our classifier:

1. Higher Granularity of Data:

* Instead of solely measuring "hours per week," I suggest collecting data at a more granular level, such as "hours per day," broken down by individual days. This will provide a clearer understanding of players' commitment and playing habits.
* By incorporating this level of detail, we can distinguish between casual players who excel despite limited playtime and professional players who dedicate extensive hours to the game, such as streamers. This differentiation can help reveal important patterns and performance indicators.

2. Emphasize Decision-Making and Reaction Abilities:

* Our chi-square and information gain analysis revealed that features related to players' decision-making and reaction abilities significantly contributed to higher accuracy in our classification model.
* To strengthen our classifier, I recommend collecting additional data that explicitly measures these aspects. Consider including metrics such as strategic thinking, adaptability, resource management, and reaction times. These insights will provide valuable predictors of a player's performance and skill level.

3. Expand Dataset with High-Quality Matches:

* Given the diverse strategies and play styles in StarCraft, our classifier would benefit from an expanded dataset containing high-quality matches.
* I propose focusing on acquiring data from top-tier players, such as professional gamers, as they possess a deep understanding of the game mechanics and consistently exhibit exceptional skills.
* Additionally, it is worth noting that the number of professional players may be limited. Therefore, securing a larger dataset with high-quality matches will provide a more comprehensive representation of different playstyles and skill levels.

4. Collect Data on Regional Servers:

* StarCraft is played across multiple regional servers, and player rankings can vary across these regions.
* To improve our classification system, I recommend collecting data from various regional servers, such as Korean, European, and American servers. This regional division will help us better understand the performance and skill levels specific to each region.
* By accounting for regional variations, we can refine our classifier and produce more accurate predictions that align with players' rankings within their respective regions.

By following these recommendations, we can significantly enhance our player classification system. The collection of more granular data, particularly focusing on decision-making abilities, expanding the dataset with high-quality matches, and accounting for regional server differences, will provide a more robust foundation for accurate player classification.

## **Notes**
For a more in-depth explanation of the project's process and analysis, please refer to the Jupyter notebooks provided in the repository. These notebooks cover the exploratory data analysis ([EDA](https://github.com/EWolfe5/StarCraft_SkillPredictor/blob/main/EDA.ipynb)), data preprocessing, and [model](https://github.com/EWolfe5/StarCraft_SkillPredictor/blob/main/Modeling.ipynb) development steps.

While a separate data pipeline has been created, it's important to note that the code for the pipeline was derived from the initial EDA and data preprocessing stages. This ensures a consistent and reliable flow of data for training and evaluating the classification model.

Feel free to explore the notebooks and code to gain insights into the project's methodology and implementation. If you have any questions or suggestions, please don't hesitate to reach out.

Happy coding!
