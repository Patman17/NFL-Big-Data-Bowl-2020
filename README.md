# NFL: Big Data Bowl 2020

TDLR: [Canva NFL Big Data](https://www.canva.com/design/DADspHdgnOU/eGDqy2PaJWTmCk7ftvxrAw/view?utm_content=DADspHdgnOU&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)

![](https://github.com/patman17/NFL_Big_Data_Bowl/blob/master/images/bigdatabowl.png)
**American football** is a very complex sport with many factors and player interactions that determine the success of a play. Especially with 22 players on the field (11 on the offense and 11 on the defense), it can be extremely challenging to quantify the value of specific plays and actions within a play. Utlimately, the objective of football is for the offense to move the football towards the opposing team's side of the field to score in their end zone. The offense can achieve this through two methods:   
1) run (rush) or   
2) throw (pass) the football.   

Even with advent of the "passing league" where the specialization of players and quarterback friendly rules have made it easier to pass the football, roughly 1/3 of a team's offensive yardages still comes from rush plays. In addition, a good running team can control the tempo of the game and spread the focus of defense to both protecting against the run or pass. Thus, rushing plays are still paramount to the league and we are presented with a unique problem in the **NFL Big Data Bowl**.  

 ***How many yards will an NFL player gain after receiving a handoff?***  

With game, play, and player level data, can we construct a model to predict how many yards a rushing play will achieve? In developing this model, can we derive deeper insights on rushing plays and help the league better evaluate their ball carriers, supporting players, coach playcallings / decision and opposing defense?

This competition will again be provided with exclusive **Next Gen Stats** data that includes positional, directional, and movement data of every player on the field right at the moment of handoff. 

Please read through this document to get your environment set up and to get more details on all the files.

**NOTE** - for more information on the competition please visit [Kaggle NFL Big Data Bowl](https://www.kaggle.com/c/nfl-big-data-bowl-2020). 

## Table of Contents
1) Getting Started 
2) Installing Environment
3) File Descriptions
4) Methodology & Metric
5) Consolidated Findings & Summary
6) Last Words
7) Author

## 1) Getting Started

All data analysis and model creation have been done through usage of Python 3.7 and non-proprietary, open-source libraries. For the all workbooks, you will need to have a working python environment that utilize either Jupyter Notebook or Jupyter Lab.

The following list is all libraries utilized in creation of the models.
- Pandas
- Numpy
- Matplotlib
- tqdm
- Sklearn
- LightGBM

## 2) Installing Environment
If you do not have any python environment setup please visit following link:

[**Anaconda**- popular Python distribution w/ most common python libraries](https://www.anaconda.com/distribution/#download-section)

Anaconda should be able to set up the environment with most of the packages installed and apps such as Jupyter Notebook that can be launched from their dashboard.

The only library that is not packaged in Anaconda is **lightGBM**. Please pip install with the following line in the notebook.

```python
!pip install lightgbm
```
Any additional missing libraries you can conda install or pip install the library from the library documentation.

## 3) File Descriptions
This submission consists of several type of files that includes workbooks and supporting files. 

### Notebooks
1) **0-NFL Median Baseline.ipynb** - Jupyter notebook that construct the median benchmark we will be using for comparsion.
2) **1-NFL_RF & LGBM SP.ipynb** - Jupyter notebook that shows both the Random Forest and LightGBM models on just the raw Next Gen Stats data.
3) **2-NFL_RF & LGBM.ipynb** - Jupyter notebook with the final Featured-Engineered Random Forest and LightGBM models.
4) **3-NFL Play Plotter.ipynb** - Jupyter notebook that has the script for plotting NFL rushing plays
### Misc Files
5) **NFL_utilsV3.py** - utilies file containing all functions for data manipulation, cleaning, standardizing, feature engineering and scoring of the models
6) **NFL_play_plotterV2.py** - utilies file that contains code for plotting the rushing plays. Contains various ways to plot including by game, player or user defined criteria. 
### Datasets
10) **train.csv** - provided, unmodified training data from the competition consisting of 2017 to 2018 rushing plays.  
11) **NFL_2019.csv** - pulled training data from 2019 up to October 1st, 2019.  
12) **NFL_plot.csv** - constructed preprocessed data for plotting (able to be reconstructed under NFL Play Plotter notebook)  
### Presentation 
13) **NFL Big Data Bowl.pdf** - presentation slides of project

## 4) Methodology and Metric

**NOTE** - *For an in depth review of this project please continue forward and read through this section. However, skip to **5) Consolidated Findings and Summary** if you want the results only.*

### Evalutation Metric  
For this competition, the models will be evaluated on the Continuous Ranked Probability Score (CRPS). For each PlayId, you must predict a cumulative probability distribution for the yardage gained or lost. In other words, each column you predict indicates the probability that the team gains <= that many yards on the play. The CRPS is computed as follows:

    $$C = \frac{1}{199N} \sum_{m=1}^{N} \sum_{n=-99}^{99} (P(y \le n) -H(n - Y_m))^2,$$

where P is the predicted distribution, N is the number of plays in the test set, Y is the actual yardage and H(x) is the Heaviside step function (H(x)=1 for x ≥ 0 and zero otherwise).

### Methodology  
Beside the metric for evaluation, my approach to this competition was to construct a model that could utilize my domain knowledge of the NFL. Thus, I approached this problem with a decision tree based model framework since the features could be interpreted to either assist or hurt the model.

I will present my findings in the following format:
1) **NFL presentation** - please view my presentation at [Canva NFL Big Data](https://www.canva.com/design/DADspHdgnOU/eGDqy2PaJWTmCk7ftvxrAw/view?utm_content=DADspHdgnOU&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink) as there are some videos in the presentation. However you can also review the pdf presentation slides attached. This we give the summary of the models and findings.
2) **Median Benchmark** - after browsing the presentation you can review the **0-NFL Median Baseline.ipynb** as this will walk through constructing the benchmark that I will be comparing my model against and it will provide some context to data as well. 
3) **Raw Data Models** - in the first of model iterations we just throw the Next Gen data raw into the model without any feature engineering. The models are developed in **1-NFL_RF & LGBM SP.ipynb** 
4) **Feature-Engineered Models** - in the finalized models we construct new features center around the rusher and run in the same model framework and see how well the models work now. Refer to **2-NFL_RF & LGBM.ipynb**.  
5) **Play Plotter** - As an optional exploration, you can refer **3-NFL Play Plotter.ipynb**. This notebook has all the code I used to construct a plotting tool that was used supplementary to view the **Next Gen Data** in a meaningful way. There are several options and case example in this notebook already. One just need to uncomment the specific function to plot the rushing plays. I just plotted the first game of 2017 as an example.   

## 5) Consolidated Findings and Summary
If you haven't visit [Canva NFL Big Data](https://www.canva.com/design/DADspHdgnOU/eGDqy2PaJWTmCk7ftvxrAw/view?utm_content=DADspHdgnOU&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink), please do as it has my summary presentation.    

The result of the models as follow:  
CRSP = Continuous Ranked Probability Score  

1) **Median Benchmark**   
With Smoothing: 0.01431  
Without Smoothing: 0.01845
2) **Raw Data Models**  
Random Forest Train-Test Split: 0.01377   
Random Forest 3 K-Folds Cross Validation: 0.01382    
LGMB Train-Test Split: 0.01388   
LGMB 3 K-Folds Cross Validation: 0.01393
3) **Featured Engineered Models**  
Random Forest Train-Test Split: 0.01289     
Random Forest 3 K-Folds Cross Validation: 0.01324     
LGMB Train-Test Split: 0.01314   
LGMB 3 K-Folds Cross Validation: 0.1332 

Overall, we were able to improve our model with progressing iterations. In the Raw Data Models, we average roughly 0.0138 for the CRPS which is roughly .001 improvement over the smoothed benchmark and .004 for the regular benchmark. In th raw data, we found through feature importance it utilized primarily 3 features that all were game contextual data:
1) **yardline** - the yard marker where the line of scrimmage is at  
2) **distance** - yards needed for a first down  
3) **defenders in the box** - # of defenders lined up near the line of scrimmage  

The Raw Data Models could not utilize the Next Gen Stats data in a meaningful way. As it randomly picked attributes of different players with no consistent result. 

Next moving toward the Featured-Engineered Models we also get an improvement in CRPS of ~ 0.0131 so a decent improvement of roughly 0.0007. However, the greatest achievement was that the model was able to interpret the Next Gen Data in a meaningful way. The top 10 feature importance are:

1) **A_dx** - acceleration of the ball carrier in the x-component (towards downfield)
2) **A** - acceleration of the ball carrier in total magnitude
3) **S_dx** - speed of the ball carrier in x-component (towards downfield)
4) **Force** - weight of ball carrier times their acceleration (A)
5) **Back From Scrimmage** - yards rusher is back from scrimmage
6) **Defense Time to Rusher Minimum** - defense personnel that was the lowest time to rusher (calculated by distance/speed of defense player)
7)**S** - speed magnitude of ball carrier
8) **Defense Time to Rusher Minimum of 3 closest defensive players** - average the times of the 3 closest defensive players to the rusher
9) **Yardline_grid**- standarized yardline to the X,Y grid system that the Next Gen Stats is based off.
10) **Defense past scrimmages yards** - a basic measure of the penetration of the defense into the offensive backfield by adding up all the yards of any defensive personnel past the line of scrimmage. A higher number would mean that the defense was able to get past their blockers.

The **3 main takeaways** from developing our final model was that: 

1) Next Gen data dominated in importances. There was little game contextual data in the top features.
2) Data about the rusher / ball carrier was the most important
3) Defensive based data was then the second in importances to the model 
4) Therefore, the interactions between the rusher and defensive right at handoff is more definitive in predicting yardages of a rushing play. 

In summary, I was able to create a good model in predicting most common rush yardage situations. It still fails to predict the big breakoff plays. This is probably due to the additional factors / parameters that evolves from a more developed play. Also there is less instances of these plays to train the model on as well. In developing this model, we can see that utilizing the spatial proximity of the rusher to the defense is a great way to predict yardage. Finally, we were only given one timestamp data at handoff. We had more Next Gen data such as the aftermath of the play we could develop a very powerful model that could predict more than just yardage. 

## 6) Last Words / Future Improvements
Thanks again for **Kaggle** and **NFL** for hosting this competition. This was my first Kaggle competition and I had a ton of fun working on this problem as football is one of my passion and I was impressed with numerous ways people approached this problem. Even though my models did not score high on the evaluation metric compared to the leaderboard, I believe I was able to draw some powerful insights in the factors that determine a successful rushing play and learned a lot to develop my data science toolkit.

I hope to incorporate the following to improve my models:

1) Try the advance methods / models such as the top teams aka Neural Nets based approaches and transfomers.
2) Use more outside data / information. Especially player data to better represent the skillset of each player would be great to the model.
3) Additional feature engineering such as  better ways to identify running lanes or successful blocks.

Cheers!

## 7) Author
Patrick Ly  
[LinkedIn Profile](https://www.linkedin.com/in/patrick-m-ly/)  
[Github Profile](https://github.com/Patman17)





