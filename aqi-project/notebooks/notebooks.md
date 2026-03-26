1. Explain Block - 7; data is cleaned here and checked for abnormal PM ratios 2.5/10.

2. The Strongest Link: PM2.5 and PM10. There is a very strong positive correlation (~0.90) between PM2.5 and PM10. What it means: When one goes up, the other almost always follows. This is expected because PM2.5 is a subset of PM10. 
3. The "Independent" Gases. Surprisingly, in your current dataset, the gaseous pollutants (NO_2, SO_2, CO, and O_3) show almost zero correlation with the particulate matter (PM2.5/PM10) or with each other. The Data: Most correlation values for these gases range from -0.007 to 0.007.
4. Usually, in urban environments, NO_2 and CO correlate with PM2.5 because they all come from vehicle exhaust. The near-zero correlation suggests that either: The sources for these gases are geographically different from the dust/particulate sources. The IterativeImputer might have independently imputed missing values for these gases, potentially diluting the natural relationships in the raw data.
    This one needs an explanation; tie it to the original dataset source.

5. Explain Block - 8; Skewness and Kurtosis of data, define skewness and kurtosis
6. Very low Skewness (approx 0.01 to 0.15) indicates unusually symmetric distributions for air quality data. The Kurtosis is also low (negative), suggesting that extreme spikes are relatively rare across the entire timeline.
7. The "India-Wide" Geographic Trap. The original dataset isn't just one street corner; it’s an aggregate of many different Indian cities over 8 years.
   Source Diversity: In a city like Delhi, PM2.5 might be driven heavily by vehicle exhaust (high correlation with NO_2). But in a city like Jaisalmer, PM2.5 might be driven by desert dust (low correlation with NO_2).
   The "Dilution" Effect: When you calculate one single correlation coefficient for the entire dataset, you are essentially putting all these cities in a blender. The high-correlation cities get cancelled out by the low-correlation cities, leaving you with that "near-zero" result.
   Seasonal Shifts: In the original data, PM2.5 spikes massively during winter stubble burning, but NO_2 levels might not spike nearly as much. This seasonal "mismatch" breaks the mathematical link between the two.
8.  Explain block 8 to block having "CITY POLLUTANT BREAKDOWN (MEAN CONCENTRATIONS)"
9. Define AQI Categories as per our dataset; CPCB might not fit right here.
10. Explain blocks from "Block DSc perspective" to  "VIF block needs clean explanation" .
11. VIF (Variance Inflation Factor) is a measure used in statistics and Machine Learning to tell you if your "predictor" variables (like PM2.5, NO2, etc.) are too similar to each other. It helps you detect Multicollinearity, which is a fancy word for "redundancy" in your data.
    VIF = 1: The variable is completely unique. It’s like a scout giving you information no one else has.
    VIF between 1 and 5: The variables are slightly related, but it’s usually safe to keep them in your model. This is where your pollutants currently sit.
    VIF > 5 or 10: This is a "warning zone." It means the variable is highly redundant. The information it provides is already being told by other variables.
Classical Machine Learning models (like Logistic Regression or Linear Regression) get "confused" by high VIF.
13. Explain block "Check how pollutants change across AQI categories"

        Karthik Do Entries as required here, for better follow up.
14. 