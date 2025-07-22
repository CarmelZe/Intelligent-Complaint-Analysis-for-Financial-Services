
    ## Exploratory Data Analysis Summary
    
    The initial analysis of the complaint dataset revealed several key insights. First, the distribution of complaints across product categories was uneven, with Credit card accounting for 98.2% of complaints, while Money transfers represented only 1.8%. This suggests that certain financial products generate significantly more customer complaints than others.
    
    The complaint narratives varied substantially in length, with an average of 196.6 words per narrative (standard deviation: 216.2). The shortest narrative contained 2 words, while the longest had 6469 words. The length analysis was performed on a representative sample of the data for efficiency.
    
    After filtering for our five target products and removing empty narratives, we retained 82,164 complaints (0.5% of the original dataset). The cleaned narratives were standardized by converting to lowercase, removing special characters, and eliminating common boilerplate phrases to improve embedding quality in subsequent steps.
    