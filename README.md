# Spacy IBD Cohort Identification Models ⚗

## Open Source Spacy Models to Detect IBD Cohorts from Free-Text 🧨

### Dr Matt Stammers - 18/05/2024

Having completed a systematic review on NLP applied to Gastroenterology I have seen how little model sharing there is presently and have resolved not to do that with this IBD cohort identification project so that others can try to replicate the process and at least have something to benchmark against as a starter-point. Once the project is fully completed I will upload all the model code including the code explaining how to use UMLS properly which I had to work out myself (as most of the code out there is quite old). 

🔬

I will also explain the reasons why I chose spacy and not medspacy/scispacy etc and provide further details on how to avoid many of the pitfalls I fell into while trying to navigate what has become a very crowded field in terms of tools, most of which underperform.

🧪

Hopefully this will save future individuals going on the same journey the pain I went through working this out.

🩺

Enjoy!

Disclaimer: Please read the paper that accompanies them before playing with the pipelines. If these models are used in production on another site they are likely to underperform the original training sites performance and they cannot be relied upon alone to accurately identify IBD cohorts. 

However, if you do use them or want to collaborate then either reach out to me on github or submit and issue/pull request. 
