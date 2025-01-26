- **Prevalence of “Other”**  
  The model tends to default to “Other” for many statements. This results in high recall for “Other” but low recall for other classes. For instance, statements like “CC: Criticism of climate movement: Ad hominem attacks on key activists” were often misclassified as “Other” instead of the more specific “CC: Criticism of climate movement.”

- **Fine-Grained Categories Unrecognized**  
  Many labels, such as “URW: Amplifying war-related fears” or “CC: Hidden plots by secret schemes,” have zero recall. The complexity of these harder-to-detect narratives means the model misses key signals. For example, the statement “By continuing the war we risk WWIII” (which should fall under “URW: Amplifying war-related fears”) gets overlooked.

- **Difficulty with Ideologically Complex Statements**  
  The model struggles to capture subtle differences in statements with political or social nuances. For example, comparing “The West will attack other countries” vs. “We fear a broader conflict” requires context about future speculation—both ended up classified as “Other.”

- **Data Imbalance Impact**  
  Some categories, like “CC: Amplifying Climate Fears” or “URW: Discrediting Ukraine,” appear less frequently. The model often overlooks them. For instance, “Ukraine is associated with nazism” was entirely misclassified due to its rarity and high similarity to other war-related narratives.

- **Thresholding Might Be Too Strict**  
  Because this is a multi-label scenario, a single threshold (e.g., 0.3) can under-predict less common categories. Statements such as “Climate policies are ineffective” or “Russian invasion has strong national support” may have borderline probabilities, but get lumped into “Other” if they fall below the threshold.