# Run summary

## RQ1: which approach detects sexism best?

**test_results**

| detector | scenario | coverage | f1_macro | f1_macro_ci_low | f1_macro_ci_high | auprc | auprc_ci_low | auprc_ci_high | mcc |
|---|---|---|---|---|---|---|---|---|---|
| transformer_full | full | 1.0 | 0.832 | 0.818 | 0.845 | 0.839 | 0.818 | 0.858 | 0.665 |
| classic_xgboost_full | full | 1.0 | 0.785 | 0.77 | 0.799 | 0.771 | 0.749 | 0.794 | 0.569 |
| transformer_masked | masked | 1.0 | 0.783 | 0.769 | 0.796 | 0.701 | 0.671 | 0.73 | 0.565 |
| classic_logistic_regression_full | full | 1.0 | 0.777 | 0.761 | 0.791 | 0.732 | 0.705 | 0.758 | 0.554 |
| classic_linear_svm_full | full | 1.0 | 0.776 | 0.761 | 0.791 | 0.736 | 0.709 | 0.761 | 0.553 |
| classic_naive_bayes_full | full | 1.0 | 0.724 | 0.707 | 0.74 | 0.643 | 0.613 | 0.675 | 0.449 |
| classic_xgboost_masked | masked | 1.0 | 0.688 | 0.672 | 0.704 | 0.559 | 0.527 | 0.595 | 0.376 |
| classic_logistic_regression_masked | masked | 1.0 | 0.683 | 0.666 | 0.698 | 0.532 | 0.501 | 0.565 | 0.366 |
| classic_linear_svm_masked | masked | 1.0 | 0.677 | 0.661 | 0.692 | 0.531 | 0.499 | 0.563 | 0.354 |
| judge_gemma4_12b | judge | 1.0 | 0.674 | 0.658 | 0.69 | 0.715 | 0.685 | 0.744 | 0.464 |
| baseline_lexicon | baseline | 1.0 | 0.664 | 0.646 | 0.681 | 0.386 | 0.36 | 0.414 | 0.343 |
| classic_naive_bayes_masked | masked | 1.0 | 0.65 | 0.634 | 0.665 | 0.483 | 0.451 | 0.518 | 0.3 |
| judge_consensus | judge | 1.0 | 0.638 | 0.624 | 0.653 | 0.724 | 0.694 | 0.752 | 0.436 |
| judge_llama3.1_latest | judge | 1.0 | 0.626 | 0.611 | 0.641 | 0.705 | 0.676 | 0.735 | 0.411 |
| judge_qwen3.5_9b | judge | 1.0 | 0.596 | 0.581 | 0.61 | 0.705 | 0.675 | 0.734 | 0.383 |
| baseline_trivial | baseline | 1.0 | 0.431 | 0.427 | 0.435 | 0.242 | 0.229 | 0.256 | 0.0 |

**comparison_mcnemar**

| first | second | n | only_first_correct | only_second_correct | statistic | p | p_holm | significant |
|---|---|---|---|---|---|---|---|---|
| transformer_full | classic_xgboost_full | 4000 | 333 | 176 | 47.811 | 0.0 | 0.0 | True |
| transformer_full | baseline_lexicon | 4000 | 589 | 191 | 202.063 | 0.0 | 0.0 | True |
| classic_xgboost_full | baseline_lexicon | 4000 | 557 | 316 | 65.979 | 0.0 | 0.0 | True |
| transformer_masked | classic_xgboost_masked | 4000 | 514 | 244 | 95.463 | 0.0 | 0.0 | True |
| transformer_masked | baseline_lexicon | 4000 | 615 | 388 | 50.923 | 0.0 | 0.0 | True |
| classic_xgboost_masked | baseline_lexicon | 4000 | 565 | 608 | 1.504 | 0.22 | 0.22 | False |
| classic_naive_bayes_full | classic_naive_bayes_masked | 4000 | 315 | 82 | 135.577 | 0.0 | 0.0 | True |
| classic_logistic_regression_full | classic_logistic_regression_masked | 4000 | 405 | 108 | 170.791 | 0.0 | 0.0 | True |
| classic_linear_svm_full | classic_linear_svm_masked | 4000 | 426 | 109 | 186.647 | 0.0 | 0.0 | True |
| classic_xgboost_full | classic_xgboost_masked | 4000 | 412 | 128 | 148.313 | 0.0 | 0.0 | True |
| transformer_full | transformer_masked | 4000 | 305 | 134 | 65.831 | 0.0 | 0.0 | True |
| judge_consensus | transformer_full | 4000 | 287 | 1203 | 561.896 | 0.0 | 0.0 | True |
| judge_consensus | classic_xgboost_full | 4000 | 380 | 1139 | 378.251 | 0.0 | 0.0 | True |
| judge_consensus | baseline_lexicon | 4000 | 693 | 1211 | 140.383 | 0.0 | 0.0 | True |

## RQ2: is there signal beyond the lexicon?

**subgroup_auc**

| detector | scenario | subgroup | n | prevalence | subgroup_auc |
|---|---|---|---|---|---|
| classic_naive_bayes_full | full | with_term | 668 | 0.57 | 0.851 |
| classic_naive_bayes_full | full | without_term | 3332 | 0.177 | 0.783 |
| classic_naive_bayes_masked | masked | with_term | 668 | 0.57 | 0.615 |
| classic_naive_bayes_masked | masked | without_term | 3332 | 0.177 | 0.779 |
| classic_logistic_regression_full | full | with_term | 668 | 0.57 | 0.911 |
| classic_logistic_regression_full | full | without_term | 3332 | 0.177 | 0.823 |
| classic_logistic_regression_masked | masked | with_term | 668 | 0.57 | 0.629 |
| classic_logistic_regression_masked | masked | without_term | 3332 | 0.177 | 0.81 |
| classic_linear_svm_full | full | with_term | 668 | 0.57 | 0.916 |
| classic_linear_svm_full | full | without_term | 3332 | 0.177 | 0.821 |
| classic_linear_svm_masked | masked | with_term | 668 | 0.57 | 0.628 |
| classic_linear_svm_masked | masked | without_term | 3332 | 0.177 | 0.809 |
| classic_xgboost_full | full | with_term | 668 | 0.57 | 0.943 |
| classic_xgboost_full | full | without_term | 3332 | 0.177 | 0.833 |
| classic_xgboost_masked | masked | with_term | 668 | 0.57 | 0.627 |
| classic_xgboost_masked | masked | without_term | 3332 | 0.177 | 0.825 |
| judge_consensus | judge | with_term | 668 | 0.57 | 0.851 |
| judge_consensus | judge | without_term | 3332 | 0.177 | 0.89 |
| judge_gemma4_12b | judge | with_term | 668 | 0.57 | 0.852 |
| judge_gemma4_12b | judge | without_term | 3332 | 0.177 | 0.87 |
| judge_llama3.1_latest | judge | with_term | 668 | 0.57 | 0.864 |
| judge_llama3.1_latest | judge | without_term | 3332 | 0.177 | 0.857 |
| judge_qwen3.5_9b | judge | with_term | 668 | 0.57 | 0.823 |
| judge_qwen3.5_9b | judge | without_term | 3332 | 0.177 | 0.888 |
| baseline_lexicon | baseline | with_term | 668 | 0.57 | 0.557 |
| baseline_lexicon | baseline | without_term | 3332 | 0.177 | 0.5 |
| baseline_trivial | baseline | with_term | 668 | 0.57 | 0.5 |
| baseline_trivial | baseline | without_term | 3332 | 0.177 | 0.5 |
| transformer_full | full | with_term | 668 | 0.57 | 0.966 |
| transformer_full | full | without_term | 3332 | 0.177 | 0.898 |
| transformer_masked | masked | with_term | 668 | 0.57 | 0.732 |
| transformer_masked | masked | without_term | 3332 | 0.177 | 0.898 |

**influential_terms**

| model | feature | shap_importance | direction | lexicon_fraction | lexicon_term |
|---|---|---|---|---|---|
| classic_xgboost_full | wom | 0.226 | sexist | 0.0 | False |
| classic_xgboost_full | bitch | 0.161 | sexist | 0.986 | True |
| classic_xgboost_full | fem | 0.131 | sexist | 0.0 | False |
| classic_xgboost_full | men  | 0.13 | sexist | 0.0 | False |
| classic_xgboost_full | ? | 0.073 | not_sexist | 0.0 | False |
| classic_xgboost_full |  whor | 0.069 | sexist | 0.978 | True |
| classic_xgboost_full | puss | 0.069 | sexist | 0.932 | True |
| classic_xgboost_full | itch | 0.067 | sexist | 0.799 | True |
| classic_xgboost_full | bitc | 0.066 | sexist | 0.982 | True |
| classic_xgboost_full | whor | 0.063 | sexist | 0.967 | True |
| classic_xgboost_full |  bitc | 0.058 | sexist | 0.993 | True |
| classic_xgboost_full | . | 0.058 | sexist | 0.0 | False |
| classic_xgboost_full | cun | 0.056 | sexist | 0.943 | True |
| classic_xgboost_full | omen  | 0.051 | sexist | 0.0 | False |
| classic_xgboost_full |  wom | 0.043 | sexist | 0.0 | False |
| classic_xgboost_full | men | 0.043 | sexist | 0.0 | False |
| classic_xgboost_full | [ | 0.042 | not_sexist | 0.0 | False |
| classic_xgboost_full | slut | 0.041 | sexist | 0.954 | True |
| classic_xgboost_full | women | 0.041 | sexist | 0.0 | False |
| classic_xgboost_full |  fem | 0.04 | sexist | 0.0 | False |
| classic_xgboost_full | thot | 0.039 | sexist | 0.0 | False |
| classic_xgboost_full | sa | 0.038 | not_sexist | 0.0 | False |
| classic_xgboost_full |  w | 0.037 | sexist | 0.009 | False |
| classic_xgboost_full |  cun | 0.036 | sexist | 0.96 | True |
| classic_xgboost_full | b | 0.033 | sexist | 0.038 | False |
| classic_xgboost_full |  a  | 0.032 | sexist | 0.0 | False |
| classic_xgboost_full |  th | 0.031 | not_sexist | 0.0 | False |
| classic_xgboost_full |  ma | 0.03 | sexist | 0.005 | False |
| classic_xgboost_full | as  | 0.029 | not_sexist | 0.0 | False |
| classic_xgboost_full | en | 0.029 | not_sexist | 0.0 | False |
| classic_xgboost_full |  puss | 0.028 | sexist | 0.956 | True |
| classic_xgboost_full | en  | 0.028 | sexist | 0.0 | False |
| classic_xgboost_full | y  | 0.028 | not_sexist | 0.012 | False |
| classic_xgboost_full |  ho | 0.027 | sexist | 0.038 | False |
| classic_xgboost_full | itc | 0.027 | sexist | 0.787 | True |
| classic_xgboost_full |  pe | 0.027 | not_sexist | 0.0 | False |
| classic_xgboost_full | i  | 0.026 | not_sexist | 0.0 | False |
| classic_xgboost_full | en ar | 0.026 | sexist | 0.0 | False |
| classic_xgboost_full | wome | 0.024 | sexist | 0.0 | False |
| classic_xgboost_full | s a | 0.024 | sexist | 0.016 | False |
| classic_xgboost_full |  a | 0.023 | not_sexist | 0.0 | False |
| classic_xgboost_full | [u | 0.023 | not_sexist | 0.0 | False |
| classic_xgboost_full | wo | 0.022 | sexist | 0.0 | False |
| classic_xgboost_full | i' | 0.022 | not_sexist | 0.0 | False |
| classic_xgboost_full | es | 0.022 | sexist | 0.022 | False |
| classic_xgboost_full | m | 0.022 | not_sexist | 0.005 | False |
| classic_xgboost_full | ho | 0.022 | sexist | 0.041 | False |
| classic_xgboost_full | th | 0.021 | not_sexist | 0.0 | False |
| classic_xgboost_full | re | 0.021 | not_sexist | 0.02 | False |
| classic_xgboost_full | whore | 0.021 | sexist | 0.983 | True |
| classic_xgboost_masked | wom | 0.113 | sexist | 0.0 | False |
| classic_xgboost_masked |  . | 0.102 | sexist | 0.0 | False |
| classic_xgboost_masked | men  | 0.08 | sexist | 0.0 | False |
| classic_xgboost_masked | he | 0.069 | sexist | 0.0 | False |
| classic_xgboost_masked | fem | 0.066 | sexist | 0.0 | False |
| classic_xgboost_masked | ? | 0.05 | not_sexist | 0.0 | False |
| classic_xgboost_masked | omen  | 0.05 | sexist | 0.0 | False |
| classic_xgboost_masked |  a  | 0.049 | sexist | 0.0 | False |
| classic_xgboost_masked |  wom | 0.038 | sexist | 0.0 | False |
| classic_xgboost_masked | [ | 0.037 | not_sexist | 0.0 | False |
| classic_xgboost_masked |  fem | 0.036 | sexist | 0.0 | False |
| classic_xgboost_masked | women | 0.026 | sexist | 0.0 | False |
| classic_xgboost_masked | h | 0.026 | not_sexist | 0.0 | False |
| classic_xgboost_masked | thot | 0.025 | sexist | 0.0 | False |
| classic_xgboost_masked | [u | 0.023 | not_sexist | 0.0 | False |
| classic_xgboost_masked | ] | 0.023 | not_sexist | 0.0 | False |
| classic_xgboost_masked | as  | 0.021 | not_sexist | 0.0 | False |
| classic_xgboost_masked | . | 0.021 | sexist | 0.0 | False |
| classic_xgboost_masked | sa | 0.019 | not_sexist | 0.0 | False |
| classic_xgboost_masked | wome | 0.018 | sexist | 0.0 | False |
| classic_xgboost_masked | te | 0.018 | not_sexist | 0.0 | False |
| classic_xgboost_masked | men | 0.017 | sexist | 0.0 | False |
| classic_xgboost_masked |  ,  | 0.016 | sexist | 0.0 | False |
| classic_xgboost_masked | i' | 0.016 | not_sexist | 0.0 | False |
| classic_xgboost_masked |  , | 0.015 | sexist | 0.0 | False |
| classic_xgboost_masked | s a | 0.014 | sexist | 0.0 | False |
| classic_xgboost_masked | i  | 0.014 | not_sexist | 0.0 | False |
| classic_xgboost_masked | en ar | 0.014 | sexist | 0.0 | False |
| classic_xgboost_masked | o  | 0.013 | sexist | 0.0 | False |
| classic_xgboost_masked |  [ur | 0.013 | not_sexist | 0.0 | False |
| classic_xgboost_masked | omen | 0.013 | sexist | 0.0 | False |
| classic_xgboost_masked |  be | 0.012 | sexist | 0.0 | False |
| classic_xgboost_masked | ers | 0.012 | not_sexist | 0.0 | False |
| classic_xgboost_masked | chi | 0.011 | not_sexist | 0.0 | False |
| classic_xgboost_masked | n | 0.011 | sexist | 0.0 | False |
| classic_xgboost_masked | r li | 0.011 | sexist | 0.0 | False |
| classic_xgboost_masked | ho | 0.01 | sexist | 0.0 | False |
| classic_xgboost_masked |  [url | 0.01 | not_sexist | 0.0 | False |
| classic_xgboost_masked | i hav | 0.01 | not_sexist | 0.0 | False |
| classic_xgboost_masked |  is a | 0.01 | sexist | 0.0 | False |
| classic_xgboost_masked |  fa | 0.01 | sexist | 0.0 | False |
| classic_xgboost_masked | ma | 0.01 | sexist | 0.0 | False |
| classic_xgboost_masked | en | 0.009 | not_sexist | 0.0 | False |
| classic_xgboost_masked |  no  | 0.009 | sexist | 0.0 | False |
| classic_xgboost_masked |  ma | 0.009 | sexist | 0.0 | False |
| classic_xgboost_masked | ike a | 0.009 | sexist | 0.0 | False |
| classic_xgboost_masked | : | 0.009 | not_sexist | 0.0 | False |
| classic_xgboost_masked | re | 0.009 | not_sexist | 0.0 | False |
| classic_xgboost_masked | wo | 0.009 | sexist | 0.0 | False |
| classic_xgboost_masked |  [ | 0.009 | not_sexist | 0.0 | False |

## RQ3: are the judges reliable?

**judges_prompt_selection**

| model | prompt | n | coverage | auprc |
|---|---|---|---|---|
| gemma4:12b | minimal | 385 | 1.0 | 0.785 |
| gemma4:12b | definition | 385 | 1.0 | 0.784 |
| gemma4:12b | examples | 385 | 1.0 | 0.761 |
| qwen3.5:9b | minimal | 385 | 1.0 | 0.555 |
| qwen3.5:9b | definition | 385 | 1.0 | 0.649 |
| qwen3.5:9b | examples | 385 | 1.0 | 0.677 |
| llama3.1:latest | minimal | 385 | 1.0 | 0.642 |
| llama3.1:latest | definition | 385 | 1.0 | 0.685 |
| llama3.1:latest | examples | 385 | 1.0 | 0.685 |

**judges_stability**

| model | kind | alpha | ci_low | ci_high | versions |
|---|---|---|---|---|---|
| gemma4:12b | repetition | 1.0 | 1.0 | 1.0 | 3 |
| gemma4:12b | reformulation | 0.853 | 0.818 | 0.886 | 5 |
| qwen3.5:9b | repetition | 1.0 | 1.0 | 1.0 | 3 |
| qwen3.5:9b | reformulation | 0.804 | 0.764 | 0.84 | 5 |
| llama3.1:latest | repetition | 1.0 | 1.0 | 1.0 | 3 |
| llama3.1:latest | reformulation | 0.755 | 0.711 | 0.798 | 5 |

**judges_agreement**

| first | second | n | kappa | krippendorff_alpha | ci_low | ci_high |
|---|---|---|---|---|---|---|
| judge_gemma4_12b | judge_qwen3.5_9b | 3999 | 0.574 | nan | nan | nan |
| judge_gemma4_12b | judge_llama3.1_latest | 3999 | 0.624 | nan | nan | nan |
| judge_qwen3.5_9b | judge_llama3.1_latest | 4000 | 0.626 | nan | nan | nan |
| panel | panel | 4000 | nan | 0.607 | 0.588 | 0.624 |

**judges_explanation_cases**

| case | id | target | judge | verdict | explanation |
|---|---|---|---|---|---|
| true_positive | 12453 | 1 | judge_gemma4_12b | 1.0 | The text uses a derogatory slur ("bitch") to describe a person's behavior or attitude, which constitutes sexist language. |
| true_positive | 12453 | 1 | judge_qwen3.5_9b | 1.0 | The post is sexist because it relies on a derogatory gender stereotype that equates women with "bitches" and suggests their value or behavior is defined by this insult. |
| true_positive | 12453 | 1 | judge_llama3.1_latest | 1.0 | The post uses the derogatory term "bitch" to refer to a woman, which is a sexist slur. |
| true_positive | 1549 | 1 | judge_gemma4_12b | 1.0 | The text is sexist because it implies that a woman's status as a "female" exempts her from being treated as a responsible adult by the legal system. |
| true_positive | 1549 | 1 | judge_qwen3.5_9b | 1.0 | The post is sexist because it relies on harmful gender stereotypes that women are inherently dangerous or emotionally unstable while men are rational, and it explicitly advocates for treating a woman differently (and more harshly) than a man solely based on her gender. |
| true_positive | 1549 | 1 | judge_llama3.1_latest | 1.0 | The post implies that the woman would be treated unfairly by the justice system because of her gender, and also uses a derogatory tone to describe her appearance. |
| true_positive | 15247 | 1 | judge_gemma4_12b | 1.0 | The statement promotes a negative and discriminatory stereotype by generalizing the behavior and character of all women. |
| true_positive | 15247 | 1 | judge_qwen3.5_9b | 1.0 | The post is sexist because it relies on a gender stereotype that falsely claims women who express indifference are actually caring deeply, thereby dismissing their autonomy and character based solely on their gender. |
| true_positive | 15247 | 1 | judge_llama3.1_latest | 1.0 | This post is sexist because it implies that women are dishonest and manipulative when they say they don't care, perpetuating a negative stereotype about women's behavior. |
| false_positive | 1621 | 0 | judge_gemma4_12b | 1.0 | The statement implies a sexist stereotype that a woman's actions are inherently suspicious or problematic based solely on her gender. |
| false_positive | 1621 | 0 | judge_qwen3.5_9b | 1.0 | The post is sexist because it implies that a woman's actions or character are responsible for a man's behavior, thereby blaming the victim and reinforcing harmful gender stereotypes. |
| false_positive | 1621 | 0 | judge_llama3.1_latest | 1.0 | The post implies that the woman's actions are the cause of the man's situation, perpetuating a sexist stereotype that women are responsible for men's problems. |
| false_positive | 13723 | 0 | judge_gemma4_12b | 0.0 | The text is a neutral report of a news headline and does not contain language that discriminates against or demeans individuals based on gender. |
| false_positive | 13723 | 0 | judge_qwen3.5_9b | 1.0 | The post is sexist because it dismisses and mocks a woman's valid criticism of a TV show by attributing her stance to "intolerant feminist" bias rather than engaging with the actual argument about representation. |
| false_positive | 13723 | 0 | judge_llama3.1_latest | 1.0 | The post is sexist because it uses the term "feminist" in a derogatory way to imply that women who advocate for equality are intolerant and tyrannical. |
| false_positive | 17364 | 0 | judge_gemma4_12b | 1.0 | The phrase "body slam her" uses a violent action directed at a female subject, which can be interpreted as promoting gender-based violence or objectification. |
| false_positive | 17364 | 0 | judge_qwen3.5_9b | 1.0 | The phrase "body slam her" is a violent command directed at a woman that implies physical abuse based solely on her gender, which constitutes sexism. |
| false_positive | 17364 | 0 | judge_llama3.1_latest | 1.0 | The post contains a violent threat directed towards a woman, which is a clear example of sexism. |
| false_negative | 9100 | 1 | judge_gemma4_12b | 0.0 | The text expresses frustration with a situation or activity but does not target, disparage, or discriminate against a person or group based on their gender. |
| false_negative | 9100 | 1 | judge_qwen3.5_9b | 0.0 | The post expresses frustration with a media event but does not contain any abuse, negative sentiment, or stereotypes directed specifically at women based on their gender. |
| false_negative | 9100 | 1 | judge_llama3.1_latest | 0.0 | The post is criticizing a media event, but it does not contain any language or sentiment directed towards women based on their gender, making it not sexist. |
| false_negative | 19416 | 1 | judge_gemma4_12b | 0.0 | The text contains a swear word but does not target, disparage, or discriminate against anyone based on their gender. |
| false_negative | 19416 | 1 | judge_qwen3.5_9b | 0.0 | The post is not sexist because it uses the word "bitch" as an insult for a man's situation rather than directing gender-based abuse or stereotypes toward women. |
| false_negative | 19416 | 1 | judge_llama3.1_latest | 0.0 | The post contains a derogatory term ("bitch") but it is directed at the speaker's own frustration and not specifically at women, and the rest of the post is a sarcastic and humorous rant rather than a sexist attack. |
| false_negative | 16237 | 1 | judge_gemma4_12b | 0.0 | The text describes a conflict involving an accusation of sexual behavior rather than expressing prejudice or discrimination based on gender. |
| false_negative | 16237 | 1 | judge_qwen3.5_9b | 1.0 | The post is sexist because it validates and amplifies a derogatory accusation against a woman by framing her as an "actual whore" based on the false premise that she implicitly promised sex, thereby reinforcing harmful gender stereotypes about women's sexuality. |
| false_negative | 16237 | 1 | judge_llama3.1_latest | 0.0 | The post is discussing a situation where someone is being accused of "whoring", but it's not making a judgment or expressing a negative sentiment towards women based on their gender, it's rather a neutral description of the argument. |
| disagreement | 9989 | 1 | judge_gemma4_12b | 0.0 | The statement expresses a personal feeling or fear and does not use derogatory language, stereotypes, or discriminatory tropes to demean women as a group. |
| disagreement | 9989 | 1 | judge_qwen3.5_9b | 1.0 | The statement "Strong women scare me" is sexist because it expresses fear and discomfort based on a negative gender stereotype that attributes threatening qualities to women who are strong, thereby devaluing their empowerment. |
| disagreement | 9989 | 1 | judge_llama3.1_latest | 1.0 | The post expresses a negative sentiment directed towards women based on their gender, implying that women who are strong are intimidating or threatening. |
| disagreement | 15395 | 0 | judge_gemma4_12b | 0.0 | The text describes a situation without using derogatory language or promoting prejudice against a specific gender. |
| disagreement | 15395 | 0 | judge_qwen3.5_9b | 1.0 | The post is sexist because it expresses a negative sentiment and dismissive attitude towards women by suggesting they are unreasonable or difficult to deal with, which reinforces harmful gender stereotypes. |
| disagreement | 15395 | 0 | judge_llama3.1_latest | 1.0 | The post is sexist because it implies that a woman's opinions are invalid or untrustworthy simply because of her gender and ethnicity. |
| disagreement | 2299 | 0 | judge_gemma4_12b | 0.0 | The text is a humorous observation about gender roles and does not contain discriminatory language or prejudice. |
| disagreement | 2299 | 0 | judge_qwen3.5_9b | 1.0 | The post is sexist because it relies on a gender stereotype by implying that women are incapable of driving trucks, thereby denying them the ability to perform this physically demanding job based solely on their gender. |
| disagreement | 2299 | 0 | judge_llama3.1_latest | 1.0 | The post uses the derogatory term "wymyns" to refer to women, which is a sexist slur. |

## RQ4: how much does each approach cost?

**cost_by_method**

| method | scenario | preparation_time_s | test_time_s | ms_per_prediction | explanation_time_s | mean_vram_mb | mean_device_vram_mb |
|---|---|---|---|---|---|---|---|
| classic_naive_bayes_full | full | 110.066 | 1.306 | 0.327 | 0.0 | 0.0 | 1017.254 |
| classic_naive_bayes_masked | masked | 99.576 | 1.309 | 0.327 | 0.0 | 260.772 | 1235.842 |
| classic_logistic_regression_full | full | 192.921 | 1.314 | 0.328 | 0.0 | 0.0 | 1174.414 |
| classic_logistic_regression_masked | masked | 155.568 | 1.294 | 0.323 | 0.0 | 262.021 | 1238.085 |
| classic_linear_svm_full | full | 168.244 | 1.348 | 0.337 | 0.0 | 0.0 | 1080.732 |
| classic_linear_svm_masked | masked | 146.41 | 1.351 | 0.338 | 0.0 | 261.785 | 1237.661 |
| classic_xgboost_full | full | 10172.947 | 1.967 | 0.492 | 0.0 | 1279.337 | 2301.219 |
| classic_xgboost_masked | masked | 5027.804 | 1.495 | 0.374 | 0.0 | 1263.838 | 2062.849 |
| judge_gemma4_12b | judge | 731.455 | 692.085 | 173.021 | 2581.818 | 0.0 | 9857.999 |
| judge_llama3.1_latest | judge | 202.474 | 229.274 | 57.318 | 2232.072 | 0.004 | 6181.132 |
| judge_qwen3.5_9b | judge | 818.547 | 1053.385 | 263.346 | 4103.226 | 0.004 | 7455.435 |
| baseline_lexicon | baseline | 0.0 | 0.256 | 0.064 | 0.0 | 151.0 | 924.6 |
| baseline_trivial | baseline | 0.013 | 0.002 | 0.0 | 0.0 | 151.0 | 931.5 |
| transformer_full | full | 3756.726 | 6.475 | 1.619 | 0.0 | 5767.283 | 6543.377 |
| transformer_masked | masked | 4100.349 | 5.304 | 1.326 | 0.0 | 5290.624 | 6066.659 |
| judge_consensus | judge | 1752.476 | 1974.744 | 493.686 | 8917.115 | 0.003 | 7948.036 |

**execution_time**

| phase | stage | seconds |
|---|---|---|
| Preprocessing | hardware | 3.6 |
| Preprocessing | data | 2.0 |
| Preprocessing | cleaning | 48.1 |
| Preprocessing | lexicon | 4.4 |
| Data preparation | split | 2.0 |
| Data preparation | eda | 3.0 |
| Experiments | classical | 16082.9 |
| Experiments | transformer | 7914.0 |
| Experiments | judges_dev | 1756.6 |
| Evaluation | test | 37.5 |
| Evaluation | judges_test | 10894.1 |
| Evaluation | comparison | 52.9 |
| Evaluation | interpretation | 118.4 |
| Evaluation | cost | nan |
| Evaluation | report | nan |
| Total | (whole run) | 36919.5 |

## Hyperparameter choices

**classical_best**

| algorithm | scenario | trials | pruned_trials | best_trial | auprc_cv | auprc_cv_std | threshold | f1_macro_cv | params |
|---|---|---|---|---|---|---|---|---|---|
| naive_bayes | full | 60 | 0 | 16 | 0.632 | 0.017 | 0.586 | 0.722 | {"ngrams": "char_1_5", "min_df": 5, "sublinear_tf": true, "alpha": 0.11820592607623592} |
| logistic_regression | full | 60 | 0 | 31 | 0.705 | 0.011 | 0.325 | 0.76 | {"ngrams": "char_1_5", "min_df": 4, "sublinear_tf": true, "C": 3.985966240792658, "class_weight": "none"} |
| linear_svm | full | 60 | 0 | 53 | 0.711 | 0.012 | -0.28 | 0.761 | {"ngrams": "char_1_5", "min_df": 2, "sublinear_tf": true, "C": 0.27312953079768093, "class_weight": "none"} |
| xgboost | full | 60 | 16 | 48 | 0.749 | 0.01 | 0.273 | 0.777 | {"ngrams": "char_1_5", "min_df": 2, "sublinear_tf": true, "selection_percentile": 9.023988197951157, "n_estimators": 4248, "learning_rate": 0.005238107791655772, "subsample": 0.8338256063603535, "max_depth": 14, "min_child_weight": 1.390188084033847, "colsample_bytree": 0.7205750126865221, "colsample_bylevel": 0.7048524006992983, "reg_lambda": 0.0204739066983355, "reg_alpha": 0.12041847967259801} |
| naive_bayes | masked | 60 | 0 | 41 | 0.482 | 0.023 | 0.584 | 0.652 | {"ngrams": "char_1_5", "min_df": 5, "sublinear_tf": true, "alpha": 0.07561509651628713} |
| logistic_regression | masked | 60 | 0 | 54 | 0.507 | 0.023 | 0.529 | 0.67 | {"ngrams": "char_1_5", "min_df": 3, "sublinear_tf": true, "C": 2.2016979115051156, "class_weight": "balanced"} |
| linear_svm | masked | 60 | 0 | 51 | 0.506 | 0.023 | 0.045 | 0.669 | {"ngrams": "char_1_5", "min_df": 1, "sublinear_tf": true, "C": 0.24294016614030237, "class_weight": "balanced"} |
| xgboost | masked | 60 | 21 | 11 | 0.537 | 0.021 | 0.307 | 0.678 | {"ngrams": "char_1_5", "min_df": 5, "sublinear_tf": false, "selection_percentile": 8.15965155772192, "n_estimators": 2917, "learning_rate": 0.0074088301244323, "subsample": 0.6443479113560235, "max_depth": 6, "min_child_weight": 1.9958819368134357, "colsample_bytree": 0.43138600350401884, "colsample_bylevel": 0.7513620687253133, "reg_lambda": 0.05082095614942784, "reg_alpha": 0.004781329349214188} |

**transformer_chosen**

| base_model | revision | stage_hash | scenario | seed | learning_rate | batch_size | best_epoch | threshold | max_length | validation_auprc | validation_auprc_seed_mean | validation_auprc_seed_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| microsoft/deberta-v3-base | 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 | cb24e2ecdc62a664a2561c712136744b603a266a0af673f291145fb0257fec6f | full | 46 | 0.0 | 32 | 2 | 0.503 | 512 | 0.852 | 0.846 | 0.004 |
| microsoft/deberta-v3-base | 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 | cb24e2ecdc62a664a2561c712136744b603a266a0af673f291145fb0257fec6f | masked | 43 | 0.0 | 16 | 2 | 0.392 | 512 | 0.733 | 0.725 | 0.006 |

## Alerts

- **judges_dev:** llama3.1:latest: reformulation alpha = 0.75, below 0.8.
- **judges_test:** gemma4:12b: answered 100.0% of the test.
- **judges_test:** qwen3.5:9b: 0.1% of the explanations reached the token limit.
