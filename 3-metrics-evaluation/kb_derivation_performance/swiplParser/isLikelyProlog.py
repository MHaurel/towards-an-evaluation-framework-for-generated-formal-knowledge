from swiplParser import is_likely_prolog
from swiplParser import evaluate_prolog_syntax_per_clause

reasoning = """
% Define the symptoms and lab findings
symptom(abdominal_pain).
symptom(yellowish_discoloration_of_eyes).
symptom(itching).
symptom(weight_loss).
symptom(dark_urine).
symptom(clay_colored_stool).
lab_result(alanine_transaminase, 184).
lab_result(aspartate_transaminase, 191).
lab_result(alkaline_phosphatase, 387).
lab_result(total_bilirubin, 18).
ca19_9(positive).
serology_negative(hepatotropic_viruses).
abdominal_ct_findings(multifocal_short_segmental_stricture_of_bile_duct).
abdominal_ct_findings(mild_dilation).
abdominal_ct_findings(hypertrophy_of_caudate_lobe).
abdominal_ct_findings(atrophy_of_left_lateral_and_right_posterior_segments).
biopsy_result(periductal_fibrosis).
biopsy_result(atypical_bile_duct_cells_in_desmoplastic_stroma).

% Define possible conditions and their associations
condition(cholangiocarcinoma).
condition(chronic_hepatitis).
condition(primary_sclerosing_cholangitis).
condition(liver_fluke_infection).
condition(biliary_atresia).

% Define rules for cholangiocarcinoma
rule(cholangiocarcinoma, symptom(abdominal_pain)).
rule(cholangiocarcinoma, symptom(yellowish_discoloration_of_eyes)).
rule(cholangiocarcinoma, symptom(itching)).
rule(cholangiocarcinoma, symptom(weight_loss)).
rule(cholangiocarcinoma, lab_result(alanine_transaminase, Value), Value > 100).
rule(cholangiocarcinoma, lab_result(aspartate_transaminase, Value), Value > 100).
rule(cholangiocarcinoma, lab_result(alkaline_phosphatase, Value), Value > 300).
rule(cholangiocarcinoma, lab_result(total_bilirubin, Value), Value > 10).
rule(cholangiocarcinoma, ca19_9(positive)).
rule(cholangiocarcinoma, serology_negative(hepatotropic_viruses)).
rule(cholangiocarcinoma, abdominal_ct_findings(multifocal_short_segmental_stricture_of_bile_duct)).
rule(cholangiocarcinoma, abdominal_ct_findings(mild_dilation)).
rule(cholangiocarcinoma, abdominal_ct_findings(hypertrophy_of_caudate_lobe)).
rule(cholangiocarcinoma, abdominal_ct_findings(atrophy_of_left_lateral_and_right_posterior_segments)).
rule(cholangiocarcinoma, biopsy_result(periductal_fibrosis)).
rule(cholangiocarcinoma, biopsy_result(atypical_bile_duct_cells_in_desmoplastic_stroma)).

% Define rules for other conditions
rule(chronic_hepatitis, symptom(abdominal_pain)).
rule(chronic_hepatitis, symptom(yellowish_discoloration_of_eyes)).
rule(chronic_hepatitis, symptom(itching)).
rule(chronic_hepatitis, lab_result(alanine_transaminase, Value), Value > 100).
rule(chronic_hepatitis, lab_result(aspartate_transaminase, Value), Value > 100).
rule(chronic_hepatitis, lab_result(alkaline_phosphatase, Value), Value > 300).
rule(chronic_hepatitis, lab_result(total_bilirubin, Value), Value > 10).
rule(chronic_hepatitis, serology_positive(hepatotropic_viruses)).

rule(primary_sclerosing_cholangitis, symptom(abdominal_pain)).
rule(primary_sclerosing_cholangitis, symptom(yellowish_discoloration_of_eyes)).
rule(primary_sclerosing_cholangitis, symptom(itching)).
rule(primary_sclerosing_cholangitis, lab_result(alanine_transaminase, Value), Value > 100).
rule(primary_sclerosing_cholangitis, lab_result(aspartate_transaminase, Value), Value > 100).
rule(primary_sclerosing_cholangitis, lab_result(alkaline_phosphatase, Value), Value > 300).
rule(primary_sclerosing_cholangitis, lab_result(total_bilirubin, Value), Value > 10).
rule(primary_sclerosing_cholangitis, abdominal_ct_findings(multifocal_short_segmental_stricture_of_bile_duct)).
rule(primary_sclerosing_cholangitis, abdominal_ct_findings(mild_dilation)).
rule(primary_sclerosing_cholangitis, abdominal_ct_findings(hypertrophy_of_caudate_lobe)).
rule(primary_sclerosing_cholangitis, abdominal_ct_findings(atrophy_of_left_lateral_and_right_posterior_segments)).
rule(primary_sclerosing_cholangitis, biopsy_result(periductal_fibrosis)).

rule(liver_fluke_infection, symptom(abdominal_pain)).
rule(liver_fluke_infection, symptom(yellowish_discoloration_of_eyes)).
rule(liver_fluke_infection, symptom(itching)).
rule(liver_fluke_infection, lab_result(alanine_transaminase, Value), Value > 100).
rule(liver_fluke_infection, lab_result(aspartate_transaminase, Value), Value > 100).
rule(liver_fluke_infection, lab_result(alkaline_phosphatase, Value), Value > 300).
rule(liver_fluke_infection, lab_result(total_bilirubin, Value), Value > 10).
rule(liver_fluke_infection, abdominal_ct_findings(multifocal_short_segmental_stricture_of_bile_duct)).
rule(liver_fluke_infection, abdominal_ct_findings(mild_dilation)).
rule(liver_fluke_infection, abdominal_ct_findings(hypertrophy_of_caudate_lobe)).
rule(liver_fluke_infection, abdominal_ct_findings(atrophy_of_left_lateral_and_right_posterior_segments)).
rule(liver_fluke_infection, biopsy_result(periductal_fibrosis)).
rule(liver_fluke_infection, biopsy_result(atypical_bile_duct_cells_in_desmoplastic_stroma)).

rule(biliary_atresia, symptom(abdominal_pain)).
rule(biliary_atresia, symptom(yellowish_discoloration_of_eyes)).
rule(biliary_atresia, symptom(itching)).
rule(biliary_atresia, lab_result(alanine_transaminase, Value), Value > 100).
rule(biliary_atresia, lab_result(aspartate_transaminase, Value), Value > 100).
rule(biliary_atresia, lab_result(alkaline_phosphatase, Value), Value > 300).
rule(biliary_atresia, lab_result(total_bilirubin, Value), Value > 10).
rule(biliary_atresia, abdominal_ct_findings(mild_dilation)).
rule(biliary_atresia, abdominal_ct_findings(hypertrophy_of_caudate_lobe)).
rule(biliary_atresia, abdominal_ct_findings(atrophy_of_left_lateral_and_right_posterior_segments)).
rule(biliary_atresia, biopsy_result(periductal_fibrosis)).

% Query for the most likely condition
query(condition(Condition)) :- 
    findall(Condition, (rule(Condition, Symptom), symptom(Symptom)), Conditions),
    findall(Condition, (rule(Condition, LabResult), lab_result(LabResult, Value), Value > 100), LabConditions),
    findall(Condition, (rule(Condition, CTFinding), abdominal_ct_findings(CTFinding)), CTConditions),
    findall(Condition, (rule(Condition, BiopsyResult), biopsy_result(BiopsyResult)), BiopsyConditions),
    merge(Conditions, LabConditions, CTConditions, BiopsyConditions, AllConditions),
    count(AllConditions, Count),
    max_count(Count, Condition).
"""

# print(is_likely_prolog(reasoning))
if __name__ == "__main__":
    print(evaluate_prolog_syntax_per_clause(reasoning))