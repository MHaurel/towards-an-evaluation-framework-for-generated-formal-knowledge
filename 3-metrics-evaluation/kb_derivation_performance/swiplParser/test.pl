 tags and a final answer. The question: Provide which anatomical structure likely injured. We'll output the answer: "left diaphragm" or "left hemidiaphragm."<think>
% Facts
wound_location(left_midaxillary_line, upper_8th_rib).
vital_sign(pulse, 110).
vital_sign(respirations, 22).
vital_sign(blood_pressure, 90/65).

% Anatomical structures with their typical locations
anatomical_structure(left_diaphragm, located_at, upper_8th_rib_midaxillary).
anatomical_structure(left_lung_lower_lobe, located_at, below_9th_rib).
anatomical_structure(spleen, located_at, left_upper_quadrant).

% Rule: structure injured if wound location matches its typical location
likely_injury(WoundLoc, Injury) :-
    anatomical_structure(Injury, located_at, WoundLoc).

% Query to find the most likely injury
?- likely_injury(upper_8th_rib_midaxillary, Injury).
Injury = left_diaphragm.