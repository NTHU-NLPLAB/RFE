LABEL_PATTERNS = {
    "AIM":{ #"IN_ORDER_TO_FORMULAIC", # could be OWN, could be OTH
           #"US_AGENT",
           "OUR_AIM_AGENT",
           "AIM_REF_AGENT",
           "US_SHOW_FORMULAIC", # added by me
           "NOVEL_FORMULAIC", #added by me
           },

    "BAS": {"GENERAL_FORMULAIC",
            "US_PREVIOUS_FORMULAIC",
            "CONTINUE_FORMULAIC",
            "MOTIVATING_FORMULAIC",
            },

    "BKG": {"GAP_FORMULAIC",
            #"TRADITION_FORMULAIC",
            "BACKGROUND_FORMULAIC",
            "USEFUL_FORMULAIC",
            "GAP_AGENT",
            #"PROBLEM_AGENT", # too general => remove
            "SOLUTION_AGENT", # too general => remove
            "NEGATIVE_RESULT_FORMULAIC", # added by me
            "DESCRIPTION_FORMULAIC", # added by me
            "PROBLEM_SOLUTION_AGENT", # added by me
            "THEM_AGENT",
            },

    "CTR": {#"DISCOURSE_CONTRAST_FORMULAIC",
            "CONTRAST2_FORMULAIC",
            "CONTRAST_FORMULAIC",
            # COMPARISON_FORMULAIC from OTH??
            },
            
    "OTH": {"THEM_FORMULAIC",
            "SIMILARITY_FORMULAIC",
            "COMPARISON_FORMULAIC",
            "US_PREVIOUS_AGENT",
            "REF_AGENT",
            "THEM_PRONOUN_AGENT",
            "THEM_ACTIVE_AGENT",
            "PRIOR_WORK_FORMULAIC",
            "GENERAL_AGENT",
            "ALIGN_FORMULAIC",
            },

    "OWN": {"METHOD_FORMULAIC",
            "REF_US_AGENT", # moved from OWN to AIM and then back here
            "HERE_FORMULAIC",
            "FUTURE_FORMULAIC",
            "DETAIL_FORMULAIC",
            "USE_FORMULAIC",
            "FUTURE_WORK_FORMULAIC",
            #"HEDGING_FORMULAIC", # too general, removed
            "PRESENT_WORK_FORMULAIC",
            "EXTENDING_WORK_FORMULAIC",
            "EXTENDING_WORK2_FORMULAIC",
            "US_CONCLUDE_FORMULAIC", # added by me
            "IMPROVE_FORMULAIC",
            },

    "TXT": {"TEXTSTRUCTURE_FORMULAIC",
            "GRAPHIC_FORMULAIC",
            "NO_TEXTSTRUCTURE_FORMULAIC",
            "TEXTSTRUCTURE_AGENT"
            }
}