ALL_CONCEPT_LEXICONS = {
    #"NEGATION": ["no", "not", "nor", "non", "neither", "none", "never", "aren't", "can't", "cannot", "hadn't", "hasn't", "haven't", "isn't", "didn't", "don't", "doesn't", "n't", "wasn't", "weren't", "nothing", "nobody", "less", "least", "little", "scant", "scarcely", "rarely", "hardly", "few", "rare", "unlikely"],
    "FEW_ADJ": ["no", "neither", "never", "n't", "few", "little", "scant", "barely", "rarely", "hardly", "few", "almost no", "barely any", "hardly any"],
    "3RD_PERSON_PRONOUN_(NOM)": ["they", "he", "she", "theirs", "hers", "his"],
    "OTHERS_NOM": ["they", "he", "she", "theirs", "hers", "his", "CITATION"],
    "3RD_PERSON_PRONOUN_(ACC)": ["her", "him", "them"], 
    "OTHERS_ACC": ["her", "him", "them"], 
    #"OTHERS_POSS": ["their", "his", "her"],
    "OTHERS_POSS": ["their", "his", "her",], # they, "its"
    "3RD_PERSON_REFLEXIVE": ["themselves", "himself", "herself"],
    "1ST_PERSON_PRONOUN_(NOM)": ["ours", "mine"], #"we", "i", 
    "SELF_NOM": ["we", "i"],
    "1ST_PERSON_PRONOUN_(ACC)": ["us", "me"],
    "SELF_ACC": ["us", "me"],
    "1ST_POSS_PRONOUN": ["my", "our"],
    "SELF_POSS": ["my", "our"],
    "1ST_PERSON_REFLEXIVE ": ["ourselves", "myself"],
    "REFERENTIAL": ["this", "that", "those", "these"],
    "REFERENTIAL_PL": ["those", "these"],
    "REFLEXIVE": ["itself ourselves", "myself", "themselves", "himself", "herself"],
    "QUESTION": ["?", "how", "why", "whether", "wonder"],
    #"GIVEN": ["noted", "mentioned", "addressed", "illustrated", "described", "discussed", "given", "outlined", "presented", "proposed", "reported", "shown", "taken"],
    "GIVEN": ["note", "mention", "address", "illustrate", "describe", "discuss", "give", "outline", "present", "propose", "report", "show", "take"],
    
    "PROFESSIONALS": ["collegue", "community", "computer scientist", "computational linguist", "discourse analyst", "expert", "investigator", "linguist", "logician", "philosopher", "psycholinguist", "psychologist", "researcher", "scholar", "semanticist", "scientist"],

    "DISCIPLINE": ["computerscience", "computer linguistics", "computational linguistics", "discourse analysis", "logics", "linguistics", "psychology", "psycholinguistics", "philosophy", "semantics", "lexical semantics", "several disciplines", "various disciplines"],
    
    "TEXT_NOUN": ["paragraph", "section", "subsection", "chapter"],
    
    "SIMILAR_NOUN": ["analogy", "similarity"],

    "SIMILAR_ADJ_PHRASE": ["in line with", "in common with", "analogous to", "similar to", "in analogy to", "similarty with", "the same as"],

# "braodly comparable", "directly comparable", "roughly comparable"
    "SIMILAR_ADJ": ["similar", "analogous", "kindred", "parallel", "identical", "comparable"],

    "COMPARISON_NOUN": ["accuracy", "baseline", "comparison", "competition", "evaluation", "inferiority", "measure", "measurement", "performance", "precision", "optimum", "recall", "superiority"],
    
    "CONTRAST_NOUN": ["contrast", "conflict", "clash", "difference", "point of departure"],

    "CONTRAST_ADJ": ["considerable", "crucial", "essential", "fundamental", "main", "majoir", "marked", "significant", "striking", "substantial"], # MEDAL

    'WIN_PHRASE': ["advantage to", "advantage over", "benefit over"],

    'CONTRAST_ADJ_PHRASE': ["#JJ contrast to", "in contrast to", "in contrast with", "in comparison to", "compare to", "difference from", "difference to", "difference between", "distinction between", "as oppose to", "contrary to"], #"while", "whereas", "unlike", 

    'OTHERS_PREP': ["whereas", "unlike", "while", "against"],

    "AIM_NOUN": ["aim", "direction", "goal", "intention", "objective", "purpose", "task", "theme", "topic", "concern", "emphasis", "focus"],

    "ARGUMENTATION_NOUN": ["assumption", "belief", "hypothesis", "hypotheses", "claim", "conclusion", "confirmation", "opinion", "recommendation", "stipulation", "view"],

    "PROBLEM_NOUN": ["Achilles heel", "caveat", "challenge", "complication", "contradiction", "damage", "danger", "deadlock", "defect", "detriment", "difficulty", "dilemma", "disadvantage", "disregard", "doubt", "downside", "drawback", "error", "failure", "fault", "foil", "flaw", "handicap", "hindrance", "hurdle", "ill", "inflexibility", "impediment", "imperfection", "intractability", "inefficiency", "inadequacy", "inability", "lapse", "limitation", "malheur", "mishap", "mischance", "mistake", "obstacle", "oversight", "pitfall", "problem", "shortcoming", "threat", "trouble", "vulnerability", "absence", "dearth", "deprivation", "lack", "loss", "fraught", "proliferation", "spate"],

    "QUESTION_NOUN": ["question", "conundrum", "enigma", "paradox", "phenomena", "phenomenon", "puzzle", "riddle"],

    "SOLUTION_NOUN": ["answer", "accomplishment", "achievement", "advantage", "benefit", "breakthrough", "contribution", "explanation", "feature", "idea", "improvement", "innovation", "insight", "justification", "proposal", "proof", "remedy", "success", "triumph", "verification", "victory"], # "solution"

    "INTEREST_NOUN": ["attention", "quest"],

    "RESULT_NOUN": ["evidence", "experiment", "finding", "progress", "observation", "outcome", "result"],

    #"METRIC_NOUN": ["bleu", "F-score", "F1-score", "F score", "F1 score", "precision", "recall", "accuracy", "correlation"],

    "CHANGE_NOUN": [ "adaptation", "enhancement", "extension", "generalization", "development", "modification", "refinement", "version", "variant", "variation"],

    "PRESENTATION_NOUN": ["article", "draft", "manuscript", "paper", "project", "report", "study", "work"],
    
    "NEED_NOUN": ["necessity", "motivation"],

    "WORK_NOUN": ["account", "algorithm", "analysis", "analyses", "approach", "approaches", "application", "architecture", "characterization", "characterisation", "component", "design", "extension", "formalism", "formalization", "formalisation", "framework", "implementation", "investigation", "machinery", "method", "methodology", "model", "module", "moduls", "process", "procedure", "program", "prototype", "research", "researches", "strategy", "system", "technique", "theory", "tool", "treatment", "work"],

    #"TRADITION_NOUN": ["acceptance", "community", "convention", "disciples", "disciplines", "folklore", "literature", "mainstream", "school", "tradition", "textbook"], # too divergent

    "CHANGE_ADJ": ["alternate", "alternative"],
    
    "GOOD_ADJ": ["adequate", "advantageous", "appealing", "appropriate", "attractive", "automatic", "beneficial", "capable", "cheerful", "clean", "clear", "compact", "compelling", "competitive", "comprehensive", "consistent", "convenient", "convincing", "constructive", "correct", "desirable", "distinctive", "efficient", "effective", "elegant", "encouraging", "exact", "faultless", "favourable", "feasible", "flawless", "good", "helpful", "impeccable", "innovative", "insightful", "intensive", "meaningful", "neat", "perfect", "plausible", "positive", "polynomial", "powerful", "practical", "preferable", "precise", "principled", "promising", "pure", "realistic", "reasonable", "reliable", "right", "robust", "satisfactory", "simple", "sound", "successful", "sufficient", "systematic", "tractable", "usable", "useful", "valid","valuable", "unlimited", "well worked out", "well", "enough", "well - motivated"],
    
    "BAD_ADJ": ["absent", "ad - hoc", "adhoc", "ad hoc", "annoying", "ambiguous", "arbitrary", "awkward", "bad", "brittle", "brute - force", "brute force", "careless", "confounding", "contradictory", "defect", "defunct", "disturbing", "elusive", "erraneous", "expensive", "exponential", "false", "fallacious", "frustrating", "haphazard", "ill - defined", "imperfect", "impossible", "impractical", "imprecise", "inaccurate", "inadequate", "inappropriate", "incomplete", "incomprehensible", "inconclusive", "incorrect", "inelegant", "inefficient", "inexact", "infeasible", "infelicitous", "inflexible", "implausible", "inpracticable", "improper", "insufficient", "intractable", "invalid", "irrelevant", "labour - intensive", "laborintensive", "labour intensive", "labor intensive", "laborious", "limited - coverage", "limited coverage", "limited", "limiting", "meaningless", "modest", "misguided", "misleading", "nonexistent", "NP - hard", "NP - complete", "NP hard", "NP complete", "questionable", "pathological", "poor", "prone", "protracted", "restricted", "scarce", "simplistic", "suspect", "time - consuming", "time consuming", "toy", "unacceptable", "unaccounted for", "unaccounted - for", "unaccounted", "unattractive", "unavailable", "unavoidable", "unclear", "uncomfortable", "unexplained", "undecidable", "undesirable", "unfortunate", "uninnovative", "uninterpretable", "unjustified", "unmotivated", "unnatural", "unnecessary", "unorthodox", "unpleasant", "unpractical", "unprincipled", "unreliable", "unsatisfactory", "unsound", "unsuccessful", "unsuited", "unsystematic", "untractable", "unwanted", "unwelcome", "useless", "vulnerable", "weak", "wrong", "too", "overly", "only"],

    "BEFORE_ADJ": ["early", "initial", "past", "previous", "prior"],
    
    "CONTRAST_ADJ": ["different", "distinguishing", "contrary", "competing", "rival"],

    "CONTRAST_ADV": ["differently", "distinguishingly", "contrarily", "otherwise", "other than", "contrastingly", "imcompatibly", "on the other hand", ],

    "TRADITION_ADJ": ["better known", "better - known", "cited", "classic", "common", "conventional", "current", "customary", "established", "existing", "extant", "available", "favourite", "fashionable", "general", "obvious", "long - standing", "mainstream", "modern", "naive", "orthodox", "popular", "prevailing", "prevalent", "published", "quoted", "seminal", "standard", "textbook", "traditional", "trivial", "typical", "widespread", "well - established", "well - known", "widely assumed", "unanimous", "usual"],

    "MANY": ["extensive", "a number of", "a body of", "a substantial number of", "a substantial body of", "most", "many", "several", "various", "some"],

    "HELP_NOUN": ['help', 'aid', 'assistance', 'support' ],

    "SENSE_NOUN": ['sense', 'spirit', ],

    "GRAPHIC_NOUN": ['table', 'tab', 'figure', 'fig', 'example' ],
    
    "COMPARISON_ADJ": ["evaluative", "superior", "inferior", "optimal", "better", "best", "worse", "worst", "greater", "larger", "faster", "weaker", "stronger"],

    "PROBLEM_ADJ": ["demanding", "difficult", "hard", "non - trivial", "nontrivial"],
    
    "RESEARCH_ADJ": ["empirical", "experimental", "exploratory", "ongoing", "quantitative", "qualitative", "preliminary", "statistical", "underway"],

    "AWARE_ADJ": ["unnoticed", "understood", "unexplored"],

    "NEED_ADJ": ["necessary", "indispensable", "requisite"],

    "NEW_ADJ": ["new", "novel", "state - of - the - art", "state of the art", "leading - edge", "leading edge", "enhanced"],

    "FUTURE_ADJ": ["further", "future"],

    "POTENTIAL_ADJ": [ "possible", "potential", "conceivable", "viable"], # previously HEDGE_ADJ,

    "HEDGE_ADJ": [], # from GD
    
    "MAIN_ADJ": ["main", "key", "basic", "central", "crucial", "critical", "essential", "eventual", "fundamental", "great", "important", "key", "largest", "main", "major", "overall", "primary", "principle", "serious", "substantial", "ultimate"],

    "CURRENT_ADV": ["currently", "presently", "at present", "recently"],
    
    "CURRENT_ADJ": ["current", "present"], # added 30 mar 2023

    "TEMPORAL_ADV": ["finally", "briefly", "next"],

    "SPECULATION": [],

    "CONTRARY": [],

    "SUBJECTIVITY": [],

    "STARSEM_NEGATION": [  "contrary", "without", "n't", "none", "nor", "nothing", "nowhere", "refused", "nobody", "means", "never", "neither", "absence", "except", "rather", "no", "for", "fail", "not", "neglected", "less", "prevent",  # not used
 ],

    'DOWNTONERS': [ 'almost', 'barely', 'hardly', 'merely', 'mildly', 'nearly', 'only', 'partially', 'partly', 'practically', 'scarcely', 'slightly', 'somewhat', ],

    'AMPLIFIERS': [ 'absolutely', 'altogether', 'completely', 'enormously', 'entirely', 'extremely', 'fully', 'greatly', 'highly', 'intensely', 'strongly', 'thoroughly', 'totally', 'utterly', 'very', ],

    
    'PUBLIC_VERBS': ['acknowledge', 'admit', 'agree', 'assert', 'claim', 'declare', 'deny', 'explain', 'hint', 'insist', 'mention', 'proclaim', 'promise', 'protest', 'remark', 'report', 'say', 'suggest', 'write', ], #'complain', 'reply', 'swear', 
    
    'PRIVATE_VERBS': [ 'anticipate', 'assume', 'believe', 'conclude', 'decide', 'demonstrate', 'determine', 'discover', 'doubt', 'estimate', 'find', 'hear', 'hope', 'imagine', 'imply', 'indicate', 'infer', 'know', 'learn', 'notice', 'prove', 'realize', 'realise', 'recognize', 'recognise', 'remember', 'reveal', 'see', 'show', 'suppose', 'think', 'understand', ], #'feel',  'guess', 'forget', 'fear', 'mean', 
    
    'SUASIVE_VERBS': [ 'agree', 'arrange', 'ask', 'beg', 'command', 'decide', 'demand', 'grant', 'insist', 'instruct', 'ordain', 'pledge', 'pronounce', 'propose', 'recommend', 'request', 'stipulate', 'suggest', 'urge', ]


    }


ALL_ACTION_LEXICONS = {
    "AFFECT": ["afford", "believe", "decide", "feel", "hope", "imagine", "regard", "trust", "think"], 

    "ARGUMENTATION": ["agree", "accept", "advocate", "argue", "claim", "conclude", "comment", "defend", "embrace", "hypothesize", "imply", "insist", "posit", "postulate", "reason", "recommend", "speculate", "stipulate", "suspect"],

    "AWARE": ["be unaware", "be familiar with", "be aware", "be not aware", "know of"],

    "BETTER_SOLUTION": ["boost", "enhance", "defeat", "improve", "go beyond", "perform better", "outperform", "outweigh", "surpass"],
    
    "CHANGE": ["adapt", "adjust", "augment", "combine", "change", "decrease", "elaborate", "expand", "expand on", "extend", "derive", "incorporate", "increase", "manipulate", "modify", "optimize", "optimise", "refine", "render", "replace", "revise", "substitute", "tailor", "upgrade"], 
         
    "COMPARISON": ["assess", "compare", "compete", "evaluate", "test"],
         
    "DENOTATION": ["be", "denote", "represent" ],

    "INSPIRATION": ["inspire", "motivate" ],

    "AGREE": ["agree with", "side with" ],

    "CONTINUE": ["adopt", "base", "be base on", 'base on', "derive from", "originate in",  "borrow", "build on", "follow", "following", "originate from", "originate in", 'start from', 'proceed from'],

    "CONTRAST": ["be different from", "be distinct from", "conflict", "contrast", "clash", "differ from", "distinguish", "differentiate", "disagree", "disagreeing", "dissent", "oppose", "contrast #RB with", "contrast with"],

    "FUTURE_INTEREST": [ "be interest in", "plan on", "plan to", "expect to", "intend to", "hope to"],

    "HEDGING_MODALS": ["could", "might", "may", "should" ],

    "FUTURE_MODALS": ["will", "going to" ],

    "SHOULD": ["should" ],

    "INCREASE": ["increase", "grow", "intensify", "build up", "explode" ],

    "INTEREST": ["aim", "ask", "address", "attempt", "be concern", "be interest", "be motivate", "concern", "consider", "concentrate on", "explore", "focus", "intend to", "look at how", "pursue", "seek", "study", "try", "target", "want", "wish", "wonder"], # "like to"

    "NEED": ["be dependent on", "be reliant on", "depend on", "lack", "need", "necessitate", "require", "rely on"],

    "PRESENTATION": ["describe", "discuss", "give", "introduce", "note", "notice", "point out", "present", "propose", "put forward", "recapitulate", "remark", "report", "say", "show", "sketch", "state", "suggest", "talk about", "indicate"], 

    "PROBLEM": ["abound", "aggravate", "arise", "be cursed", "be incapable of", "be force to", "be limite to", "be problematic", "be restrict to", "be trouble", "be unable to", "contradict", "damage", "degrade", "degenerate", "fail", "fall prey to", "fall short to", "force", "force", "hinder", "impair", "impede", "inhibit", "misclassify", "misjudge", "mistake", "misuse", "neglect", "obscure", "overestimate", "over - estimate", "overfit", "over - fit", "overgeneralize", "over - generalize", "overgeneralise", "over - generalise", "overgenerate", "over - generate", "overlook", "pose", "plague", "preclude", "prevent", "remain", "resort to", "restrain", "run into", "settle for", "spoil", "suffer from", "threaten", "thwart", "underestimate", "under - estimate", "undergenerate", "under - generate", "violate", "waste", "worsen"], 
        
    "RESEARCH": ["apply", "analyze", "analyse", "build", "calculate", "categorize", "categorise", "characterize", "characterise", "choose", "check", "classify", "collect", "compose", "compute", "conduct", "confirm", "construct", "count", "define", "delineate", "design", "detect", "determine", "equate", "estimate", "examine", "expect", "formalize", "formalise", "formulate", "gather", "identify", "implement", "indicate", "inspect", "integrate", "interpret", "investigate", "isolate", "maximize", "maximise", "measure", "minimize", "minimise", "observe", "predict", "realize", "realise", "reconfirm", "revalidate", "simulate", "select", "specify", "test", "verify", "work on"], 

    "SEE": [ "see", "view", "treat", "consider" ],

    "SIMILAR": ["bear comparison", "be analogous to", "be alike", "be related to", "be closely relate to", "be reminiscent of", "be the same as", "be similar to", "be in a similar vein to", "have much in common with", "have a lot in common with", "pattern with", "resemble", "correspond to"], #IYWS-MEDAL

    "SIMILAR_NOUN": ["resemblance", "parallel"], # "bear a #JJ resemblance between" #IYWS-MEDAL

    "SIMILAR_CONNECTORS": ["likewise", "in the same way ,", "in the same way as", "in the same way that", "as it be the case", "as #RD as", "as #VV in", "as #VV above", "as #VV by"], #IYWS-MEDAL

    "DEGREE_ADJ": ["significant", "striking", "certain", "close",], #IYWS-MEDAL

    "DEGREE_ADV": [ "broadly", "fairly", "somewhat", "strikingly" ], # just use #RB?? #IYWS-MEDAL

    "SOLUTION": ["accomplish", "account for", "achieve", "apply to", "answer", "alleviate", "allow for", "allow", "allow", "avoid", "benefit", "capture", "clarify", "circumvent", "contribute", "cope with", "cover", "cure", "deal with", "demonstrate", "develop", "devise", "discover", "elucidate", "escape", "explain", "fix", "gain", "go a long way", "guarantee", "handle", "help", "implement", "justify", "lend itself", "make progress", "manage", "mend", "mitigate", "model", "obtain", "offer", "overcome", "perform", "preserve", "prove", "provide", "realize", "realise", "rectify", "refrain from", "remedy", "resolve", "reveal", "scale up", "sidestep", "solve", "succeed", "tackle", "take care of", "take into account", "treat", "warrant", "work well", "yield"],

    "TEXTSTRUCTURE": ["begin by", "illustrate", "conclude by", "organize", "organise", "outline", "return to", "review", "start by", "structure", "summarize", "summarise", "turn to"], 
         
    "USE": ["apply", "employ", "use", "make use of", "utilize", "implement", 'resort to']
    }
