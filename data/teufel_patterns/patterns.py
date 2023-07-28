FORMULAIC_PATTERNS = {
    'GENERAL_FORMULAIC': [ 'in @TRADITION_ADJ #JJ @WORK_NOUN', # BAS
                           'in @TRADITION_ADJ use @WORK_NOUN',
                           'in @TRADITION_ADJ @WORK_NOUN',
                           'in @MANY #JJ @WORK_NOUN',
                           'in @MANY @WORK_NOUN',
                           'in @BEFORE_ADJ #JJ @WORK_NOUN',
                           'in @BEFORE_ADJ @WORK_NOUN',
                           'in other #JJ @WORK_NOUN',
                           #'in other @WORK_NOUN',
                           'in such @WORK_NOUN' 
                           ],

    'THEM_FORMULAIC': [ 'according to CITATION', # OTH, BKG?
                        'like CITATION',
                        'CITATION style',
                        'a la CITATION',
                        'CITATION - style',
                        # below: added by me 31 mar 2023
                         "in the view of", 
                         "et al . @PRIVATE_VERBS that",
                         "et al . @PRESENTATION that",
                         "early @WORK_NOUN",
                        # added 08 jun 2023 from obsrving trainin data
                         "@GIVEN in CITATION",
                         "@PRESENTATION in CITATION",
                         '@OTHERS_POSS @WORK_NOUN @COMPARISON',
                        'the @WORK_NOUN @COMPARISON', # that => the
                        'the @PRESENTATION_NOUN @COMPARISON', # that => the
                        '@OTHERS_POSS @PRESENTATION_NOUN @COMPARISON',
                         ] ,
    
    'US_SHOW_FORMULAIC': [ # added by me 30 mar 2023 # AIM
                        'this @PRESENTATION_NOUN @PRESENTATION', # added by me, moved from REF_US_AGENT and added @PRESENTATION
                        'this @PRESENTATION_NOUN @INTEREST',
                        "the @CURRENT_ADJ @PRESENTATION_NOUN @PRESENTATION",
                        "the @CURRENT_ADJ @WORK_NOUN @PRESENTATION",
                        "this @CURRENT_ADJ @PRESENTATION_NOUN @PRESENTATION",
                        "this @CURRENT_ADJ @WORK_NOUN @PRESENTATION",
                        "in this @WORK_NOUN , @SELF_NOM @INTEREST",
                        "in this @WORK_NOUN , @SELF_NOM @PRESENTATION",
                        "in this @WORK_NOUN @SELF_NOM @INTEREST",
                        "in this @WORK_NOUN @SELF_NOM @PRESENTATION",
                        'in this @PRESENTATION_NOUN , @SELF_NOM @INTEREST', # moved from HERE_FORMULAIC bc jason wants this to belong to AIM
                        'in this @PRESENTATION_NOUN @SELF_NOM @INTEREST',
                        'in this @PRESENTATION_NOUN @SELF_NOM @PRESENTATION',
                        "in this @PRESENTATION_NOUN , @SELF_NOM @PRESENTATION" ,
                        #"in this @WORK_NOUN ,",
                        "we @PRESENTATION #DT", # added
                        "we @INTEREST #DT", # added
                        'this @WORK_NOUN @COMPARISON',
                        '@SELF_POSS @WORK_NOUN @COMPARISON',
                        'this @PRESENTATION_NOUN @COMPARISON',
                        '@SELF_POSS @PRESENTATION_NOUN @COMPARISON',
    ],
    'US_CONCLUDE_FORMULAIC': [ # OWN
                        # added 30 mar 2023
                        "the @CURRENT_ADJ @PRESENTATION_NOUN @SOLUTION",
                        "the @CURRENT_ADJ @WORK_NOUN @SOLUTION",
                        "this @CURRENT_ADJ @SOLUTION",
                        "@SELF_POSS @CURRENT_ADJ @PRESENTATION_NOUN @SOLUTION",
                        "@SELF_POSS @CURRENT_ADJ @WORK_NOUN @SOLUTION",
                        "@SELF_POSS @PRESENTATION_NOUN @SOLUTION",
                        "@SELF_POSS @WORK_NOUN @SOLUTION",
                        "@SELF_NOM @SOLUTION",
                        # "@SELF_POSS @CURRENT_ADJ @WORK_NOUN #VV", # might be noisy
    ],

    'US_PREVIOUS_FORMULAIC': [ # BAS
                                #'@SELF_NOM have previously',  # could also be OWN (when taking abt method)
                               #'@SELF_NOM have early', # previously ... earlier
                               '@SELF_NOM have elsewhere',
                               # '@SELF_NOM elsewhere',
                            #    '@SELF_NOM previously',
                            #    '@SELF_NOM earlier',
                            #    'elsewhere @SELF_NOM', # needs to be BOS to work otherwise noisy
                            #    'elswhere @SELF_NOM',
                            #    'elsewhere , @SELF_NOM',
                            #    'elswhere , @SELF_NOM',
                               'presented elsewhere',
                               '@SELF_NOM have @ARGUMENTATION elsewhere',
                               '@SELF_NOM have @SOLUTION elsewhere',
                               '@SELF_NOM will show elsewhere',
                               '@SELF_NOM will argue elsewhere',
                               'elsewhere SELFCITATION',
                               'in a @BEFORE_ADJ @PRESENTATION_NOUN ,',
                               'in an @BEFORE_ADJ @PRESENTATION_NOUN ,',
                               'another @PRESENTATION_NOUN' ],
                               
    'TEXTSTRUCTURE_FORMULAIC': [ 'then @SELF_NOM describe',  # TXT
                                 'then , @SELF_NOM describe',
                                 'next @SELF_NOM describe',
                                 'next , @SELF_NOM describe',
                                 'finally @SELF_NOM describe',
                                 'finally , @SELF_NOM describe',
                                 'then @SELF_NOM present',
                                 'then , @SELF_NOM present',
                                 'next @SELF_NOM present',
                                 'next , @SELF_NOM present',
                                 'finally @SELF_NOM present',
                                 'finally , @SELF_NOM present',
                                 'briefly describe',
                                 'briefly introduce',
                                 'briefly present',
                                 'briefly discuss' ],

    'HERE_FORMULAIC': [  # OWN
                        'the present @PRESENTATION_NOUN',
                        '@SELF_NOM here #VV', # added #VV
                        'here @SELF_NOM #VV',
                        'here , @SELF_NOM #VV',
                        'the @WORK_NOUN @GIVEN here', # modified; orig: @GIVEN here
                        '@SELF_NOM now #VV',
                        'now @SELF_NOM #VV',
                        'now , @SELF_NOM #VV',
                        '@SELF_NOM @GIVEN now',
                        'herein' ],

    'NOVEL_FORMULAIC': [ # AIM?
                        'a @NEW_ADJ @WORK_NOUN',
                        'an @NEW_ADJ @WORK_NOUN',  
                        '@SELF_NOM @PRESENTATION a @WORK_NOUN of',
                        '@SELF_NOM @PRESENTATION an @WORK_NOUN of',
                        
                        # below: moved from METHOD_FORMULAIC
                        'the problem of #RB #VV',
                        'the problem of #VV', # diff
                        'the problem of how to',
    ],

    "DESCRIPTION_FORMULAIC":[ # BKG
                        'be a @TRADITION_ADJ @WORK_NOUN',
                        'be an @TRADITION_ADJ @WORK_NOUN',

                        # below: moved from EXTENDING_WORK2_FORMULAIC
                        '#NN be #DD @CHANGE_NOUN of',
                        '#NN be #DD #JJ @CHANGE_NOUN of',
                        '#DD #NN @DENOTATION #DD @CHANGE_NOUN of',
                        '@TEXT_NOUN @CONTINUE CITATION',
                        '#NN @CONTINUE #NN of CITATION',
                        'be @SEE as an @CHANGE_NOUN',
                        '@CHANGE #DD #NN of CITATION',
                        'it be @PRIVATE_VERBS to #VV',
                        'it be @PRIVATE_VERBS that',
                        'one way to',
                        'have be @PRIVATE_VERBS',
                        "have @CURRENT_ADV be @USE",
                        "there have be",
                        "there have #RB be",
                        "have @RESEARCH",
                        "have @USE",
                        "have @NEED",
                        "have @INCREASE",
                        "have become",
                        "have be #RB #VV",
                        "have be a #JJ @WORK_NOUN",
                        "have be a #JJ research topic",
                        "can be #VV",
                        "can #RB be #VV",
                        "be usually #VV",
                        "be usually a",
                        "be usually an"
                        "hold potential for",

                        # below: from Glasman-Deal
                        '#NN is a #NN', # GD
                        'one way to #VV'
    ],

    'METHOD_FORMULAIC': [ # OWN
                          '@SOLUTION a #JJ @WORK_NOUN of',
                          '@SOLUTION an #JJ @WORK_NOUN of',
                          '@SOLUTION a #NN @WORK_NOUN of',
                          '@SOLUTION an #NN @WORK_NOUN of',
                          '@SOLUTION a #JJ #NN @WORK_NOUN of',
                          '@SOLUTION an #JJ #NN @WORK_NOUN of',
                          '@SOLUTION a @WORK_NOUN for',
                          '@SOLUTION an @WORK_NOUN for',
                          '@SOLUTION a #JJ @WORK_NOUN for',
                          '@SOLUTION an #JJ @WORK_NOUN for',
                          '@SOLUTION a #NN @WORK_NOUN for',
                          '@SOLUTION an #NN @WORK_NOUN for',
                          '@SOLUTION a #JJ #NN @WORK_NOUN for',
                          '@SOLUTION an #JJ #NN @WORK_NOUN for',
                          '@WORK_NOUN design to #VV', #diff
                          '@WORK_NOUN intend for',
                          '@WORK_NOUN for #VV',
                          # '@WORK_NOUN for the #NN', # noisy
                          '@WORK_NOUN design to #VV', # diff
                          # '@WORK_NOUN to the #NN', # noisys
                          # '@WORK_NOUN to #NN', # noisy
                          #'@WORK_NOUN to #VV',
                          #'@WORK_NOUN for #JJ #VV', # diff
                          # '@WORK_NOUN for the #JJ #NN',
                          #'@WORK_NOUN to the #JJ #NN',
                          #'@WORK_NOUN to #JJ #VV',
                          "@RESULT_NOUN be @COMPARISON",
                          "@SELF_NOM #RB #VV",
                          "#NN be @USE",
                          "#NN be @USE #IN"
                          # "would benefit" #??
                          ], 

    'CONTINUE_FORMULAIC': [ 'follow CITATION', # BAS
                            'follow the @WORK_NOUN of CITATION',
                            'follow the @WORK_NOUN give in CITATION',
                            'follow the @WORK_NOUN present in CITATION',
                            'follow the @WORK_NOUN propose in CITATION',
                            'follow the @WORK_NOUN discuss in CITATION',
                            'base on CITATION',
                            '@CONTINUE CITATION',
                            '@CONTINUE the @WORK_NOUN',
                            '@CONTINUE a @WORK_NOUN',
                            '@CONTINUE an @WORK_NOUN',
                            '@CONTINUE @OTHERS_POSS @WORK_NOUN',
                            '@CONTINUE @SELF_POSS @WORK_NOUN',
                            '@AGREE CITATION',
                            '@AGREE the @WORK_NOUN',
                            '@AGREE a @WORK_NOUN',
                            '@AGREE an @WORK_NOUN',
                            '@AGREE @OTHERS_POSS @WORK_NOUN',
                            '@AGREE @SELF_POSS @WORK_NOUN',
                            'base on the @WORK_NOUN of CITATION',
                            'base on the @WORK_NOUN give in CITATION',
                            'base on the @WORK_NOUN present in CITATION',
                            'base on the @WORK_NOUN propose in CITATION',
                            'base on the @WORK_NOUN discuss in CITATION',
                            'adopt CITATION',
                            'start point for @REFERENTIAL @WORK_NOUN',
                            'start point for @SELF_POSS @WORK_NOUN',
                            'as a start point',
                            'as start point',
                            # 'use CITATION', # diff # too short
                            # 'base @SELF_POSS',
                            # below: modified (add @ARGUMENTATION_NOUN)
                            'support @SELF_POSS @ARGUMENTATION_NOUN',
                            'support @OTHERS_POSS @ARGUMENTATION_NOUN',
                            'lend support to @SELF_POSS @ARGUMENTATION_NOUN',
                            'lend support to @OTHERS_POSS @ARGUMENTATION_NOUN',
                            # new
                            '@CONTINUE the @WORK_NOUN of',
                            '@AGREE the @WORK_NOUN of'
                            ],

    'DISCOURSE_CONTRAST_FORMULAIC': [ # CTR; not using
                                      #'however', # too noisy
                                      'nevertheless', #
                                      'nonetheless', #
                                      'unfortunately', 
                                      'yet', #
                                      'although',  #
                                      'whereas' ,
                                      "albeit"
                                      ],
    
    "GAP_FORMULAIC": [ # BKG? OWN?
    # not included in Jurgens et al 2018. has to only be used in subject positions (see: p. 198 of teufel 2000)
                            "as far as @SELF_NOM know",
                            "to @SELF_NOM knowledge", # diff
                            "to the best of @SELF_NOM knowledge",
                            "to our knowledge",
                            'little attention have be pay', # IYWS-MEDAL
                            "have receive #JJ attention", # IYWS-MEDAL
                            "have be @USE #IN", # IYWS-MEDAL
                            "have yet to be",
                            "have not be #VV"
                        ],

    "FUTURE_FORMULAIC": [  #OWN?
    # in Jurgens et al 2018?
                            "in the future",
                            "in the near future",
                            "@FUTURE_ADJ @WORK_NOUN",
                            "@FUTURE_ADJ @AIM_NOUN",
                            "@FUTURE_ADJ development",
                            "need further", # diff
                            "require further",
                            "beyond the scope",
                            "avenue for improvement",
                            "avenue for @FUTURE_ADJ improvement",
                            "avenue for @FUTURE_ADJ research",
                            "area for @FUTURE_ADJ improvement",
                            "area for improvement",
                            "promising avenue"
    ],

    'SIMILARITY_FORMULAIC': [ # OTH
    # not in jurgens 2018
                        "along the same lines",
                        "in a similar vein",
                        "as in @SELF_POSS @WORK_NOUN",
                        "as in @SELF_POSS #JJ @WORK_NOUN",

                        "as in CITATION",
                        "as did CITATION",
                        "like in CITATION",
                        "like CITATION ' s",
                        
                        "@SIMILAR_ADJ_PHRASE CITATION",
                        "@SIMILAR_ADJ_PHRASE @SELF_POSS @WORK_NOUN",
                        "@SIMILAR_ADJ_PHRASE @SELF_POSS #JJ @WORK_NOUN",
                        "@SIMILAR_ADJ_PHRASE @OTHERS_POSS @WORK_NOUN",
                        "@SIMILAR_ADJ_PHRASE @OTHERS_POSS #JJ @WORK_NOUN",
                        "@SIMILAR_ADJ_PHRASE @TRADITION_ADJ @WORK_NOUN",
                        "@SIMILAR_ADJ_PHRASE @BEFORE_ADJ @WORK_NOUN",

                        "similarity with @MANY @WORK_NOUN", # diff
                        
                        "similar to @SELF_ACC",
                        "similar to that describe here",
                        "similar to that of",
                        "similar to those of",
                        "similar to @MANY @WORK_NOUN",

                        "a similar #NN to CITATION",
                        "a similar #NN to @OTHERS_POSS @WORK_NOUN",
                        "a similar #NN to @SELF_POSS @WORK_NOUN",

                        "analogous to @MANY @WORK_NOUN",
                        "analogous to @OTHERS_ACC",
                        "analogous to that described here",
                        "analogous to @SELF_ACC",

                        "the same #NN as @OTHERS_POSS @WORK_NOUN",
                        "the same #NN as @OTHERS_POSS #JJ @WORK_NOUN",
                        "the same #NN as CITATION",

                        "in common with @MANY @WORK_NOUN",
                        
                        "most relevant to @SELF_POSS @WORK_NOUN",
                        "most relevant to @SELF_POSS #JJ @WORK_NOUN",
    ],

    'GRAPHIC_FORMULAIC': [ #'@GRAPHIC_NOUN #CD'# TXT
                          "see @GRAPHIC_NOUN #CD",
                          "as @PRESENTATION in @GRAPHIC_NOUN #CD",
                          "be @PRESENTATION in @GRAPHIC_NOUN #CD",
                          '@GRAPHIC_NOUN #CD @PRESENTATION',
                          "include in @GRAPHIC_NOUN #CD"
                         ], 

    'CONTRAST2_FORMULAIC': [ 'this @WORK_NOUN @CONTRAST', # CTR
                            '@SELF_POSS @WORK_NOUN @CONTRAST',
                            'this @PRESENTATION_NOUN @CONTRAST',
                            '@SELF_POSS @PRESENTATION_NOUN @CONTRAST',
                            'compare to @OTHERS_POSS @WORK_NOUN',
                            'compare to @OTHERS_POSS @PRESENTATION_NOUN',
                            '@OTHERS_POSS @WORK_NOUN @CONTRAST',
                            'that @WORK_NOUN @CONTRAST',
                            'that @PRESENTATION_NOUN @CONTRAST',
                            '@OTHERS_POSS @PRESENTATION_NOUN @CONTRAST',
                            "in contrast"
                            ], # lacks support formulaic, e.g. our work (adv) supports...

    'COMPARISON_FORMULAIC': [ # OTH
                              '@GIVEN #NN to @SIMILAR', # [note/mention/address/discuss] #NN [is similar to]
                              '@SELF_POSS #NN @SIMILAR', # [our, my] #NN [is similar to]
                              '@SELF_POSS @PRESENTATION_NOUN @SIMILAR', # originally ... @PRESENTATION ...
                              # 'a @SELF_POSS @PRESENTATION @SIMILAR', 
                              'this @WORK_NOUN @PRESENTATION @SIMILAR', 
                              'the @WORK_NOUN @PRESENTATION @SIMILAR', 
                              'a @SIMILAR_ADJ @WORK_NOUN be',
                              'be closely relate to',
                              #'be @SIMILAR_ADJ to', # too general
                              'along the line of CITATION',
                              ],
    
    'CONTRAST_FORMULAIC': [ 'in contrast with', # CTR
                              'in comparison to',
                              
                              # OTHERS_PREP: while, against, unlike, whereas
                              '@OTHERS_PREP @OTHERS_ACC',  # "her", "him", "them"
                              '@OTHERS_PREP @OTHERS_POSS @WORK_NOUN',  # "their", "his", "her"
                              '@OTHERS_PREP @MANY @WORK_NOUN',
                              '@OTHERS_PREP @TRADITION_ADJ @WORK_NOUN',
                              '@OTHERS_PREP @BEFORE_ADJ @WORK_NOUN',
                              '@OTHERS_PREP @1ST_PERSON_PRONOUN_(NOM)', # "ours", "mine"
                              '@OTHERS_PREP @SELF_POSS #NN', # "my", "our"
                              #'@OTHERS_PREP @SELF_ACC', # "us", "me"
                              '@OTHERS_PREP CITATION',

                              '#JJ than CITATION',
                              #'#JJ than @SELF_ACC', # added #JJ
                              '#JJ than @SELF_POSS', # added #JJ
                              '#JJ than @OTHERS_ACC', # added #JJ
                              '#JJ than @OTHERS_POSS', # added #JJ
                              '#JJ than @TRADITION_ADJ @WORK_NOUN', # added #JJ
                              '#JJ than @BEFORE_ADJ @WORK_NOUN', # added #JJ
                              '#JJ than @MANY @WORK_NOUN', # added #JJ
                              'point of departure from @SELF_POSS',
                              'points of departure from @OTHERS_POSS',
                              'points of departure from CITATION',

                              '@WIN_PHRASE @OTHERS_ACC',
                              '@WIN_PHRASE @TRADITION_ADJ',
                              '@WIN_PHRASE @MANY @WORK_NOUN',
                              '@WIN_PHRASE @BEFORE_ADJ @WORK_NOUN',
                              '@WIN_PHRASE @OTHERS_POSS',
                              '@WIN_PHRASE CITATION',

                              "advantage of @SELF_ACC",
                              "advantage of @SELF_POSS",
                              "advantage of @1ST_PERSON_PRONOUN_(NOM)",

                              '@CONTRAST_ADJ_PHRASE CITATION',
                              '@CONTRAST_ADJ_PHRASE @TRADITION_ADJ #NN',
                              '@CONTRAST_ADJ_PHRASE @MANY @WORK_NOUN',
                              '@CONTRAST_ADJ_PHRASE @BEFORE_ADJ @WORK_NOUN',
                              '@CONTRAST_ADJ_PHRASE @OTHERS_ACC', # "her", "him", "them"
                              '@CONTRAST_ADJ_PHRASE @OTHERS_POSS', # "their", "his", "her"
                              '@CONTRAST_ADJ_PHRASE @SELF_ACC', # "us", "me"
                              '@CONTRAST_ADJ_PHRASE @SELF_POSS', # "my", "our"
                              '@CONTRAST_ADJ_PHRASE @1ST_PERSON_PRONOUN_(NOM)', # "ours", "mine"

                              'compare to @OTHERS_POSS @WORK_NOUN',
                              'compare to @OTHERS_POSS @PRESENTATION_NOUN',
                              
                              ],
    'IMPROVE_FORMULAIC': [ # added 2 apr 2023 by me; OWN? (CTR?)
                          'significantly @BETTER_SOLUTION',
                          'a significant improvement of',
                          'be significantly #JJR',
                          "exhibit significantly #JJR"
                        ],

    'ALIGN_FORMULAIC': ['in the @SENSE_NOUN of CITATION'], # OWN => OTH

    'AFFECT_FORMULAIC': [ 'hopefully', 'thankfully', 'fortunately', 'unfortunately' ], # ??

    'GOOD_FORMULAIC': [ '@GOOD_ADJ' ], # OWN, BAS
    #'BAD_FORMULAIC': [ '@BAD_ADJ' ],
    'TRADITION_FORMULAIC': [ '@TRADITION_ADJ' ], # x
    'IN_ORDER_TO_FORMULAIC': [ 'in order to' ], # AIM => OWN? OTH?

    'DETAIL_FORMULAIC': ['@SELF_NOM have also #VV', # OWN
                         '@SELF_NOM also #VV',
                         'this @PRESENTATION_NOUN also #VV',
                         'this @PRESENTATION_NOUN has also #VV',
                          "specifically , @SELF_NOM", # added by me 31 mar 2023
                          "in specific , @SELF_NOM", # added by me 31 mar 2023
                            ],
    
    'NO_TEXTSTRUCTURE_FORMULAIC': [ # '( @TEXT_NOUN CREF )',  # TXT
                                    'as explain in @TEXT_NOUN CREF',
                                    'as explain in the @BEFORE_ADJ @TEXT_NOUN',
                                    'as @GIVEN early in this @TEXT_NOUN',
                                    'as @GIVEN below',
                                    'as @GIVEN in @TEXT_NOUN CREF',
                                    'as @GIVEN in the @BEFORE_ADJ @TEXT_NOUN',
                                    'as @GIVEN in the next @TEXT_NOUN',
                                    '#NN @GIVEN in @TEXT_NOUN CREF',
                                    '#NN @GIVEN in the @BEFORE_ADJ @TEXT_NOUN',
                                    '#NN @GIVEN in the next @TEXT_NOUN',
                                    '#NN @GIVEN below',
                                    'cf. @TEXT_NOUN CREF',
                                    'cf. @TEXT_NOUN below',
                                    'cf. the @TEXT_NOUN below',
                                    'cf. the @BEFORE_ADJ @TEXT_NOUN',
                                    'cf. @TEXT_NOUN above',
                                    'cf. the @TEXT_NOUN above',
                                    'cfXXX @TEXT_NOUN CREF', # cfXXX ??
                                    'cfXXX @TEXT_NOUN below',
                                    'cfXXX the @TEXT_NOUN below',
                                    'cfXXX the @BEFORE_ADJ @TEXT_NOUN',
                                    'cfXXX @TEXT_NOUN above',
                                    'cfXXX the @TEXT_NOUN above',
                                    'e. g. , @TEXT_NOUN CREF',
                                    'e. g , @TEXT_NOUN CREF',
                                    'e. g. @TEXT_NOUN CREF',
                                    'e. g @TEXT_NOUN CREF',
                                    'e.g., @TEXT_NOUN CREF',
                                    'e.g. @TEXT_NOUN CREF',
                                    'compare @TEXT_NOUN CREF',
                                    'compare @TEXT_NOUN below',
                                    'compare the @TEXT_NOUN below',
                                    'compare the @BEFORE_ADJ @TEXT_NOUN', 
                                    'compare @TEXT_NOUN above',
                                    'compare the @TEXT_NOUN above',
                                    'see @TEXT_NOUN CREF',
                                    'see the @BEFORE_ADJ @TEXT_NOUN',
                                    'recall from the @BEFORE_ADJ @TEXT_NOUN',
                                    'recall from the @TEXT_NOUN above',
                                    'recall from @TEXT_NOUN CREF',
                                    '@SELF_NOM shall see below',
                                    '@SELF_NOM will see below',
                                    '@SELF_NOM shall see in the next @TEXT_NOUN',
                                    '@SELF_NOM will see in the next @TEXT_NOUN',
                                    '@SELF_NOM shall see in @TEXT_NOUN CREF',
                                    '@SELF_NOM will see in @TEXT_NOUN CREF',
                                    'example in @TEXT_NOUN CREF',
                                    'example CREF in @TEXT_NOUN CREF',
                                    'example CREF and CREF in @TEXT_NOUN CREF',
                                    'example in @TEXT_NOUN CREF' ],

    'USE_FORMULAIC': [ '@SELF_NOM @USE', # OWN #, AIM?
                       #'@WORK_NOUN @USE',
                       '@SELF_NOM @RESEARCH',
                       #'be @USE to',
                       #'can be #VV use', #can be /solved/ using
                       #'@SELF_POSS @WORK_NOUN be @CONTINUE',
                       #'@SELF_POSS #JJ @WORK_NOUN be @CONTINUE',
                       '@SOLUTION with the @HELP_NOUN of',
                       '@SOLUTION with the @WORK_NOUN of',
                       "this @WORK_NOUN @USE"
                       ],

    'FUTURE_WORK_FORMULAIC': [ '@FUTURE_ADJ @WORK_NOUN',   # OWN
                               '@FUTURE_ADJ @AIM_NOUN',
                               '@FUTURE_ADJ @CHANGE_NOUN',
                               'a @POTENTIAL_ADJ @AIM_NOUN',
                               'one @POTENTIAL_ADJ @AIM_NOUN',
                               '#NN be also @POTENTIAL_ADJ',
                               'in the future',
                               '@SELF_NOM @FUTURE_INTEREST',
                               ], # teufel doesnt include patterns with @INTEREST lexicon
                               # which might miss moves like 'we aim to...'

    'HEDGING_FORMULAIC': [ '@HEDGING_MODALS be @RESEARCH',  # OWN? # too general, removed
                           '@HEDGING_MODALS be @CHANGE',
                           '@HEDGING_MODALS be @SOLUTION',
                           ], # "hedging is discouraged in extraction" teufel 2000 5.2.2.1

    'GD_SELF_PROBLEM': ["it be recognise that", 
                            "it be recognize that",
                            "not perfect",
                            "not identical",
                            "slightly problematic",
                            "nigligible",
                            "a preliminary attempt",
                            "necessarily",
                            "impractical",
                            "be hard to",
                            "be difficult to",
                            "unavoidable",
                            "impossible",
                            "reasonably #JJ",
                            "currently in progress",
                            "currently underway",], # from pp. 86-87

    'PRESENT_WORK_FORMULAIC': [ '@SELF_NOM be @CURRENT_ADV @RESEARCH',  # OWN
                                '@SELF_NOM be @RESEARCH @CURRENT_ADV',
                                "this @PRESENTATION_NOUN be support by", # added by me
                                ],

    'EXTENDING_WORK_FORMULAIC': [ '@CHANGE the @WORK_NOUN', # OWN
                                  '@CHANGE this @WORK_NOUN',
                                  '@SELF_POSS @WORK_NOUN be @CHANGE',
                                  '@SELF_POSS #JJ @WORK_NOUN be @CHANGE',
                                  '@SELF_POSS @WORK_NOUN @CHANGE',
                                  '@SELF_POSS #JJ @WORK_NOUN @CHANGE',
                                  '@CHANGE the #JJ @WORK_NOUN',
                                  '@SELF_NOM @CHANGE'
                                  ],

    'EXTENDING_WORK2_FORMULAIC': [ '@SELF_NOM @CHANGE #DD @WORK_NOUN',  # OWN
                                   '@SELF_POSS @WORK_NOUN @CHANGE',
                                   '@CHANGE from CITATION',
                                   '@CHANGE from #NN of CITATION',
                                   '@SELF_POSS @CHANGE_NOUN of CITATION',
                                   '@SELF_POSS @WORK_NOUN @CONTINUE',
                                   '@SELF_POSS @WORK_NOUN be #DD @CHANGE_NOUN',
                                   '@SELF_POSS @WORK_NOUN be #VV #DD @CHANGE_NOUN',
                                   
                                  ],

    'USEFUL_FORMULAIC': [ 'have show @GOOD_ADJ for', #BKG
                          'have @MANY @POTENTIAL_ADJ use',
                          'have @MANY @POTENTIAL_ADJ use in',
                          'have @MANY application in',
                          'be @USE in @MANY #NN',
                          'be @USE to #VV',
                          'be #RB @USE to #VV',
                          "have be @USE to #VV", # GD
                          "have be #RB @USE to #VV" #GD
                        ], 

    'MOTIVATING_FORMULAIC': [ 'as @PRESENTATION in CITATION', # BAS
                              'as @PRESENTATION by CITATION',
                              'this be a #JJ convention',
                              'this be a #RB #JJ convention',
                              '@CONTINUE the #NN result',
                              '@CONTINUE the #JJ result',
                              '@CONTINUE CITATION, @SELF_NOM',
                              '@AGREE the #NN result',
                              '@AGREE the #JJ result',
                              #'@INSPRATION by the #NN result',
                              #'@INSPIRATION by the #JJ result',
                              '@INSPIRATION by',
                              'from CITATION , @SELF_NOM',
                              #'#NN be @MAIN_ADJ in',
                              #'#NN be @MAIN_ADJ for',
                              'it be @MAIN_ADJ not to', 
                              '@AGREE CITATION, @SELF_NOM',
                              # 'have be @PRESENTATION',  # SUPER NOISY :( 
                              # '#NN need to @USE', ??
                              
                              ],

    "NEGATIVE_RESULT_FORMULAIC": [ # BKG
                                  'negative @RESULT_NOUN for',
                                  'negative @RESULT_NOUN that',
                                  '@FEW_ADJ @RESULT_NOUN for', # added by me
                                  '@FEW_ADJ @RESULT_NOUN that', # added by me
    ],
    "PRIOR_WORK_FORMULAIC": [ '@BEFORE_ADJ @PRESENTATION_NOUN @SELF_NOM', # BAS, OTH
                              '@BEFORE_ADJ @PRESENTATION_NOUN , @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION_NOUN @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION_NOUN , @SELF_NOM',
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN @SELF_NOM',
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN , @SELF_NOM',
                              '@BEFORE_ADJ @PRESENTATION_NOUN CITATION @SELF_NOM',
                              '@BEFORE_ADJ @PRESENTATION_NOUN CITATION , @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION_NOUN CITATION @SELF_NOM',
                              'a @BEFORE_ADJ @PRESENTATION_NOUN CITATION , @SELF_NOM',
                              
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN CITATION',
                              '@SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN SELFCITATION',
                              'first @PRESENTATION_NOUN in CITATION',
                              '@PRESENTATION_NOUN #RB in CITATION', # originally RR, prob should be RB
                              '@PRESENTATION_NOUN #JJ in CITATION',
                              '@BEFORE_ADJ @CHANGE_NOUN of @SELF_POSS @WORK_NOUN',
                              '@CHANGE on @BEFORE_ADJ @PRESENTATION_NOUN in SELFCITATION',
                              '@CHANGE @BEFORE_ADJ @PRESENTATION_NOUN in SELFCITATION',
                              '@CHANGE @BEFORE_ADJ @PRESENTATION_NOUN SELFCITATION',
                              '@CHANGE on @SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN in SELFCITATION',
                              '@CHANGE @SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN in SELFCITATION',
                              '@CHANGE @SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN SELFCITATION',
                              'in @SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN CITATION',

                              # below: moved from motivating_formulaic 30 mar 2023
                              'CITATION have @RESEARCH it',
                              'CITATION have @PRESENTATION that',
                              'CITATION @PRESENTATION that',
                              'CITATION #RB @PRESENTATION that', # RB = Adverb
                              "have @PUBLIC_VERBS that",
    ],
    "BACKGROUND_FORMULAIC": [ # BKG ; new pattern
                              # below: moved from motivating_formulaic 30 mar 2023
                              'have remain a @PROBLEM_NOUN', 
                              'their importance have @INCREASE',
                              '@RESEARCH in @DISCIPLINE @PRESENTATION',
                              '@RESEARCH in #NN @PRESENTATION',
                              '@RESEARCH in #NN #NN @PRESENTATION',
                              '@RESEARCH in #JJ #NN @PRESENTATION',
                              'it be well document',
                              'it have be well document',
                              'prove to be @GOOD_ADJ in',
                              '@PRESENTATION to be @GOOD_ADJ in',
                              'prove to be @GOOD_ADJ for',
                              '@PRESENTATION to be @GOOD_ADJ for',
                              "@PUBLIC_VERBS by the literature",  # added by me 31mar2023
                              "prove @GOOD_ADJ",
                              #"@SUASIVE_VERBS to"
                              ],

    }

    # upward arrows in teufel 2000 denotes a trigger word:
    # Pattern matching procedures on such a large scale are slow. We reduce the number of 
    # comparisons necessary with a trigger mechanism: only to those sentences containing
    # a trigger (a rare word which covers as many patterns as possible) are searched, and 
    # they are searched only for those patterns which do contain the trigger. Triggers are 
    # marked by the signal [upward arrow] directly in the pattern.
                              


AGENT_PATTERNS = {

    'US_AGENT': [ #'@SELF_NOM', # not used (possiblly #OWN or #AIM but very noisy)
                  '@SELF_POSS #JJ @WORK_NOUN',
                  '@SELF_POSS #JJ @PRESENTATION_NOUN',
                  '@SELF_POSS #JJ @ARGUMENTATION_NOUN',
                  '@SELF_POSS #JJ @SOLUTION_NOUN',
                  '@SELF_POSS #JJ @RESULT_NOUN',
                  '@SELF_POSS @WORK_NOUN',
                  '@SELF_POSS @PRESENTATION_NOUN',
                  '@SELF_POSS @ARGUMENTATION_NOUN',
                  '@SELF_POSS @SOLUTION_NOUN',
                  '@SELF_POSS @RESULT_NOUN',
                  '@WORK_NOUN @GIVEN here',
                  '@WORK_NOUN @GIVEN below',
                  '@WORK_NOUN @GIVEN in this @PRESENTATION_NOUN',
                  '@WORK_NOUN @GIVEN in @SELF_POSS @PRESENTATION_NOUN',
                  'the @SOLUTION_NOUN @GIVEN here',
                  'the @SOLUTION_NOUN @GIVEN in this @PRESENTATION_NOUN',
                  'the first author',
                  'the second author',
                  'the third author',
                  #'one of the author',
                  'one of us' ],

    'REF_US_AGENT': [  # OWN
                      'the @CURRENT_ADJ @PRESENTATION_NOUN',
                      'the @CURRENT_ADJ #JJ @PRESENTATION_NOUN',
                      'the @WORK_NOUN @GIVEN',
                      'the current #JJ @WORK_NOUN', # moved from REF_AGENT
                      'the current @WORK_NOUN', # moved from REF_AGENT
                      ],

    'OUR_AIM_AGENT': [ '@SELF_POSS @AIM_NOUN', # AIM
                       'the point of this @PRESENTATION_NOUN',
                       'the @AIM_NOUN of this @PRESENTATION_NOUN',
                       'the @AIM_NOUN of the @GIVEN @WORK_NOUN',
                       'the @AIM_NOUN of @SELF_POSS @WORK_NOUN',
                       'the @AIM_NOUN of @SELF_POSS @PRESENTATION_NOUN',
                       'the most @MAIN_ADJ feature of @SELF_POSS @WORK_NOUN',
                       'contribution of this @PRESENTATION_NOUN',
                       'contribution of the @GIVEN @WORK_NOUN',
                       'contribution of @SELF_POSS @WORK_NOUN',
                       'the question @GIVEN in this @PRESENTATION_NOUN',
                       'the question @GIVEN here',
                       '@SELF_POSS @MAIN_ADJ @AIM_NOUN',
                       '@SELF_POSS @AIM_NOUN in this @PRESENTATION_NOUN',
                       '@SELF_POSS @AIM_NOUN here',
                       'the #JJ point of this @PRESENTATION_NOUN',
                       'the #JJ purpose of this @PRESENTATION_NOUN',
                       'the #JJ @AIM_NOUN of this @PRESENTATION_NOUN',
                       'the #JJ @AIM_NOUN of the @GIVEN @WORK_NOUN',
                       'the #JJ @AIM_NOUN of @SELF_POSS @WORK_NOUN',
                       'the #JJ @AIM_NOUN of @SELF_POSS @PRESENTATION_NOUN',
                       'the #JJ question @GIVEN in this @PRESENTATION_NOUN',
                       'the #JJ question @GIVEN here' ],

    'AIM_REF_AGENT':  [ '@SOLUTION_NOUN of this @WORK_NOUN', # AIM
                       "the @MAIN_ADJ @QUESTION_NOUN of this @WORK_NOUN", # added by me
                        'the most important @SOLUTION_NOUN of this @WORK_NOUN',
                        # 'the @AIM_NOUN', # noisy
                        #'the #JJ @AIM_NOUN'  # noisy
                        ],
                        
    'US_PREVIOUS_AGENT': [ # 'SELFCITATION', # OTH
                           'this @BEFORE_ADJ @PRESENTATION_NOUN',
                           '@SELF_POSS @BEFORE_ADJ @PRESENTATION_NOUN',
                           '@SELF_POSS @BEFORE_ADJ @WORK_NOUN',
                           'in CITATION , @SELF_NOM',
                           'in CITATION @SELF_NOM',
                           'the @WORK_NOUN @GIVEN in SELFCITATION',
                           'in @BEFORE_ADJ @PRESENTATION CITATION @SELF_NOM',
                           'in @BEFORE_ADJ @PRESENTATION CITATION , @SELF_NOM',
                           'in a @BEFORE_ADJ @PRESENTATION CITATION @SELF_NOM',
                           'in a @BEFORE_ADJ @PRESENTATION CITATION , @SELF_NOM',
                           ],

    'REF_AGENT': [ # some removed for being too general #OTH 
                  # '@REFERENTIAL #JJ @WORK_NOUN', 
                  # '@REFERENTIAL @WORK_NOUN' # only plural referentials apply
                  'those @WORK_NOUN',
                  'those #JJ @WORK_NOUN',
                   'these @WORK_NOUN',
                   'these #JJ @WORK_NOUN',
                   'this sort of @WORK_NOUN',
                   'this kind of @WORK_NOUN',
                   'this type of @WORK_NOUN',
                   
                   #'the @WORK_NOUN', 
                   #'the @PRESENTATION_NOUN',
                   #'the author',
                   #'the authors' 
                   ],

    # 'THEM_PRONOUN_AGENT': [ '@OTHERS_NOM' ], # OTH? Might be super noisy

    'THEM_ACTIVE_AGENT' : [ 'CITATION @PRESENTATION', # OTH
                           # below: moved from AIM_REF_AGENT
                            '@OTHERS_POSS @AIM_NOUN', 
                            '@OTHERS_POSS #JJ @AIM_NOUN',
                            #'@REFERENTIAL #JJ @AIM_NOUN',
                            'the #JJ @AIM_NOUN', 
                           ] , 

    'THEM_AGENT': [ # OTH? Might also be noisy? => moved to # BKG
                    'CITATION \'s #NN',  
                    'CITATION \'s @PRESENTATION_NOUN',
                    'CITATION \'s @WORK_NOUN',
                    'CITATION \'s @ARGUMENTATION_NOUN',
                    'CITATION \'s #JJ @PRESENTATION_NOUN',
                    'CITATION \'s #JJ @WORK_NOUN',
                    'CITATION \'s #JJ @ARGUMENTATION_NOUN',
                    'the CITATION @WORK_NOUN',
                    'the @WORK_NOUN @GIVEN in CITATION',
                    'the @WORK_NOUN of CITATION',
                    '@OTHERS_POSS @PRESENTATION_NOUN',
                    '@OTHERS_POSS @WORK_NOUN',
                    '@OTHERS_POSS @RESULT_NOUN',
                    '@OTHERS_POSS @ARGUMENTATION_NOUN',
                    '@OTHERS_POSS @SOLUTION_NOUN',
                    '@OTHERS_POSS #JJ @PRESENTATION_NOUN',
                    '@OTHERS_POSS #JJ @WORK_NOUN',
                    '@OTHERS_POSS #JJ @RESULT_NOUN',
                    '@OTHERS_POSS #JJ @ARGUMENTATION_NOUN',
                    '@OTHERS_POSS #JJ @SOLUTION_NOUN',
                    '#NNP and #NNP',
                    ],

    'GAP_AGENT':  [ 'none of these @WORK_NOUN', # BKG
                    'none of those @WORK_NOUN',
                    'no @WORK_NOUN',
                    'no #JJ @WORK_NOUN',
                    'none of these @PRESENTATION_NOUN',
                    'none of those @PRESENTATION_NOUN',
                    'no @PRESENTATION_NOUN',
                    'no #JJ @PRESENTATION_NOUN',
                      ],

    'GENERAL_AGENT': [ '@TRADITION_ADJ #JJ @WORK_NOUN', # BKG => OTH
                       '@TRADITION_ADJ use @WORK_NOUN',
                       '@TRADITION_ADJ @WORK_NOUN',
                       '@MANY @BEFORE_ADJ @WORK_NOUN', # #JJ => @BEFORE_ADJ
                       '@MANY @WORK_NOUN',
                       '@BEFORE_ADJ #JJ @WORK_NOUN',
                       '@BEFORE_ADJ @WORK_NOUN',
                       '@BEFORE_ADJ #JJ @PRESENTATION_NOUN',
                       '@BEFORE_ADJ @PRESENTATION_NOUN',
                       'other #JJ @WORK_NOUN',
                       'other @WORK_NOUN',
                       'such @WORK_NOUN',
                       'these #JJ @PRESENTATION_NOUN',
                       'these @PRESENTATION_NOUN',
                       'those #JJ @PRESENTATION_NOUN',
                       'those @PRESENTATION_NOUN',
                       '@REFERENTIAL authors',
                       '@MANY author',
                       'researcher in @DISCIPLINE',
                       #'@PROFESSIONALS' 
                       ],

    'PROBLEM_AGENT': [ '@REFERENTIAL #JJ @PROBLEM_NOUN', # can be both BKG and OWN => remove
                       '@REFERENTIAL @PROBLEM_NOUN',
                       'the @PROBLEM_NOUN' ],

    'SOLUTION_AGENT': [ '@REFERENTIAL_PL #JJ @SOLUTION_NOUN', # BKG
                       '@REFERENTIAL_PL @SOLUTION_NOUN',
                       #'the @SOLUTION_NOUN',  # removed (too genral)
                       #'the #JJ @SOLUTION_NOUN' 
                       ],

    'PROBLEM_SOLUTION_AGENT': ['many of @REFERENTIAL @SOLUTION_NOUN', # BKG
                               'many of @REFERENTIAL @PROBLEM_NOUN'
                               ],

    'TEXTSTRUCTURE_AGENT': [ '@TEXT_NOUN CREF', # TXT
                             '@TEXT_NOUN CREF and CREF',
                             'this @TEXT_NOUN',
                             'next @TEXT_NOUN',
                             'next #CD @TEXT_NOUN',
                             'concluding @TEXT_NOUN',
                             '@BEFORE_ADJ @TEXT_NOUN',
                             '@TEXT_NOUN above',
                             '@TEXT_NOUN below',
                             'following @TEXT_NOUN',
                             'remaining @TEXT_NOUN',
                             'subsequent @TEXT_NOUN',
                             'following #CD @TEXT_NOUN',
                             'remaining #CD @TEXT_NOUN',
                             'subsequent #CD @TEXT_NOUN',
                             '@TEXT_NOUN that follow',
                             'rest of this @PRESENTATION_NOUN',
                             'remainder of this @PRESENTATION_NOUN',
                             'in @TEXT_NOUN CREF , @SELF_NOM',
                             'in this @TEXT_NOUN , @SELF_NOM',
                             'in this @TEXT_NOUN @SELF_NOM',
                             'in the next @TEXT_NOUN , @SELF_NOM',
                             'in the next @TEXT_NOUN @SELF_NOM',
                             'in @BEFORE_ADJ @TEXT_NOUN , @SELF_NOM',
                             'in the @BEFORE_ADJ @TEXT_NOUN , @SELF_NOM',
                             'in the @TEXT_NOUN above , @SELF_NOM',
                             'in the @TEXT_NOUN below , @SELF_NOM',
                             'in the following @TEXT_NOUN , @SELF_NOM',
                             'in the following @TEXT_NOUN @SELF_NOM',
                             'in the remaining @TEXT_NOUN , @SELF_NOM',
                             'in the remaining @TEXT_NOUN @SELF_NOM',
                             'in the subsequent @TEXT_NOUN , @SELF_NOM',
                             'in the subsequent @TEXT_NOUN @SELF_NOM',
                             'in the @TEXT_NOUN that follow , @SELF_NOM',
                             'in the rest of this @PRESENTATION_NOUN , @SELF_NOM',
                             'in the rest of this @PRESENTATION_NOUN @SELF_NOM',
                             'in the remainder of this @PRESENTATION_NOUN , @SELF_NOM',
                             'in the remainder of this @PRESENTATION_NOUN @SELF_NOM',
                             'below , @SELF_NOM',
                             'below @SELF_NOM',
                             'the @AIM_NOUN of this @TEXT_NOUN' ],

    }

