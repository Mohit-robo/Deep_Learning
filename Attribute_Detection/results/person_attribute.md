                SigLIP2 Attribute Detection — Benchmark Report             
    ────────────────────────────────────────────────────────────────────────

    MULTI-CLASS GROUPS  (top-1 accuracy)

    gender           █████████████████████████████░   98.9%  (988/999)
        man                                99.1%  (n=540)
        woman                              98.7%  (n=459)
    age              █████████████████░░░░░░░░░░░░░   58.9%  (588/999)
        teenager                           68.7%  (n=265)
        young adult                        26.8%  (n=280)
        middle-aged person                 91.3%  (n=219)
        a photo of an elderly person       55.7%  (n=235)
    angle            ██████████████████████░░░░░░░░   74.7%  (746/999)
        the front                          88.9%  (n=370)
        the back                           71.9%  (n=299)
        the side                           61.2%  (n=330)
    hair             ████████████████████░░░░░░░░░░   67.4%  (627/930)
        black hair                         51.1%  (n=90)
        brown hair                         49.5%  (n=295)
        blonde hair                        74.2%  (n=295)
        grey or white hair                 99.4%  (n=175)
        shaved or very short hair          56.0%  (n=75)
    top_type         ████████████████████████████░░   94.2%  (941/999)
        a t-shirt                          91.6%  (n=155)
        a tank top                         100.0%  (n=175)
        a hoodie                           98.8%  (n=170)
        a knit sweater                     82.9%  (n=205)
        a leather jacket                   96.6%  (n=145)
        a puffer jacket                    98.0%  (n=149)
    top_color        ████████████████████████████░░   94.1%  (940/999)
        a white top                        88.6%  (n=185)
        a black top                        88.1%  (n=135)
        a grey top                         96.8%  (n=125)
        a navy blue top                    92.0%  (n=125)
        a bright red top                   96.4%  (n=139)
        a forest green top                 98.7%  (n=155)
        a yellow top                       99.3%  (n=135)
    bottom_type      ████████████████░░░░░░░░░░░░░░   54.1%  (540/999)
        denim jeans                        38.2%  (n=170)
        formal trousers                    59.1%  (n=159)
        cargo pants                        45.5%  (n=165)
        sweatpants                         63.4%  (n=175)
        shorts                             100.0%  (n=150)
        leggings                           25.0%  (n=180)
    bottom_color     █████████████████░░░░░░░░░░░░░   59.3%  (592/999)
        white trousers or bottoms          78.8%  (n=170)
        black trousers or bottoms          55.9%  (n=195)
        blue or dark blue trousers or bottoms 51.8%  (n=195)
        dark grey trousers or bottoms      65.9%  (n=214)
        khaki or beige trousers or bottoms 47.6%  (n=225)

  BINARY GROUPS  (sigmoid threshold)

    Group                     Acc    Prec     Rec      F1    TP    FP    TN    FN
    ----------------------  ------  ------  ------  ------  ----  ----  ----  ----
    carrying_backpack        85.5%   50.0%    0.7%    1.4%     1     1   853   144
    carrying_handbag         84.6%    0.0%    0.0%    0.0%     0     0   845   154
    holding_umbrella         86.9%  100.0%    3.0%    5.8%     4     0   864   131
    holding_smartphone       86.5%    0.0%    0.0%    0.0%     0     0   864   135
    wearing_cap              87.0%    0.0%    0.0%    0.0%     0     0   869   130
    wearing_sunglasses       81.8%  100.0%    1.6%    3.2%     3     0   814   182

    ────────────────────────────────────────────────────────────────────────
    Mean accuracy (all groups) : 79.5%
    Mean accuracy (multi-class): 75.2%
    Mean accuracy (binary)     : 85.4%
    Throughput                 : 6.8 images / second
    ────────────────────────────────────────────────────────────────────────



## Qwen3-vl:2b VLM 


                SigLIP2 Attribute Detection — Benchmark Report             
    ────────────────────────────────────────────────────────────────────────

    MULTI-CLASS GROUPS  (top-1 accuracy)

    gender           ███████████████████░░░░░░░░░░░   65.0%  (13/20)
        man                                100.0%  (n=5)
        woman                              53.3%  (n=15)
    age              ████████████░░░░░░░░░░░░░░░░░░   40.0%  (8/20)
        young adult                        50.0%  (n=10)
        middle-aged person                 20.0%  (n=5)
        a photo of an elderly person       40.0%  (n=5)
    angle            █████████████████████░░░░░░░░░   70.0%  (14/20)
        the front                          100.0%  (n=10)
        the back                           40.0%  (n=5)
        the side                           40.0%  (n=5)
    hair             ███████████████████░░░░░░░░░░░   65.0%  (13/20)
        brown hair                         20.0%  (n=5)
        blonde hair                        60.0%  (n=5)
        grey or white hair                 80.0%  (n=5)
        shaved or very short hair          100.0%  (n=5)
    top_type         ███████████████████░░░░░░░░░░░   65.0%  (13/20)
        a t-shirt                          100.0%  (n=5)
        a knit sweater                     100.0%  (n=5)
        a leather jacket                   30.0%  (n=10)
    top_color        ██████████████████████░░░░░░░░   75.0%  (15/20)
        a white top                        60.0%  (n=5)
        a black top                        60.0%  (n=5)
        a navy blue top                    90.0%  (n=10)
    bottom_type      █████████████████████░░░░░░░░░   70.0%  (14/20)
        denim jeans                        60.0%  (n=5)
        cargo pants                        100.0%  (n=5)
        sweatpants                         20.0%  (n=5)
        leggings                           100.0%  (n=5)
    bottom_color     ██████████████████░░░░░░░░░░░░   60.0%  (12/20)
        black trousers or bottoms          80.0%  (n=15)
        dark grey trousers or bottoms       0.0%  (n=5)

  BINARY GROUPS  (sigmoid threshold)

    Group                     Acc    Prec     Rec      F1    TP    FP    TN    FN
    ----------------------  ------  ------  ------  ------  ----  ----  ----  ----
    carrying_backpack        75.0%    0.0%    0.0%    0.0%     0     0    15     5
    carrying_handbag         90.0%    0.0%    0.0%    0.0%     0     2    18     0
    holding_umbrella        100.0%    0.0%    0.0%    0.0%     0     0    20     0
    holding_smartphone      100.0%  100.0%  100.0%  100.0%     5     0    15     0
    wearing_cap             100.0%  100.0%  100.0%  100.0%     5     0    15     0
    wearing_sunglasses       90.0%  100.0%   60.0%   75.0%     3     0    15     2

    ────────────────────────────────────────────────────────────────────────
    Mean accuracy (all groups) : 76.1%
    Mean accuracy (multi-class): 63.7%
    Mean accuracy (binary)     : 92.5%
    Throughput                 : 0.0 images / second
    ────────────────────────────────────────────────────────────────────────
