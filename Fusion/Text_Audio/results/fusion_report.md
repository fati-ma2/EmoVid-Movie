# Fusion on the professor's split

Dev 442 clips / 50 films; test 439 clips / 50 films. Macro F1 with 95% film-level bootstrap intervals.

## Single models

| model | dev macro F1 | test macro F1 | test accuracy |
|---|---|---|---|
| text | 0.299 (0.231 to 0.348) | 0.305 (0.252 to 0.344) | 31.9% |
| audio | 0.253 (0.193 to 0.296) | 0.197 (0.151 to 0.235) | 20.7% |
| video | 0.375 (0.293 to 0.436) | 0.345 (0.297 to 0.380) | 37.8% |

## Dev: every rule (out-of-sample; weighted and stacking cross-fitted over 5 folds of films)

| combination | equal | product | weighted | stacking | chosen |
|---|---|---|---|---|---|
| text+audio | 0.270 | 0.288 | 0.304 | 0.221 | weighted |
| text+video | 0.382 | 0.387 | 0.389 | 0.295 | weighted |
| audio+video | 0.341 | 0.354 | 0.373 | 0.297 | weighted |
| text+audio+video | 0.362 | 0.386 | 0.364 | 0.305 | product |

## Test: the rule chosen on dev for each combination (difference with video alone, paired)

| combination | rule | test macro F1 | test accuracy | difference with video |
|---|---|---|---|---|
| text+audio | weighted (text 0.7, audio 0.3) | 0.291 (0.239 to 0.330) | 31.4% | -0.054 (-0.110 to -0.002); share <= 0: 98% |
| text+video **(final)** | weighted (text 0.4, video 0.6) | 0.341 (0.288 to 0.376) | 37.4% | -0.004 (-0.026 to +0.012); share <= 0: 68% |
| audio+video | weighted (audio 0.2, video 0.8) | 0.340 (0.285 to 0.377) | 37.6% | -0.005 (-0.030 to +0.018); share <= 0: 70% |
| text+audio+video | product | 0.365 (0.296 to 0.418) | 39.6% | +0.020 (-0.031 to +0.068); share <= 0: 24% |

## Complementarity on dev

| combination | oracle (at least one model right) | all models agree |
|---|---|---|
| text+audio | 43.0% | 25.3% |
| text+video | 50.0% | 34.4% |
| audio+video | 50.2% | 26.0% |
| text+audio+video | 57.7% | 11.8% |

Final system (best dev score): text+video, weighted rule.

Per-class F1 of the final system:

| class | dev | test |
|---|---|---|
| amusement | 0.472 | 0.342 |
| awe | 0.346 | 0.329 |
| contentment | 0.389 | 0.397 |
| excitement | 0.409 | 0.177 |
| anger | 0.373 | 0.540 |
| disgust | 0.281 | 0.308 |
| fear | 0.440 | 0.450 |
| sadness | 0.404 | 0.182 |
