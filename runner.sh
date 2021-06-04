bash runnerfile

## if run in parallel
#np=5
#cat runnerfile | xargs -L 1 -I CMD -P ${np} bash -c CMD

## after getting all repetition results, to aggregate all of them
python aggregateResults.py
