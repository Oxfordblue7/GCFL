bash runnerfile

## if run in parallel
np=10
cat runnerfile | xargs -L 1 -I CMD -P ${np} bash -c CMD
