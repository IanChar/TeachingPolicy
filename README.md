Code committed was pair programmed.

# TeachingPolicy
Use Bayesian optimization to find the best teaching policy.

## Dependency

```
pip install bayesian-optimization
```

## Running Simulations

#### Generating test set

In order to evaluate, a file containing test questions must first be generated. This can be done by...

```
python plan_eval.py
```

By default this command will generate 100 test questions in the file `100test.py`. This can be changed by altering the main section ot `plan_eval.py`.

#### Running `policy_opt.py`

In order to run simulations type the command

```
python policy_opt.py
```

By default this will optimize for the teaching policy that has the highest student performance for a set number of examples. The function `optimize_fastest` can be used to find the policy such that students achieve a certain performance threshold with the fewest number of examples. 

Note that if the test file generated was not `100test.py` this will need to be changed in the code. Similarly, default parameters can be changed at the top and bottom of the file.
