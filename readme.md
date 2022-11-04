To run my code, first of all, you need to unzip the program.zip first. 
Then, run the `cd program` to enter the directory, and 
then, you can run `python train.py` to run my program.

Or alternatively, you can run the following command to run my program in the background thread, and the output will be yielded into the log.out file: 

```python 
nohup python -u train.py > log.out 2>&1 &
```

Also, as you can see from the above command, the log.out contains all the outputs of my program, so if you want, you can have a look at the log.out file