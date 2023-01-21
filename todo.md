
- asserts on the matrices having the form they need
- check that (0, 0) works nice

- why Up2P gives such a bad results ? 
- check that metrics are consistent 

- YES: why rotation matrix after prerotation gets affected in all the dimensions ? - multiplication mismatch somewhere?
- YES: check that for Identity prerotation results are of the same magnitude for both no- and prerotation based solvers
    Solution: test more than for one point - to make it slightly more robust

- YES: why euler angles are sometime (-180, y, 180) and sometimes (0, y, 0) - is it possible that this introduces a bug somewhere?  
    Problem: main metrics for orientation erorr is sensitive to those cases

