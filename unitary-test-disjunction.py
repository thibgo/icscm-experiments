import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from icscm import InvariantCausalSCM

#################################### conjunction
print('#################################### conjunction learning:')

X = pd.DataFrame(columns=["e", "A", "B", "C", "D", "E"], data={
                                                        'e': [0,0,0,0,0,0,1,1,1,1,1,1], 
                                                        'A': [0,0,0,0,0,0,0,0,0,0,0,0], 
                                                        'B': [0,1,1,1,0,0,0,1,1,1,0,0], 
                                                        'C': [1,0,1,1,0,0,1,0,1,1,0,0], 
                                                        'D': [1,1,1,1,1,0,1,1,1,1,1,0], 
                                                        'E': [1,1,1,1,1,1,1,1,1,1,1,1]})
y = np.array([0,0,1,1,0,0,0,0,1,1,0,0])

model = InvariantCausalSCM(model_type='conjunction', pruning=False)
model.fit(X, y)
pred = model.predict(X)

score = accuracy_score(y, pred)
if score > 0.99:
    print('#### TEST PASSED ####')
else:
    print('#### TEST FAILED ####')
    print(model)
    print( '  y ', y)
    print('pred', pred)
    print('accuracy', accuracy_score(y, pred))

#################################### disjunction
print('#################################### disjunction learning:')

X = pd.DataFrame(columns=["e", "A", "B", "C", "D", "E"], data={
                                                        'e': [0,0,0,0,0,0,1,1,1,1,1,1], 
                                                        'A': [0,0,0,0,0,0,0,0,0,0,0,0], 
                                                        'B': [0,1,1,0,0,0,0,1,1,0,0,0], 
                                                        'C': [1,0,1,0,0,0,1,0,1,0,0,0], 
                                                        'D': [1,1,1,1,1,0,1,1,1,1,1,0], 
                                                        'E': [1,1,1,1,1,1,1,1,1,1,1,1]})
y = np.array([1,1,1,0,0,0,1,1,1,0,0,0])



model = InvariantCausalSCM(model_type='disjunction', pruning=False)
model.fit(X, y)
pred = model.predict(X)

score = accuracy_score(y, pred)
if score > 0.99:
    print('#### TEST PASSED ####')
else:
    print('#### TEST FAILED ####')
    print(X)
    print(model)
    print( '  y ', y)
    print('pred', pred)
    print('accuracy', accuracy_score(y, pred))
    print('model stream')
    print(model.stream.getvalue())