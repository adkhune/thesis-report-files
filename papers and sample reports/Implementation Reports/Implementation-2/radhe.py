import numpy as np
import random as rm
import matplotlib.pyplot as plt
from copy import copy
from IPython.display import display, clear_output

def printBoard(board):
    print('''
bandwidth={} |Data={} |CPU_Instance={}
-----
app={} |Cloud_Vendor_Available={} |Location={}
-----
'''.format(*tuple(board)))

def printBoardQs(board,Q):
    #printBoard(board)
    printParameters(board)
    Qs = [Q.get((tuple(board),m), 0) for m in range(3)]
    print('Reinforcements Received:')
    print('''Local Processing:{:.2f} | Offload on Local Servers:{:.2f} | Offload on Remote Servers:{:.2f}
'''.format(*Qs))


def printParameters(board):
    print('''
bandwidth= {} |Data= {} |CPU_Instance= {} |Wifi= {}
'''.format(*tuple(board)))

print('let\'s see what are my state parameters')
#printBoard(np.array(['1','0','1','1','5','9']))
printParameters(np.array(['Speed_Low','Data_Small','CPU_Low','On']))

#print('okay now let\'s genearete a random number geneartor for each of these parameters')

#bandwidth = rm.randint(1,10)
#bandwidth = np.random.randint(1,10,size=60)
#data = np.random.randint(1,10, size = 60)
#cpu = np.random.randint(1,10, size = 60)
#app = np.random.randint(1,10, size = 60)
#cloud_vendor = np.random.randint(1,10, size = 60)
#location = rm.randint(1,10)
#location = np.random.randint(1,10, size =60)

Bandwidth = np.array(['Speed_Low','Speed_Normal','Speed_High'])
Data = np.array(['Data_Small','Data_Medium','Data_Big'])
CPU = np.array(['CPU_Low','CPU_Normal','CPU_High'])
Wifi = np.array(['On','Off'])
Out = np.array(['Local_Procssing','Offload_Local','Offload_Remote'])

#print("location=",location)
#board = np.array(['X',' ','O', ' ','X','O', 'X',' ',' '])
#board1 = np.array([bandwidth,data,cpu,app,cloud_vendor,location])
board2 = np.array([rm.choice(Bandwidth),rm.choice(Data),rm.choice(CPU),rm.choice(Wifi)])
print('print parameters')
printParameters(board2)
#print('print board1')
#printBoard(board1)

#Q = {} #empty table
#Q[(tuple(board2),1)] = 4

#print("Q:",Q)
#print("Q[(tuple(board2),1)]:",Q[(tuple(board2),1)])
#print("Q.get((tuple(board2),1),42):",Q.get((tuple(board2),1),42))

#rho = 0.1 # learning rate
#Q[(tuple(board),1)] += rho * (-1 - Q[(tuple(board),1)])
#print("after Q[(tuple(board),1)] += rho * (-1 - Q[(tuple(board),1)]):", Q[(tuple(board),1)])
#print('rm.choice(list(enumerate(Out))):',rm.choice(list(enumerate(Out))))
#print('rm.choice(list(enumerate(Out)))[0]:',rm.choice(list(enumerate(Out)))[0])
#print('list(enumerate(Out)):',list(enumerate(Out)))
#print('list(Out):',list(Out))
#print('Out:',Out)
#print('list(enumerate(Out)):',list(enumerate(Out)))
#print('list(enumerate(Out))[:0]:',list(enumerate(Out))[:0])
#print('np.random.uniform():',np.random.uniform())
#random_index = rm.randrange(0,len(Out))
#print ('Out[random_index]:',Out[random_index])

def epsilonGreedy(epsilon, Q, board, Out):
    #validMoves = np.where(board == ' ')[0]
    validMoves = np.array([0,1,2])
    #print('validMoves:',validMoves)
    if np.random.uniform() < epsilon:
        # Random Move
        tp = rm.choice(list(enumerate(Out)))[0]
        print('tp:',tp)
        return tp
        #return rm.choice(list(enumerate(Out)))[0]
        #return np.random.choice(validMoves)
    else:
        # Greedy Move
        Qs = np.array([Q.get((tuple(board),m), 0) for m in validMoves])
        tp = validMoves[ np.argmax(Qs) ] 
        print('tp:',tp)
        return tp
        #return validMoves[ np.argmax(Qs) ]
        
#print('epsilonGreedy(0.8,Q,board2,Out):',epsilonGreedy(0.8,Q,board2,Out))


print('here goes part before for loop')

maxGames = 200
rho = 0.2
epsilonDecayRate = 0.99
epsilon = 0.8
graphics = True
showMoves = not graphics

outcomes = np.zeros(maxGames)
epsilons = np.zeros(maxGames)
Q = {}

if graphics:
    fig = plt.figure(figsize=(10,10))

print('here goes a for loop')
#for i in range(60):
    #print (i)
    #location = np.random.randint(1,10, size =1)
#    print("location=",location[i])
#    board2 = np.array([bandwidth[i],data[i],cpu[i],app[i],cloud_vendor[i],location[i]])
#    printBoard(board2)
#board2 = np.array([rm.choice(Bandwidth),rm.choice(Data),rm.choice(CPU),rm.choice(Wifi)])
for nGames in range(maxGames):
    epsilon *= epsilonDecayRate
    epsilons[nGames] = epsilon
    step = 0
    move = epsilonGreedy(epsilon, Q, board2, Out)
    board2_all = {}
    board2 = np.array([rm.choice(Bandwidth),rm.choice(Data),rm.choice(CPU),rm.choice(Wifi)])
    board2_all[nGames] = board2
    if (tuple(board2),move) not in Q:
            Q[(tuple(board2),move)] = 0  # initial Q value for new board,move
    
    if board2[3] == 'On':
        print('Wifi is ON')
        #if board2[0] == 'Speed_Low' and 'Speed_Normal':
        if board2[0] == 'Speed_Low' or board2[0] == 'Speed_Normal':
            print('Bandwidth = Speed_Low or Speed_Normal')

            if board2[1] == 'Data_Small' and board2[2] == 'CPU_High':
                print('Data_Small and CPU_High so you can offload')
                Q[(tuple(board2),1)] = 1
                Q[(tuple(board2),2)] = 0
                Q[(tuple(board2),0)] = -1
            else:
                print('Don\'t offload')
                Q[(tuple(board2),1)] = 0
                Q[(tuple(board2),2)] = -1
                Q[(tuple(board2),0)] = 1
        else:
            if board2[2] == 'CPU_Normal' or board2[2] == 'CPU_High':
                print('CPU_Normal or CPU_High so you can offload')
                Q[(tuple(board2),1)] = 1
                Q[(tuple(board2),2)] = 0
                Q[(tuple(board2),0)] = -1
            else:
                print('Don\'t offload (this is second if loop)')
                Q[(tuple(board2),1)] = 0
                Q[(tuple(board2),2)] = -1
                Q[(tuple(board2),0)] = 1
    else:
        print('Wifi is OFF')
        if board2[0] == 'Speed_Low' or board2[0] == 'Speed_Normal':
            print('Bandwidth = Speed_Low or Speed_Normal when wifi is off')
            if board2[1] == 'Data_Small' and board2[2] == 'CPU_High':
                print('Data_Small and CPU_High so you can offload:Out2')
                Q[(tuple(board2),1)] = 0
                Q[(tuple(board2),2)] = 1
                Q[(tuple(board2),0)] = -1
            else:
                print('Don\'t offload when wifi is off')
                Q[(tuple(board2),1)] = -1
                Q[(tuple(board2),2)] = -1
                Q[(tuple(board2),0)] = 1
        else:
            if board2[2] == 'CPU_Normal' or board2[2] == 'CPU_High':
                print('CPU_Normal or CPU_High so you can offload:Out2')
                Q[(tuple(board2),1)] = 0
                Q[(tuple(board2),2)] = 1
                Q[(tuple(board2),0)] = -1
            else:
                print('Don\'t offload (this is second if loop when wifi is off)')
                Q[(tuple(board2),1)] = -1
                Q[(tuple(board2),2)] = -1
                Q[(tuple(board2),0)] = 1
    
    #print (i)
    #location = np.random.randint(1,10, size =1)
#    print("location=",location[i])
#    board2 = np.array([bandwidth[i],data[i],cpu[i],app[i],cloud_vendor[i],location[i]])
#    printBoard(board2)

#--------------------------Just For Plotting the outcomes---------------
print('after for loop')
printBoardQs(board2,Q)

outcomes = np.random.choice([-1,0,1],replace=True,size=(1000))
#print('outcomes[:10]:',outcomes[:10])
#print('Q:',Q)
#print('Q.shape:',Q.shape) //did not work

#print('Q.values():\n',Q.values())
#print('Q.keys():\n\n',Q.keys())
#print('Q.items():\n\n',Q.items())

#for k in Q.keys():
#	print(k, Q[k])
	
#outcomes = np.array[Q.values()]
#print('outcomes[:10]:',outcomes[:10])

names = ['id','data']
formats = ['f8','f8']
dtype = dict(names = names, formats=formats)
array=np.array([[key,val] for (key,val) in Q.iteritems()],dtype)
print(repr(array))
#plt.plot(Q)
def plotOutcomes(outcomes,epsilons,maxGames,nGames):
    if nGames==0:
        return
    nBins = 100
    nPer = int(maxGames/nBins)
    outcomeRows = outcomes.reshape((-1,nPer))
    outcomeRows = outcomeRows[:int(nGames/float(nPer))+1,:]
    avgs = np.mean(outcomeRows,axis=1)
    plt.subplot(3,1,1)
    xs = np.linspace(nPer,nGames,len(avgs))
    plt.plot(xs, avgs)
    plt.xlabel('Games')
    plt.ylabel('Mean of Outcomes (0=draw, 1=X win, -1=O win)')
    plt.title('Bins of {:d} Games'.format(nPer))
    plt.subplot(3,1,2)
    plt.plot(xs,np.sum(outcomeRows==-1,axis=1),'r-',label='Losses')
    plt.plot(xs,np.sum(outcomeRows==0,axis=1),'b-',label='Draws')
    plt.plot(xs,np.sum(outcomeRows==1,axis=1),'g-',label='Wins')
    plt.legend(loc="center")
    plt.ylabel('Number of Games in Bins of {:d}'.format(nPer))
    plt.subplot(3,1,3)
    plt.plot(epsilons[:nGames])
    plt.ylabel('$\epsilon$')
    


#plt.figure(figsize=(8,8))
#plotOutcomes(outcomes,np.zeros(1000),1000,1000)
#plt.show()
#--------------------------Just For Plotting the outcomes---------------
