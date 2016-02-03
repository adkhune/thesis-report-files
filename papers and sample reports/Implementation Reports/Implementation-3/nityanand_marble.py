import neuralnetworkQ as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy



print( '\n------------------------------------------------------------')
print( "Reinforcement Learning Example: Dynamic Marble on a Track")

# Define the problem

def reinforcement(s,sn):
    #goal = 5
    #return 0 if abs(sn[0]-goal) < 1 else -1
    if sn[0] < 4 and sn[1] == -1:
        r = 0
    elif sn[0] > 3 and sn[0] < 7 and sn[1] == 0:
        r = 0
    elif sn[0] > 6 and sn[1] == 1:
        r = 0
    else:
        r = -1  
    return r       

def initialState():
    #return np.array([10*np.random.random_sample(), 0.0])
    #initialState = np.random.randint(1,10)
    initialStates = 0
    process = -1.0
    #return np.array([np.random.randint(1,10), 0.0])
    return np.array([initialStates, process])
    #return np.array([initialStates])

def nextState(s,a):
#    s = copy.copy(s)   # s[0] is position, s[1] is velocity. a is -1, 0 or 1
#    deltaT = 0.1                           # Euler integration time step
#    s[0] += deltaT * s[1]                  # Update position
#    s[1] += deltaT * (2 * a - 0.2 * s[1])  # Update velocity. Includes friction
#    if s[0] < 0:        # Bound next position. If at limits, set velocity to 0.
#        s = [0,0]
#    elif s[0] > 10:
##        s = [10,0]
    s = copy.copy(s)
    if s[0] < 10:
        s[0] += 1 #location
        s[1] = a
        #s[1] = np.random.randint(-1,2) #action -1:local 0:offload_local 1:offload_remote
    return s

validActions = (-1,0,1)

# training Loop
gamma = 0.5
nh = 5
nTrials = 100
nStepsPerTrial = 10
nSCGIterations = 10
finalEpsilon = 0.01
epsilonDecay = np.exp(np.log(finalEpsilon)/(nTrials)) # to produce this final value

nnet = nn.NeuralNetworkQ(3,nh,1,((0,10), (-1,1), (-1,1)))
epsilon = 1
epsilonTrace = np.zeros(nTrials)
rtrace = np.zeros(nTrials)
for trial in range(nTrials):
    # Collect nStepsPerRep samples of X, R, Qn, and Q, and update epsilon
    X,R,Qn,Q,epsilon = nnet.makeSamples(initialState,nextState,reinforcement,validActions,nStepsPerTrial,epsilon)
    # Update the Q neural network.
    nnet.train(X,R,Qn,Q,gamma=gamma, nIterations=nSCGIterations) #  weightPrecision=1e-8, errorPrecision=1e-10)
    epsilon *= epsilonDecay
    # Rest is for plotting
    epsilonTrace[trial] = epsilon
    rtrace[trial] = np.mean(R)

    print('Trial',trial,'mean R',np.mean(R))


##  Plotting functions

def plotStatus(net,trial,epsilonTrace,rtrace):
#    plt.subplot(4,3,1)
#    plt.plot(epsilonTrace[:trial+1])
#    plt.ylabel("Random Action Probability ($\epsilon$)")
#    plt.ylim(0,1)
#    plt.savefig("tpp1")
#########################################################################    
#    plt.subplot(4,3,2)
#    plt.plot(X[:,0])
#    plt.plot([0,X.shape[0]], [5,5],'--',alpha=0.5,lw=5)
#    plt.ylabel("$x$")
#    plt.ylim(-1,11)
#    #qs = [[net.use([s,0,a]) for a in actions] for s in range(11)]
#    qs = net.use(np.array([[s,0,a] for a in validActions for s in range(11)]))
#    #print np.hstack((qs,-1+np.argmax(qs,axis=1).reshape((-1,1))))
#    plt.savefig("tpp2")
#########################################################################    
#    plt.subplot(4,3,3)
#    acts = ["L","0","R"]
#    actsiByState = np.argmax(qs.reshape((len(validActions),-1)),axis=0)
#    for i in range(11):
#        plt.text(i,0,acts[actsiByState[i]])
#        plt.xlim(-1,11)
#        plt.ylim(-1,1)
#    plt.text(2,0.2,"Policy for Zero Velocity")
#    plt.axis("off")
#########################################################################    
#    plt.subplot(4,3,4)
#    plt.plot(rtrace[:trial+1],alpha=0.5)
#    #plt.plot(np.convolve(rtrace[:trial+1],np.array([0.02]*50),mode='valid'))
#    binSize = 20
#    if trial+1 > binSize:
#        # Calculate mean of every bin of binSize reinforcement values
#        smoothed = np.mean(rtrace[:int(trial/binSize)*binSize].reshape((int(trial/binSize),binSize)),axis=1)
#        plt.plot(np.arange(1,1+int(trial/binSize))*binSize,smoothed)
#    plt.ylabel("Mean reinforcement")
########################################################################    
#    plt.subplot(4,3,5) #original
    plt.subplot(3,1,1) #modified
    plt.plot(X[:,0],X[:,1])
    plt.plot(X[0,0],X[0,1],'o')
    plt.xlabel("$x$")
    plt.ylabel("$\dot{x}$")
#    plt.fill_between([4,6],[-5,-5],[5,5],color="red",alpha=0.3)
    plt.xlim(-1,11)
#    plt.ylim(-5,5) #original
    plt.ylim(-2,2) #modified
########################################################################    
#    plt.subplot(4,3,6)
#    net.draw(["$x$","$\dot{x}$","$a$"],["Q"])
########################################################################
#    plt.subplot(4,3,7) #original
    plt.subplot(3,2,6) #modified
    n = 20
    n = 20
    positions = np.linspace(0,10,n)
    velocities =  np.linspace(-5,5,n)
    xs,ys = np.meshgrid(positions,velocities)
    #states = np.vstack((xs.flat,ys.flat)).T
    #qs = [net.use(np.hstack((states,np.ones((states.shape[0],1))*act))) for act in actions]
    xsflat = xs.flat
    ysflat = ys.flat
    qs = net.use(np.array([[xsflat[i],ysflat[i],a] for a in validActions for i in range(len(xsflat))]))
    #qs = np.array(qs).squeeze().T
    qs = qs.reshape((len(validActions),-1)).T
    qsmax = np.max(qs,axis=1).reshape(xs.shape)
    cs = plt.contourf(xs,ys,qsmax)
    plt.colorbar(cs)
    plt.xlabel("$x$")
    plt.ylabel("$\dot{x}$")
    plt.title("Max Q")
########################################################################    
#    plt.subplot(4,3,8) #original
    plt.subplot(3,2,5) #modified
    n = 20
    acts = np.array(validActions)[np.argmax(qs,axis=1)].reshape(xs.shape)
    cs = plt.contourf(xs,ys,acts,[-2, -0.5, 0.5, 2])
    plt.colorbar(cs)
    plt.xlabel("$x$")
    plt.ylabel("$\dot{x}$")
    plt.title("Actions")
#######################################################################
#    s = plt.subplot(4,3,10) #original
    s = plt.subplot(3,2,4) #modified
    rect = s.get_position()
    rect = s.get_position()
    ax = Axes3D(plt.gcf(),rect=rect)
    ax.plot_surface(xs,ys,qsmax,cstride=1,rstride=1,cmap=cm.jet,linewidth=0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\dot{x}$")
    #ax.set_zlabel("Max Q")
    plt.title("Max Q")
########################################################################
#    s = plt.subplot(4,3,11) #original
    s = plt.subplot(3,2,3) #modified
    rect = s.get_position()
    rect = s.get_position()
    ax = Axes3D(plt.gcf(),rect=rect)
    ax.plot_surface(xs,ys,acts,cstride=1,rstride=1,cmap=cm.jet,linewidth=0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\dot{x}$")
    #ax.set_zlabel("Action")
    plt.title("Action")

def testIt(Qnet,nTrials,nStepsPerTrial):
    xs = np.linspace(0,10,nTrials)
    ####################################################################
#    plt.subplot(4,3,12)
    for x in xs:
        s = [x,0] ## 0 velocity
        xtrace = np.zeros((nStepsPerTrial,2))
        for step in range(nStepsPerTrial):
            a,_ = Qnet.epsilonGreedy(s,validActions,0.0) # epsilon = 0
            s = nextState(s,a)
            xtrace[step,:] = s
#        plt.plot(xtrace[:,0],xtrace[:,1])
#        plt.xlim(-1,11)
#        plt.ylim(-5,5)
#        plt.plot([5,5],[-5,5],'--',alpha=0.5,lw=5)
#        plt.ylabel('$\dot{x}$')
#        plt.xlabel('$x$')
#        plt.title('State Trajectories for $\epsilon=0$')



plotStatus(nnet,nTrials,epsilonTrace,rtrace)
testIt(nnet,10,500)

plt.show()

