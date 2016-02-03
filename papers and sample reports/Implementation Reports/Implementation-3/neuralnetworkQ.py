''' Neural network with multiple hidden layers, include none.
 For nonlinear regression (prediction of real-valued outputs)
   net = NeuralNetwork(ni,nh,no)       # ni is number of attributes each sample,
                                   # nh is number of hidden units in each layer, or None
                                   # no is number of output components
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x no
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   Y,Z = net.use(Xtest,allOutputs=True)  # Y is nSamples x no, Z is nSamples x nh

  Also implements NeuralNetworkQ
'''

import scaledconjugategradient as scg
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp  # to allow access to number of elapsed iterations
import mlutils as ml

# def pickleLoad(filename):
#     with open(filename,'rb') as fp:
#         nnet = cPickle.load(fp)
#     nnet.iteration = mp.Value('i',0)
#     nnet.trained = mp.Value('b',False)
#     return nnet

class NeuralNetwork:
    def __init__(self,ni,nhs,no):
        if nhs == 0 or nhs == [0] or nhs is None or nhs == [None]:
            nhs = None
        else:
            try:
                nihs = [ni] + list(nhs)
            except:
                nihs = [ni] + [nhs]
                nhs = [nhs]
        if nhs is not None:
            self.Vs = [np.random.uniform(-0.1,0.1,size=(1+nihs[i],nihs[i+1])) for i in range(len(nihs)-1)]
            self.W = np.random.uniform(-0.1,0.1,size=(1+nhs[-1],no))
        else:
            self.Vs = None
            self.W = np.random.uniform(-0.1,0.1,size=(1+ni,no))
            
        # print [v.shape for v in self.Vs], self.W.shape
        self.ni,self.nhs,self.no = ni,nhs,no
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.iteration = mp.Value('i',0)
        self.trained = mp.Value('b',False)
        self.reason = None
        self.errorTrace = None
        
    def getSize(self):
        return (self.ni,self.nhs,self.no)

    def getErrorTrace(self):
        return self.errorTrace
    
    def getNumberOfIterations(self):
        return self.numberOfIterations
        
    def train(self,X,T,
              nIterations=100,weightPrecision=0,errorPrecision=0,verbose=False):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
        X = self._standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
        T = self._standardizeT(T)

        # Local functions used by gradientDescent.scg()

        def objectiveF(w):
            self._unpack(w)
            Y,_ = self._forward_pass(X)
            return 0.5 * np.mean((Y - T)**2)

        def gradF(w):
            self._unpack(w)
            Y,Z = self._forward_pass(X)
            delta = (Y - T) / (X.shape[0] * T.shape[1])
            dVs,dW = self._backward_pass(delta,Z)
            return self._pack(dVs,dW)

        scgresult = scg.scg(self._pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True,
                            verbose=verbose)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace) - 1
        self.trained.value = True
        return self
    
    def use(self,X,allOutputs=False):
        Xst = self._standardizeX(X)
        Y,Z = self._forward_pass(Xst)
        Y = self._unstandardizeT(Y)
        if Z is None:
            return (Y,None) if allOutputs else Y
        else:
            return (Y,Z[1:]) if allOutputs else Y

    def draw(self,inputNames = None, outputNames = None):
        ml.draw(self.Vs, self.W, inputNames, outputNames)


            
    def _standardizeX(self,X):
        return (X - self.Xmeans) / self.Xstds
    def _unstandardizeX(self,Xs):
        return self.Xstds * Xs + self.Xmeans
    def _standardizeT(self,T):
        return (T - self.Tmeans) / self.Tstds
    def _unstandardizeT(self,Ts):
        return self.Tstds * Ts + self.Tmeans

    def _forward_pass(self,X):
        if self.nhs is None:
            # no hidden units, just linear output layer
            Y = np.dot(X, self.W[1:,:]) + self.W[0:1,:]
            Zs = [X]
        else:
            Zprev = X
            Zs = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
                Zs.append(Zprev)
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
        return Y, Zs

    def _backward_pass(self,delta,Z):
        if self.nhs is None:
            # no hidden units, just linear output layer
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[0].T, delta)))
            dVs = None
        else:
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
            dVs = []
            delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
            for Zi in range(len(self.nhs),0,-1):
                Vi = Zi - 1 # because X is first element of Z
                dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                                 np.dot( Z[Zi-1].T, delta)))
                dVs.insert(0,dV)
                delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
        return dVs,dW

    def _pack(self,Vs,W):
        # r = np.hstack([V.flat for V in Vs] + [W.flat])
        # print 'pack',len(Vs), Vs[0].shape, W.shape,r.shape
        if Vs is None:
            return np.array(W.flat)
        else:
            return np.hstack([V.flat for V in Vs] + [W.flat])

    def _unpack(self,w):
        if self.nhs is None:
            self.W[:] = w.reshape((self.ni+1, self.no))
        else:
            first = 0
            numInThisLayer = self.ni
            for i in range(len(self.Vs)):
                self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
                first += (numInThisLayer+1) * self.nhs[i]
                numInThisLayer = self.nhs[i]
            self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))

    def pickleDump(self,filename):
        # remove shared memory objects. Can't be pickled
        n = self.iteration.value
        t = self.trained.value
        self.iteration = None
        self.trained = None
        with open(filename,'wb') as fp:
        #    pickle.dump(self,fp)
            cPickle.dump(self,fp)
        self.iteration = mp.Value('i',n)
        self.trained = mp.Value('b',t)


class NeuralNetworkQ(NeuralNetwork):

    def __init__(self,ni,nh,no,inputminmax):
        NeuralNetwork.__init__(self,ni,nh,no)
        inputminmaxnp = np.array(inputminmax)
        self.Xmeans = inputminmaxnp.mean(axis=1)
        self.Xstds = inputminmaxnp.std(axis=1)

    def train(self,X,R,Q,Y,gamma=1,
                 nIterations=100,weightPrecision=0,errorPrecision=0,verbose=False):
        X = self._standardizeX(X)

        def objectiveF(w):
            self._unpack(w)
            Y,_ = self._forward_pass(X)
            return 0.5 *np.mean((R+gamma*Q-Y)**2)

        def gradF(w):
            self._unpack(w)
            Y,Z = self._forward_pass(X)
            nSamples = X.shape[0]
            delta = -(R + gamma * Q - Y) / nSamples
            dVs,dW = self._backward_pass(delta,Z)
            return self._pack(dVs,dW)

        scgresult = scg.scg(self._pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True,
                            verbose=verbose)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)
        self.trained.value = True
        return self

    def use(self,X,allOutputs=False):
        Xst = self._standardizeX(X)
        Y,Z = self._forward_pass(Xst)
        # Y = self.unstandardizeT(Y)
        if Z is None:
            return (Y,None) if allOutputs else Y
        else:
            return (Y,Z[1:]) if allOutputs else Y

    def makeSamples(self, initialStateF, nextStateF, reinforcementF,
                    validActions, numSamples, epsilon, epsilonDecay=1):

        X = np.zeros((numSamples,self.getSize()[0]))
        R = np.zeros((numSamples,1))
        Qn = np.zeros((numSamples,1))
        Q = np.zeros((numSamples,1))

        s = initialStateF()
        a,q = self.epsilonGreedy(s,validActions,epsilon)

        # Collect data from numSamples steps
        for step in range(numSamples):
            sn = nextStateF(s,a)        # Update state, sn from s and a
            rn = reinforcementF(s,sn)   # Calculate resulting reinforcement
            an,qn = self.epsilonGreedy(sn,validActions,epsilon) # Forward pass for time t+1
            X[step,:], R[step,0], Qn[step,0], Q[step,0] = np.hstack((s,a)), rn, qn, q

            s,a,q = sn,an,qn               # advance one time step

            epsilon *= epsilonDecay  # decay can be 1 to not change epsilon
        
        return (X,R,Qn,Q,epsilon)

    def epsilonGreedy(self, state, validActions, epsilon):
        if np.random.uniform() < epsilon:
            # Random Move
            action = np.random.choice(validActions)
        else:
            # Greedy Move
            Qs = [self.use(np.hstack((np.array(state),a)))
                  for a in validActions]
            ai = np.argmax(Qs)
            action = validActions[ai]
        Q = self.use(np.hstack((np.array(state),action)))
        return action, Q


if __name__== "__main__":

    import copy
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    plt.ion()

    print( '\n------------------------------------------------------------')
    print( "Reinforcement Learning Example: Dynamic Marble on a Track")

    # Define the problem
    
    def reinforcement(s,s1):
        goal = 5
        return 0 if abs(s1[0]-goal) < 1 else -1

    def initialState():
        return np.array([10*np.random.random_sample(), 0.0])

    def nextState(s,a):
        s = copy.copy(s)   # s[0] is position, s[1] is velocity. a is -1, 0 or 1
        deltaT = 0.1                           # Euler integration time step
        s[0] += deltaT * s[1]                  # Update position
        s[1] += deltaT * (2 * a - 0.2 * s[1])  # Update velocity. Includes friction
        if s[0] < 0:        # Bound next position. If at limits, set velocity to 0.
            s = [0,0]
        elif s[0] > 10:
            s = [10,0]
        return s

    validActions = (-1,0,1)
    
    # training Loop
    gamma = 0.5
    nh = 5
    nTrials = 500
    nStepsPerTrial = 1000
    nSCGIterations = 10
    finalEpsilon = 0.01
    epsilonDecay = np.exp(np.log(finalEpsilon)/(nTrials)) # to produce this final value

    nnet = NeuralNetworkQ(3,nh,1,((0,10), (-3,3), (-1,1)))
    epsilon = 1
    epsilonTrace = np.zeros(nTrials)
    rtrace = np.zeros(nTrials)
    for trial in range(nTrials):
        # Collect nStepsPerRep samples of X, R, Qn, and Q, and update epsilon
        X,R,Qn,Q,epsilon = nnet.makeSamples(initialState,nextState,reinforcement,
                                            validActions,nStepsPerTrial,epsilon)
        # Update the Q neural network.
        nnet.train(X,R,Qn,Q,gamma=gamma, nIterations=nSCGIterations) #  weightPrecision=1e-8, errorPrecision=1e-10)
        epsilon *= epsilonDecay
        # Rest is for plotting
        epsilonTrace[trial] = epsilon
        rtrace[trial] = np.mean(R)

        print('Trial',trial,'mean R',np.mean(R))


    ##  Plotting functions

    def plotStatus(net,trial,epsilonTrace,rtrace):
        plt.subplot(4,3,1)
        plt.plot(epsilonTrace[:trial+1])
        plt.ylabel("Random Action Probability ($\epsilon$)")
        plt.ylim(0,1)
        plt.subplot(4,3,2)
        plt.plot(X[:,0])
        plt.plot([0,X.shape[0]], [5,5],'--',alpha=0.5,lw=5)
        plt.ylabel("$x$")
        plt.ylim(-1,11)
        #qs = [[net.use([s,0,a]) for a in actions] for s in range(11)]
        qs = net.use(np.array([[s,0,a] for a in validActions for s in range(11)]))
        #print np.hstack((qs,-1+np.argmax(qs,axis=1).reshape((-1,1))))
        plt.subplot(4,3,3)
        acts = ["L","0","R"]
        actsiByState = np.argmax(qs.reshape((len(validActions),-1)),axis=0)
        for i in range(11):
            plt.text(i,0,acts[actsiByState[i]])
            plt.xlim(-1,11)
            plt.ylim(-1,1)
        plt.text(2,0.2,"Policy for Zero Velocity")
        plt.axis("off")
        plt.subplot(4,3,4)
        plt.plot(rtrace[:trial+1],alpha=0.5)
        #plt.plot(np.convolve(rtrace[:trial+1],np.array([0.02]*50),mode='valid'))
        binSize = 20
        if trial+1 > binSize:
            # Calculate mean of every bin of binSize reinforcement values
            smoothed = np.mean(rtrace[:int(trial/binSize)*binSize].reshape((int(trial/binSize),binSize)),axis=1)
            plt.plot(np.arange(1,1+int(trial/binSize))*binSize,smoothed)
        plt.ylabel("Mean reinforcement")
        plt.subplot(4,3,5)
        plt.plot(X[:,0],X[:,1])
        plt.plot(X[0,0],X[0,1],'o')
        plt.xlabel("$x$")
        plt.ylabel("$\dot{x}$")
        plt.fill_between([4,6],[-5,-5],[5,5],color="red",alpha=0.3)
        plt.xlim(-1,11)
        plt.ylim(-5,5)
        plt.subplot(4,3,6)
        net.draw(["$x$","$\dot{x}$","$a$"],["Q"])
    
        plt.subplot(4,3,7)
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
        plt.subplot(4,3,8)
        acts = np.array(validActions)[np.argmax(qs,axis=1)].reshape(xs.shape)
        cs = plt.contourf(xs,ys,acts,[-2, -0.5, 0.5, 2])
        plt.colorbar(cs)
        plt.xlabel("$x$")
        plt.ylabel("$\dot{x}$")
        plt.title("Actions")
    
        s = plt.subplot(4,3,10)
        rect = s.get_position()
        ax = Axes3D(plt.gcf(),rect=rect)
        ax.plot_surface(xs,ys,qsmax,cstride=1,rstride=1,cmap=cm.jet,linewidth=0)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$\dot{x}$")
        #ax.set_zlabel("Max Q")
        plt.title("Max Q")
    
        s = plt.subplot(4,3,11)
        rect = s.get_position()
        ax = Axes3D(plt.gcf(),rect=rect)
        ax.plot_surface(xs,ys,acts,cstride=1,rstride=1,cmap=cm.jet,linewidth=0)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$\dot{x}$")
        #ax.set_zlabel("Action")
        plt.title("Action")

    def testIt(Qnet,nTrials,nStepsPerTrial):
        xs = np.linspace(0,10,nTrials)
        plt.subplot(4,3,12)
        for x in xs:
            s = [x,0] ## 0 velocity
            xtrace = np.zeros((nStepsPerTrial,2))
            for step in range(nStepsPerTrial):
                a,_ = Qnet.epsilonGreedy(s,validActions,0.0) # epsilon = 0
                s = nextState(s,a)
                xtrace[step,:] = s
            plt.plot(xtrace[:,0],xtrace[:,1])
            plt.xlim(-1,11)
            plt.ylim(-5,5)
            plt.plot([5,5],[-5,5],'--',alpha=0.5,lw=5)
            plt.ylabel('$\dot{x}$')
            plt.xlabel('$x$')
            plt.title('State Trajectories for $\epsilon=0$')



    plotStatus(nnet,nTrials,epsilonTrace,rtrace)
    testIt(nnet,10,500)

