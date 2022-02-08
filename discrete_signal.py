# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Lab 1: Discrete sines, cosides and complex exponentials
#
# Question 1.1: Defining the complex exponential class
#
# By Vinicius Lima (adapted from 2020 edition of the course)

import numpy as np
import math
import cmath


#######################################################################################################################
# SQUARE WAVE FUNCTION (PULSE WAVE) IS A PERIODIC FUNCTION THAT CONSTANTLY PULSES BETWEEN TWO VALUES.
# COMMON VALUES INCLUDE THE DIGIT IS SIGNAL (0,1) (-1,1) AND ( -1/2 1/2). IT IS ALSO AN ODD FUNCTION, WHICH MEANS IT IS
# SYMMETRIC AROUND THE ORIGIN
# IN FUNCTION NOTATION, THE SQUARE WAVE FUNCTION IS REPRESENTED AS FOLLOWS ( FOR A PULSE BETWEEN 1 AND -1, PULSING
# AROUND PI):
# f SQUARE(x)= {1 x <PI
#              -1 x >= PI}
#SEE: https://www.calculushowto.com/square-wave-function-pulse/
# NOTE: THE VERTICAL CONNECTORS DON'T ACTUALLY EXIST, THEY JUST EMPHASIS THAT THE GRAPH HAS A SQUARE SHAPE

#######################################################################################################################


class sqpulse():
    """
    sqpulse Generates a square spulse
    solve() generates a square pulse vector x of length N
    """
    def __init__(self, T, T0, fs):

        '''
        SEE GRAPH FROM uNITS: SAMPLING TIME AND SIGNAL DURATION
        Ts: SAMPLING TIME  IS THE LOCK TIME ELAPSED BETWEEN ROUTINE SAMPLES OF ONE PULSE (n AND n+1)
            IF THE PULSE IS 1 SECONDS AND HAS 9 SAMPLES (8 GAPS OF TIME) THEN 1 SEC/ 8 SPACES IS .125 SECONDS
            BETWEEN SAMPLE DOTS


        fs: SAMPLING FREQUENCY. THIS IS 1/Ts
            if Ts =  .125 SECONDS then fs= 1/.125 sec = 8 IS THE FREQUENCY OF TIME GAPS FOR ONE PHASE
            THEREFORE THERE ARE 8 TIME GAPS IN  ONE PHASE ( 9 SAMPLES IN ONE PHASE)


        T0: THE NON ZERO LENGTH OF THE PULSE

        '''

        """
        :param T: the duration  
        :param T0: nonzero length
        :param fs: the sampling frequency
        """
        self.T = T
        self.T0 = T0
        self.fs = fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the square pulse vector x
        :N: The sample duration
        """

        ''''
        
        M IS A DESCRETE TIME INDEX
        '''
        Ts = 1/self.fs
        N = math.floor(self.T/Ts)
        M = math.floor(self.T0/Ts)
        x = np.zeros(N)
        
        for i in range(M):
            x[i] = 1/np.sqrt(M)
        
        return x, N

class tripulse():
    """
    sqpulse Generates a triangular spulse
    solve() generates a triangular pulse vector x of length N
    """
    def __init__(self, T, T0, fs):
        """
        :param T: the duration
        :param T0: nonzero length
        :param fs: the sampling frequency
        """
        self.T = T
        self.T0 = T0
        self.fs = fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the triangular pulse vector x
        :N: The sample duration
        """
        Ts = 1/self.fs
        N = math.floor(self.T/Ts)
        M = math.floor(self.T0/Ts)
        x = np.zeros(N)
        
        for i in range(np.int(M/2)):
            x[i] = i
            x[M-i-1] = i
            
        energy = np.linalg.norm(x)
            
        for i in range(M):
            x[i] = x[i]/energy        
        
        return x, N
    
    
class sqwave():
    """
    sqpulse Generates a square spulse
    solve() generates a square pulse vector x of length N
    """
    def __init__(self, T, f0, fs):
        """
        :param T: the duration
        :param T0: nonzero length
        :param fs: the sampling frequency
        """
        self.T = T
        self.f0 = f0
        self.fs = fs
        self.N = T*fs
                
    def solve(self):
        """
        :return: [x,N]
        :x: the square pulse vector x
        :N: The sample duration
        """
        n = np.arange(self.N)
        x = np.sign(np.cos(2*cmath.pi*self.f0/self.fs*n))
        
        return x, self.N
