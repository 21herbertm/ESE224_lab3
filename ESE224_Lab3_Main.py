# ESE 224 --- Signal and Information Processing
#
#
# Lab 3: Inverse Discrete Fourier Transform


# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.io.wavfile import write


''' NOTE: YOU MAY NEED TO INSTALL PYGOBJECT AND PYCARIO AND A BUNCH OF OTHER LIBS 
SEE https://stackoverflow.com/questions/33862049/python-cannot-install-pygobject'''
import sounddevice
import playsound
import soundfile as sf
# Local files
import discrete_signal

######################################CLASSES##############################################################
'''
THIS CLASS HAS ATTRIBUTES 
    INPUT:  SIGNAL              x 
            LENGTH              N
            SAMPLING FREQUENCY  fs
            COMPRESSION TARGET  K
    FUNCTION TO CALC (solve): VECTOR WITH THE K LARGEST DFT COEFFIENCTS AND SET OF FRQUENCIES OF THE COEFFICIENTS
    
    NOTE: EACH OF THE COEFFICIENTS THAT IS KEPT REQUIRES STORAGE OF TWO NUMBERS, THE COEFFICIENT AND THE FREQUENCY.
'''
class dft_K_q16():

    '''INITIALIZE THE ATTRIBUTES OF THE CLASS SIGNAL X, LENGTH N, SAMPLING FREQUENCY fs, COMPRESSION TARGET K'''


    def __init__(self, x, fs, K):
        """
        :param X: Input DFT X
        :param fs: Input integer fs contains the sample frequency
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the iDFT.
        """
        self.x=x
        self.fs=fs
        self.N=len(self.x)
        self.K=K

    def solve(self):
        """
        \\\\\ METHOD: Compute the iDFT with truncated K coefficients of largest energy
        :fk the real frequencies
        """

        '''GENERAL PYTHON: NUMPY FUNCTION ZEROS WILL RETURN A NEW ARRAY OF A GIVEN TYPE AND SHAPE FILLED WITH ZEROS
            np.zeros((5,), dtype=int)
            array([0, 0, 0, 0, 0])
        '''

        #CREATE ARRAYS ZEROED OUT

        X=np.zeros(self.N, dtype=np.complex)
        E=np.zeros(self.N)
        X_K=np.zeros(self.K, dtype=np.complex)
        index=np.zeros(self.K)



        #???X IS AN ARRAY OF LENGTH N^2 OF
        #???THE INDEX IS K????
        #????PUT IN THE ARRAY INDEX K THE VALUE OF THE CALCULATED ENERGY
        for k in range(self.N):
            for n in range(self.N):
                X[k] = X[k]+1/np.sqrt(self.N)*self.x[n]*np.exp(-1j*2*cmath.pi*k*n/self.N)
        '''
        GENERAL PYTHON: NUMPY ARRANGE: THE ARANGE() FUNCTION IS USED TO GET EVENLY SPACED VALUES WITHIN A 
        GIVEN INTERVAL.numpy.arange([start, ]stop, [step, ]dtype=None)
        >>> np.arange(5.0)
        >>> array([ 0.,  1.,  2.,  3.,  4.])
        SEE https://www.w3resource.com/numpy/array-creation/arange.php 
        '''


        '''THE CORRESPONDING SET OF FREQUENCIES(fk)  AT WHICH  LARGEST DFT COEFFICIENTS ARE OBSERVED'''
        fk= np.arange(self.N)/self.N*self.fs
                
        for k in range(self.N):
            E[k]=abs(X[k])



        ''' NOW FIND VECTOR WITH THE K LARGEST DFT COEFFICIENTS (X_K)  AND
        THE CORRESPONDING SET OF FREQUENCIES(fk)  AT WHICH THESE COEFFICIENTS ARE OBSERVED.
        
        '''


        '''
        GENERAL PYTHON: NUMPY ARGSORT IS USED TO PERFORM AN INDIRECT SORT ALONG THE GIVEN AXIS USING THE ALGORITHM 
        SPECIFIED BY THE KIND KEYWORD. IT RETURNS AN ARRAY OF INDICES OF THE SAME SHAPE AS ARR THAT 
        THAT WOULD SORT THE ARRAY.'''
        index_temp=np.argsort(-E)        
        index=index_temp[0:self.K]
        index = index[:,np.newaxis]
        X_K=X[index]

        """ GENERAL PYTHON: NUMPY.CONCATENATE() FUNCTION CONCATENATE A SEQUENCE OF ARRAYS ALONG AN EXISTING AXIS."""
        X_K=np.concatenate((X_K,index),axis=1)




        return X_K, fk
    
    
'''
CLASS FOR DFT CALCULATIONS
    ATTRIBUTES:  x: INPUT VECTOR X OF THE DISCRETE SIGNAL
                 fs: INPUT INTEGER FS CONTAINS THE SAMPLE FREQUENCY
                 K: INPUT POSITIVE INTEGER THAT DETERMINES THE NUMBER OF COEFFIENTS USED TO CALCULATE THE DFT. 
                    If K IS NOT PROVIDED, K=length(x). 
                f: ARE THE FREQUENCIES STARTING AT F=0 AND X ARE THE CORRESPONDING FREQUENCY COMPONENTS. 
                f_c: VECTOR CONTAINING THE FREQUENCIES SUCH THAT f_c=0 is at the center
                X_c: CONTAINS THE FREQUENCY COMPONENTS CORRESPONDING TO f_c.
                    
'''
class dft():


    def __init__(self, x, fs, K=None):
        """
        :param x: Input vector x contains the discrete signal
        :param fs: Input integer fs contains the sample frequency
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. If K is not provided, K=length(x).
        """
    # START: SANITY CHECK OF INPUTS.
        if (type(fs) != int) or (fs<=0):
            raise NameError('The frequency fs should be a positive integer.')
        if not isinstance(x, np. ndarray):
            raise NameError('The input signal x must be a numpy array.')
        if isinstance(x, np. ndarray):
            if x.ndim!=1:
                raise NameError('The input signal x must be a numpy vector array.')
        self.x=x
        self.fs=fs
        self.N=len(x)
        if K == None:
            K = len(self.x)
        # START: SANITY CHECK OF INPUTS.
        if (type(K) != int) or (K <= 0) or (K < 0):
            raise NameError('K should be a positive integer.')
        self.K=K

        '''
            GENERAL PYTHON: NUMPY ARRANGE: THE ARANGE() FUNCTION IS USED TO GET EVENLY SPACED VALUES WITHIN A 
            GIVEN INTERVAL.numpy.arange([start, ]stop, [step, ]dtype=None)
            >>> np.arange(5.0)
            >>> array([ 0.,  1.,  2.,  3.,  4.])
            SEE https://www.w3resource.com/numpy/array-creation/arange.php 
            '''

        '''CREATE FRQUENCY VECTOR FROM 0 TO K USING A STEP OF 1'''
        self.f=np.arange(self.N)*self.fs/self.N


        ''' NEED TO HANDLE CONDITIONS WHERE VECTOR CONTAINING THE FREQUENCIES SUCH THAT f_c=0 is at the center'''
        ''' FLOOR ROUNDS DOWN, CEILING ROUNDS UP'''
        self.f_c=np.arange(-np.ceil(self.N/2)+1,np.floor(self.N/2)+1)*self.fs/self.N


        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
    def changeK(self,K):
        """
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. This function changes the attribute K of the class.
        """
        if (type(K) != int) or (K <= 0) or (K <  0):
            raise NameError('K should be a positive integer.')
        old_K=self.K
        self.K=K
        self.f=np.arange(self.K)*self.fs/self.K # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(K/2)+1,np.floor(self.K/2)+1)*self.fs/self.K
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
        print('The value of K was succefully change from %d to %d'%(old_K,self.K))
        pass

    def solve_using_numpy_fft(self):

        X=np.fft.fft(self.x,self.N)/np.sqrt(self.N);
        # \\\\\ CENTER FFT.
        X_c=np.roll(X,np.int(np.ceil(self.N/2-1))) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]
    
    # def solve4(self):
    #     """
    #     \\\\\ METHOD 3: Built-in fft() function
    #     Even though the matrix form is fast, it is still not fast enough for large
    #     signals x. For that, it is better to use the built in fft() function which is
    #     the optimal way to compute a dft. Besides, it is really easy to code.
    #     """
    #     X=np.fft.fft(self.x,self.N)/np.sqrt(self.N);
    #     # \\\\\ CENTER FFT.
    #     X_c=np.roll(X,np.int(np.ceil(self.N/2-1))) # Circularly shift X to get it centered in f_c==0
    #
    #     X_K = X[0:(self.K+1)]
    #
    #     return [self.f,X_K,self.f_c,X_c]



'''
QUESTION 1.1 COMPUTATION OF THE IDFT. CONSIDER A DFT X CORRESPONDING TO A
REAL SIGNAL OF EVEN DURATION N AND ASSUME THAT WE ARE GIVEN THE N/2 +
1 COEFFICIENTS CORRESPONDING TO FREQUENCIES K = 0, 1, . . . , N/2. 

CREATE A PYTHON CLASS THAT TAKES THESE N/2 COEFFICIENTS AS INPUT, AS WELL AS THE ASSOCIATED SAMPLING FREQUENCY FS
, AND RETURNS THE IDFT X = F −1
(X) OF THE GIVEN X. RETURN ALSO A VECTOR OF REAL TIMES ASSOCIATED WITH THE SIGNAL
SAMPLES.
'''
class idft_q11():
    """
    idft Inverse Discrete Fourier transform.
    """


    '''INITIALIZES THE CLASS WITH 
        X DFT ACTUAL SIGNAL
        fs  SAMPLE FREQUENY
        N IS THE DURATION
        '''
    def __init__(self, X, fs):
        """
        :PARAM X: INPUT DFT X: THE ACTUAL SIGNAL
        :PARAM FS: INPUT INTEGER fs CONTAINS THE SAMPLE FREQUENCY
        """
        self.X_dft_actualsignal=X
        self.fs_samplefreqency=fs

        ''' N IS THE DURATION'''
        self.N_duriation=2*(len(self.X_dft_actualsignal)-1)


    '''
        CALCULATE THE IDFT WITH TRUNCATED N/2+1 COEFFICIENTS
        RETURN THE  
            IDFT VECTOR:x 
            CORRESPONDING Treal vector: {Treal}
    '''
    def solve(self):


        '''GENERAL PYTHON: NUMPY FUNCTION ZEROS WILL RETURN A NEW ARRAY OF A GIVEN TYPE AND SHAPE FILLED WITH ZEROS
                    np.zeros((5,), dtype=int)
                    array([0, 0, 0, 0, 0])
        '''
        x=np.zeros(self.N_duriation)


        ''' GENERAL PYTHON:  LENGTH OF ALL N '''
        for n in range(self.N_duriation):
            x[n] = 1/np.sqrt(self.N_duriation)*self.X_dft_actualsignal[0]*np.exp(1j*2*cmath.pi*0*n/self.N_duriation)
            for k in range(1,int(self.N_duriation/2)):
                x[n] = x[n] + 1/np.sqrt(self.N_duriation)*self.X_dft_actualsignal[k]*np.exp(1j*2*cmath.pi*k*n/self.N_duriation)
                x[n] = x[n] + 1/np.sqrt(self.N_duriation)*np.conj(self.X_dft_actualsignal[k])*np.exp(-1j*2*cmath.pi*k*n/self.N_duriation)
            x[n] = x[n] + 1/np.sqrt(self.N_duriation)*self.X_dft_actualsignal[int(self.N_duriation/2)]*np.exp(1j*2*cmath.pi*(int(self.N_duriation/2))*n/self.N_duriation)



        #ALTERNATIVE METHOD USING NUMPY
        alt_x=np.fft.ifft(self.X,self.N)*np.sqrt(self.N)

        alt_Ts= 1/self.fs
        alt_Treal= np.arange(self.N)*alt_Ts




        Ts= 1/self.fs_samplefreqency

        '''
          GENERAL PYTHON: NUMPY ARRANGE: THE ARANGE() FUNCTION IS USED TO GET EVENLY SPACED VALUES WITHIN A 
          GIVEN INTERVAL.numpy.arange([start, ]stop, [step, ]dtype=None)
          >>> np.arange(5.0)
          >>> array([ 0.,  1.,  2.,  3.,  4.])
          SEE https://www.w3resource.com/numpy/array-creation/arange.php 
          '''
        Treal= np.arange(self.N)*Ts

        print(f" Question 1.1 the idft vector: {x}")
        print(f" Question 1.1 the corresponding Treal vector: {Treal}")
        return x, Treal


'''
CLASS FOR INVERSE DFT CALCULATIONS
    ATTRIBUTES:  x: INPUT VECTOR DFT X 
                 fs: INPUT INTEGER FS CONTAINS THE SAMPLE FREQUENCY
                 N: THE NUMBER OF TOTAL SIGNAL SAMPLES N
                 K: INPUT POSITIVE INTEGER THAT DETERMINES THE NUMBER OF COEFFIENTS USED TO CALCULATE THE DFT. 
                    If K IS NOT PROVIDED, K=length(x). 
                f: ARE THE FREQUENCIES STARTING AT F=0 AND X ARE THE CORRESPONDING FREQUENCY COMPONENTS. 
                f_c: VECTOR CONTAINING THE FREQUENCIES SUCH THAT f_c=0 is at the center
                X_c: CONTAINS THE FREQUENCY COMPONENTS CORRESPONDING TO f_c.

'''
class idft():
    def __init__(self, X, fs, N, K=None):
        self.X=X
        self.fs=fs
        self.N=N 
        self.K=K
        if self.K==None:
            self.K=int(len(X)/2)-1

    '''
    METHOD TO COMPUTE THE iDFT THE HARD WAY  WITH TRUNCATED K COEEFFICIENTS 
        RETURNS:    x: iDFT X OF THE DURATION N 
                    Treal: THE REAL TIME VECOR OF SIZE N '''
    def solve_using_trunc_K(self):

        #CREATE A VECTOR FILLED WITH ZEROS OF SIZE N
        x=np.zeros(self.N)


        #LOOP OVER THE LENGTH OF N AND FILL IN THE iDFT ONE BY ONE
        for n in range(self.N):
            x[n] = 1/np.sqrt(self.N)*self.X[0]*np.exp(1j*2*cmath.pi*0*n/self.N)
            for k in range(1,self.K+1):

                #GENERAL PYTHON: NUMPY CONJ OR CONJUGATE IS THE COMPLEX CONJUGATE OF A COMPLEX NUMBER IS OBTAINED
                # BY CHANGING THE SIGN OF ITS IMAGINARY PART.
                x[n] = x[n] + 1/np.sqrt(self.N)*self.X[k]*np.exp(1j*2*cmath.pi*k*n/self.N)
                x[n] = x[n] + 1/np.sqrt(self.N)*np.conj(self.X[k])*np.exp(-1j*2*cmath.pi*k*n/self.N)
                
        Ts= 1/self.fs

        #CALC THE TS
        Treal= np.arange(self.N)*Ts

        return x, Treal

    '''
      METHOD TO COMPUTE THE iDFT WITH EASY NUMPY FUNCTION ifft
          RETURNS:    x: iDFT X OF THE DURATION N 
                      Treal: THE REAL TIME VECOR OF SIZE N '''

    def solve_using_numpy_ifft(self):
        x=np.fft.ifft(self.X,self.N)*np.sqrt(self.N)
                
        Ts= 1/self.fs
        Treal= np.arange(self.N)*Ts

        return x, Treal    
    
'''
QUESTION 1.8 SIGNAL RECONSTRUCTION. CREATE A PYTHON CLASS THAT TAKES AS INPUT THE 1.8
RECONSTRUCTION OUTPUT SIGNAL OF THE CLASS
IN PART 1.6 AND RECONSTRUCTS THE ORIGINAL SIGNAL X. 
YOU CAN USE WHAT YOU SOLVED IN PART 1.1, AND PART 1.2 TO HELP SOLVE THIS PART.
'''
'''
CLASS FOR SIGNAL RECONSTRUCTION
    ATTRIBUTES:  X_k: INPUT TRUNCATED VECTOR DFT X 
                 fk:  CORRESPONDING SET OF FREQUENCIES(fk) 
                 N: THE NUMBER OF TOTAL SIGNAL SAMPLES N
                 K: INPUT POSITIVE INTEGER THAT DETERMINES THE NUMBER OF COEFFIENTS USED TO CALCULATE THE DFT. 
                    If K IS NOT PROVIDED, K=length(x). 
            

'''
class signal_Recon_K_q18():

    def __init__(self, X_k, fk):

        self.X=X_k
        self.fk=fk
        self.N=len(fk)
        self.K=self.X.shape[0]

    '''
    SOLVE FOR iDFT WITH TRUNCATED K LARGEST ENERGY
    '''
    def solve_trunc_K(self):

        x=np.zeros(self.N)
        
        for n in range(self.N):
            for k in range(self.K):
                x[n] = x[n]+1/np.sqrt(self.N)*self.X[k,0]*np.exp(1j*2*cmath.pi*self.X[k,1]*n/self.N)
            
        return x

'''
BASIC CLASS TO RECORD SOUND FROM THE MICROPHONE FOR 
    ATTRIBUTES: T:  TIME SAMPLED AT A FREQUENCY fs
                fs: FREQUENCY 
    OUTPUT: x: THE TRIANGLE PULSE VECTOR
            N: SAMPLE DURATION
            WAV FILE 
    '''

class microphoneRecordSample():

    def __init__(self, T, fs):
        self.T = T
        self.fs = fs

    def createWavFile_vector(self,filename):
        """

        :x: the triangular pulse vector x
        :N: The sample duration
        """
        print('start recording')
        # voicerecording IS  IS [x, N]
        voicerecording = sounddevice.rec(int(self.T * self.fs), self.fs, 1)
        sounddevice.wait()  # Wait until recording is finished
        print('end recording')
        write(filename, self.fs, voicerecording)  # Save as WAV file

        '''return: [x,N]'''
        playsound.playsound(filename)
        return voicerecording


######################################QUESTIONS#############################################################
"""
QUESTION 1.3 RECONSTRUCT A SQUARE PULSE
1) CREATE A SQUARE PULSE WITH A DURATION OF 
                DURATION            T=  32S
                SAMPLE FREQUENCY OF fs= 8Hz
                NON-0 LENGTH        T0= 4s
                
2) CALC THE DFT

3) USE CLASS IN q12 TO CREATE SUCCESSIVE RECONSTRUCTION OF THE PULSE

4) COMPUTE THE ENERGY OF THE DIFF SIGNALS x AND xk (THE ENERGY SHOULD DECREASE FOR INCREASING K)

5) CONDUCT THIS FOR RESULTS:
                K=2, K=4,K=8,K=16,K=32
                
6) REPEAT THIS PROCESS FOR T0=2      
"""
def recog_square_pulse_q13(T, fs, T0, K):

    #Question 1.3 G
    
    sp = discrete_signal.sqpulse(T, T0, fs)    #generate square pulse signal
    x, N = sp.solve()
    DFT = dft(x,fs)      #compute the DFT
    [f,X,f_c,X_c] = DFT.solve_using_numpy_fft()



    ''' CALL INTERNAL CLASS TO CREATE AN IDFT OBJECT'''
    iDFT = idft(X, fs, N, K)      #compute the iDFT

    '''CALL OBJECT IDFT  FUNCTION solve_using_trunc_K TO GET THE xhat_K and Treal'''
    xhat_K, Treal = iDFT.solve_using_trunc_K()




    ''' GET THE ENERGY DIFF BETWEEN x and xhat_K'''
    x_diff = x - xhat_K
    energy_diff = np.linalg.norm(x_diff)*np.linalg.norm(x_diff)
    print(energy_diff)
    
    # Plot

    origsig_title_str= f"Q1.3 Square_DFT Original Signal and its DFT for T:{T} fs:{fs}, T0:{T0}, K:{K}"
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle(origsig_title_str)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c, abs(X_c))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('DFT')
    plt.show()    
    fig.savefig("square_DFT.png", bbox_inches='tight')
    
    plt.figure()
    plt.grid(True)
    recon_title_str = f"Q:1.3 Square_reconstructed Signal T:{T} fs:{fs}, T0:{T0}, K:{K}"
    plt.title(recon_title_str)
    plt.plot(Treal, xhat_K)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.savefig('Square_reconstructed.png')
    plt.show()
#


'''QUESTION 1.4 RECONSTRUCTION OF A TRIANGULAR PULSE. GENERATE A TRIANGULAR PULSE3
OF DURATION T = 32S SAMPLED AT A RATE FS = 8HZ AND LENGTH T0 = 4S
AND COMPUTE ITS DFT. USE THE CLASS IN PART 1.2 TO CREATE SUCCESSIVE RECONSTRUCTIONS OF THE PULSE. 
COMPUTE THE ENERGY OF THE DIFFERENCE BETWEEN THE
SIGNALS X AND X˜K. REPORT YOUR RESULTS FOR K = 2, K = 4, K = 8, AND K = 16
K = 32. THIS PULSE SHOULD BE EASIER TO RECONSTRUCT THAN THE SQUARE PULSE.
IS THAT TRUE?'''
def recog_triangle_pulse_q14(T, fs, T0, K):

    
    tp = discrete_signal.tripulse(T, T0, fs)    
    x, N = tp.solve()
    DFT = dft(x,fs)
    [f,X,f_c,X_c] = DFT.solve_using_numpy_fft()

    iDFT = idft(X, fs, N, K)
    xhat_K, Treal = iDFT.solve_using_trunc_K()
    
    x_diff = x - xhat_K
    energy_diff = np.linalg.norm(x_diff)*np.linalg.norm(x_diff)
    print(energy_diff)
    
    # Plot
    origsig_title_str = f"Q:1.4 Triangular_DFT Original Signal and its DFT for T:{T} fs:{fs}, T0:{T0}, K:{K}"
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle(origsig_title_str )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(f_c, abs(X_c))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('DFT')
    plt.show() 
    fig.savefig("triangular_DFT.png", bbox_inches='tight')

    recon_title_str = f"Q 1.4 Triangular_reconstructed Signal T:{T} fs:{fs}, T0:{T0}, K:{K}"
    plt.figure()
    plt.grid(True)
    plt.title(recon_title_str)
    plt.plot(Treal, xhat_K)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.savefig('Triangular_reconstructed.png')
    plt.show()

'''
QUESTION 1.8 SIGNAL RECONSTRUCTION. CREATE A PYTHON CLASS THAT TAKES AS INPUT THE
OUTPUT OF THE CLASS IN PART 1.6 AND RECONSTRUCTS THE ORIGINAL SIGNAL X. HINT:
YOU CAN USE WHAT YOU SOLVED IN PART 1.1, AND PART 1.2 TO HELP SOLVE THIS PART.
    INPUTS:     DURATION            T
                SAMPLE FREQUENCY OF f
                NON-0 LENGTH        T0
                FREQUENCY           K
    OUTPUTS: PLOT OF THE ORIG (1.6) AND RECONSTRUCTED SIGNALS(1.8)
    
'''
def signal_recog_q18(T, fs, T0, K):

    #GET A SQUARE PULSE AND GET THE N: The sample duration AND the square pulse vector
    local_square_pulse = discrete_signal.sqpulse(T, T0, fs)
    x, N = local_square_pulse.solve()


    #GET THE OUTPUT FROM Q16 AND SOLVE THE DFT X
    DFT_K_q16 = dft_K_q16(x,fs,K)
    X_K_q16, fk_q16 = DFT_K_q16.solve()
    
    iDFT_recog = signal_Recon_K_q18(X_K_q16, fk_q16)
    xhat_K_recog = iDFT_recog.solve_trunc_K()
    
    Ts= 1/fs
    Treal= np.arange(N)*Ts
    
    # PLOT THE RECONSTRUCTED THE SIGNAL FROM AND THE ORIGNAL SIGNAL FROM 1.6
    fig, axis = plt.subplots(2)
    axis[0].grid()
    axis[1].grid()

    fig.suptitle(f'Q:1.8 Original Signal FROM Q1.6  and Reconstructed Signal Q1.8  T:{T} fs:{fs}, T0:{T0}, K:{K}' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axis[0].plot(Treal, x)
    axis[0].set_xlabel('Time (s)')
    axis[0].set_ylabel('Signal')
    axis[1].plot(Treal, xhat_K_recog)
    axis[1].set_xlabel('Time (s)')
    axis[1].set_ylabel('Signal')
    plt.show() 


'''
QUESTION 1.9 
COMPRESSION AND RECONSTRUCTION OF A SQUARE WAVE. GENERATE A
SQUARE WAVE OF DURATION T = 32S SAMPLED AT A RATE FS = 8HZ AND FREQUENCY 0.25HZ. 
COMPRESS AND RECONSTRUCT THIS WAVE USING THE FUNCTIONS IN PARTS 1.6 AND 1.8. 
TRY DIFFERENT COMPRESSION TARGETS AND REPORT THE ENERGY OF THE ERROR SIGNAL FOR K = 2, K = 4, K = 8 AND K = 16. 
THIS PROBLEM SHOULD TEACH YOU THAT A SQUARE WAVE CAN BE APPROXIMATED BETTER THAN A
SQUARE PULSE IF YOU KEEP THE SAME NUMBER OF COEFFICIENTS. THIS SHOULD
BE THE CASE BECAUSE THE SQUARE WAVE LOOKS THE SAME AT ALL POINTS, BUT THE
SQUARE PULSE DOESN’T. EXPLAIN THIS STATEMENT
'''
'''
    FUNCTION TO COMPRESS AND RECOMPRESS A SQUARE WAVE 
   INPUTS:     DURATION            T
                SAMPLE FREQUENCY OF f
                NON-0 LENGTH        T0
                FREQUENCY           K
    OUTPUTS: PLOT OF THE ORIG (1.6) AND RECONSTRUCTED SIGNALS(1.8)'''
def compression_resconst_square_wave_q19(T, fs, f0, K):
    
    
    # GET A SQUARE WAVE AND GET THE N: The sample duration AND the square pulse vector
    local_square_wave = discrete_signal.sqwave(T, f0, fs)    
    x, N = local_square_wave.solve()
    DFT = dft(x,fs)
    [f,X,f_c,X_c] = DFT.solve_using_numpy_fft()
    
    DFT_K = dft_K_q16(x,fs,K)
    X_K, fk = DFT_K.solve()
    
    iDFT = signal_Recon_K_q18(X_K, fk)
    xhat_K = iDFT.solve_trunc_K()
    
    Ts= 1/fs
    Treal= np.arange(N)*Ts
    
    # Plot
    fig, axis = plt.subplots(2)
    axis[0].grid()
    axis[1].grid()
    fig.suptitle(f'Q:1.9 Original Signal FROM SQUARE WAVE and its DFT T:{T} fs:{fs},K:{K}' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axis[0].plot(Treal, x)
    axis[0].set_xlabel('Time (s)')
    axis[0].set_ylabel('Signal')
    axis[1].plot(f_c, abs(X_c))
    axis[1].set_xlabel('Frequency (Hz)')
    axis[1].set_ylabel('DFT')
    plt.show()
    fig.savefig("Square_Wave_DFT.png", bbox_inches='tight')
    
    plt.figure()
    plt.grid(True)
    plt.title(f'Q:1.9 Reconstructed SQUARE WAVE Signal  T:{T} fs:{fs}, K:{K}')
    plt.plot(Treal, xhat_K)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.savefig('Square_Wave_reconstructed.png')
    plt.show()


'''
QUESTION 2.1 SIMPLE FUNCTION TO 
        RECORD, 
        GRAPH, 
        COMPUTE DFT 
        PLAY BACK 
        
        INPUTS  T:DURATION
                fs:SAMPLE FREQUENCY
'''
def record_graph_play_q21(T, fs):


    myvoice = microphoneRecordSample(T, fs)  
    x = myvoice.createWavFile_vector('mywavefile_q21.wav').reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve_using_numpy_fft()
    
    Ts= 1/fs
    Treal= np.arange(N)*Ts
    
    # Plot
    fig, axis = plt.subplots(2)
    axis[0].grid()
    axis[1].grid()
    fig.suptitle(f'Q:2.1 Original Voice and its DFT  T:{T} fs:{fs}' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axis[0].plot(Treal, x)
    axis[0].set_xlabel('Time (s)')
    axis[0].set_ylabel('Signal')
    axis[1].plot(f_c, X_c)
    axis[1].set_xlabel('Frequency (Hz)')
    axis[1].set_ylabel('DFT')
    plt.show()     
    fig.savefig("Recorded_Voice_and_DFT.png", bbox_inches='tight')


'''
QUESTION 2.2 VOICE COMPRESSION. THE 5 SECOND RECORDING OF YOUR VOICE AT SAMPLING FREQUENCY FS = 20KHZ 
IS COMPOSED OF 100,000 SAMPLES. 
USE THE DFT AND IDFT TO COMPRESS YOUR VOICE BY A FACTOR OF 2, I.E., STORE K =
50, 000 NUMBERS INSTEAD OF 100,000, A FACTOR OF 4, (STORE K = 25, 000 NUMBERS), A FACTOR OF 8 
(STORE K = 12, 500 NUMBERS), AND SO ON. KEEP COMPRESSING UNTIL THE SENTENCE YOU SPOKE BECOMES UNRECOGNIZABLE. YOU CAN
PERFORM THIS COMPRESSION BY KEEPING THE FIRST K DFT COEFFICIENTS OR THE
LARGEST K/2 DFT COEFFICIENTS. WHICH ONE WORKS BETTER?
'''
'''
QUESTION 2.2 COMPRESS VOICE  USING DFT AND IDFT BY FACTORS OF 2
 

        INPUTS  T:DURATION
                fs:SAMPLE FREQUENCY
'''


def q_22_1(T, fs, gamma):
    """
    Question 2.2_first K DFT coefficients

    """

    myvoice = microphoneRecordSample(T, fs)
    x = myvoice.createWavFile_vector('mywavefile_q22_1.wav').reshape(T * fs)
    N = len(x)

    DFT = dft(x, fs)
    [freqs, X, f_c, X_c] = DFT.solve_using_numpy_fft()

    truncated_sample = int(N * gamma)
    X_truncated = np.zeros(N, dtype=np.complex)
    X_truncated[0:truncated_sample] = X[0:truncated_sample]

    iDFT = idft(X_truncated, fs, N)
    xhat_K, Treal = iDFT.solve_using_numpy_ifft()

    write('myvoice_truncated.wav', fs, xhat_K.real)
    playsound.playsound('myvoice_truncated.wav')
    # Plot
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Original Voice and Reconstructed Voice')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axs[0].plot(Treal, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')
    axs[1].plot(Treal, xhat_K)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Signal')
    plt.show()
    fig.savefig("Reconstructed_Voice.png", bbox_inches='tight')

def voice_compression_q22_1(T, fs, gamma,x):


    #myvoice = microphoneRecordSample(T, fs)
    #x = myvoice.createWavFile_vector('mywavefile_q22_1.wav').reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve_using_numpy_fft()

    #for gamma in gamma_list :
    truncated_sample = int(N*gamma)
    X_truncated = np.zeros(N, dtype=np.complex)
    X_truncated[0:truncated_sample] = X[0:truncated_sample]

    iDFT = idft(X_truncated, fs, N)
    xhat_K, Treal = iDFT.solve_using_numpy_ifft()

    filename = 'myvoice_truncated_q22_1' + str(gamma) + '.wav'
    write(filename, fs, xhat_K.real)

    playsound.playsound(filename)
    # Plot
    fig, axis = plt.subplots(2)
    axis[0].grid()
    axis[1].grid()
    fig.suptitle(f'Q:22_1 Original Voice and Reconstructed Voice T:{T} fs:{fs}, gamma:{gamma}' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axis[0].plot(Treal, x)
    axis[0].set_xlabel('Time (s)')
    axis[0].set_ylabel('Signal')
    axis[1].plot(Treal, xhat_K)
    axis[1].set_xlabel('Time (s)')
    axis[1].set_ylabel('Signal')
    plt.show()
    fig.savefig("Reconstructed_Voice.png", bbox_inches='tight')
    
def voice_compression_largest_K_q22_2(T, fs, gamma):
    """
    Question 2.2 _K DFT coefficients with largest energy
    """
    
    myvoice = microphoneRecordSample(T, fs)  
    x = myvoice.createWavFile_vector('mywavefile_q22_2.wav').reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve_using_numpy_fft()
    
    truncated_sample = int(N*gamma/2)
    X_truncated = np.zeros(N, dtype=np.complex)
    E=np.zeros(N)
    for k in range(N):
        E[k]=abs(X[k])
    index_temp=np.argsort(-E)
    index=index_temp[0:truncated_sample]
    X_truncated[index]=X[index]
    
    iDFT = idft(X_truncated, fs, N)
    xhat_K, Treal = iDFT.solve_using_numpy_ifft()
    
    # Plot
    fig, axis = plt.subplots(2)
    axis[0].grid()
    axis[1].grid()
    fig.suptitle(f'Q:22.2 Original Voice and Reconstructed Voice  T:{T} fs:{fs}, gamma:{gamma}' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axis[0].plot(Treal, x)
    axis[0].set_xlabel('Time (s)')
    axis[0].set_ylabel('Signal')
    axis[1].plot(Treal, xhat_K)
    axis[1].set_xlabel('Time (s)')
    axis[1].set_ylabel('Signal')
    plt.show()     
    fig.savefig("Reconstructed_Voice_Largest.png", bbox_inches='tight')


'''
QUESTION 2.3 VOICE MASKING. SAY THAT YOU AND YOUR PARTNER SPEAK THE SAME SENTENCE. 
THE DFTS OF THE RESPECTIVE RECORDING WILL BE SIMILAR BECAUSE IT’S
THE SAME SENTENCE, BUT ALSO DIFFERENT, BECAUSE YOUR VOICES ARE DIFFERENT.
YOU CAN USE THIS FACT TO MASK YOUR VOICE BY MODIFYING ITS SPECTRUM, I.E.,
BY INCREASING THE CONTRIBUTION OF SOME FREQUENCIES AND DECREASING THE
CONTRIBUTIONS OF OTHERS. DESIGN A PROJECT TO RECORD YOUR VOICE, MAKE IT
UNRECOGNIZABLE BUT INTELLIGIBLE, AND PLAY IT IN THE SPEAKERS.
'''''
def voice_masking_q23(T, fs, threshold):
    """
    Question 2.3
    
    """
    myvoice = microphoneRecordSample(T, fs)  
    x = myvoice.createWavFile_vector('mywavefile_q23.wav').reshape(T*fs)
    N = len(x)

    DFT = dft(x,fs)
    [freqs, X, f_c, X_c] = DFT.solve_using_numpy_fft()
    
    for k in range(N):
        E=abs(X[k])
        if E > threshold:
            X[k] = 0
    
    iDFT = idft(X, fs, N)
    xhat_K, Treal = iDFT.solve_using_numpy_ifft()
    
    # Plot
    fig, axis = plt.subplots(2)
    axis[0].grid()
    axis[1].grid()
    fig.suptitle(f'Q:2.3 Original Signal and Reconstructed Signal  T:{T} fs:{fs}, threshold:{threshold} ')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axis[0].plot(Treal, x)
    axis[0].set_xlabel('Time (s)')
    axis[0].set_ylabel('Signal')
    axis[1].plot(Treal, xhat_K)
    axis[1].set_xlabel('Time (s)')
    axis[1].set_ylabel('Signal')
    plt.show() 
    fig.savefig("Reconstructed_Voice_Masking.png", bbox_inches='tight')


'''   
QUESTION 2.4
BETTER VOICE COMPRESSION. DESIGN A PROJECT THAT DIVIDES YOUR SPEECH
IN CHUNKS OF 100MS, AND COMPRESSES EACH OF THE CHUNKS BY A GIVEN FACTOR Γ. 
DESIGN THE INVERSE SYSTEM THAT TAKES THE COMPRESSED CHUNKS, RECONSTRUCTS THE INDIVIDUAL SPEECH
PIECES, STITCHES THEM TOGETHER AND PLAYS
THEM BACK IN THE SPEAKERS. YOU HAVE JUST DESIGNED A RUDIMENTARY MP3
COMPRESSOR AND PLAYER. TRY THIS OUT FOR DIFFERENT VALUES OF Γ. PUSH Γ TO
THE LARGEST POSSIBLE COMPRESSION FACTOR
'''
def voice_comparison_q24(T, fs, gamma):

    
    myvoice = microphoneRecordSample(T, fs)  
    x = myvoice.createWavFile_vector('mywavefile_q24.wav').reshape(T*fs)
    N = len(x)
    chunks = T*10
    chunk_sample = 0.1*fs
    x_recon = []
    
    for i in range(chunks):        
        current_chunk = x[int(i*chunk_sample):int(i*chunk_sample+chunk_sample)]
        n = len(current_chunk)
        DFT_chunk = dft(current_chunk, fs)
        [freqs_chunk, X_chunk, f_c_chunk, X_c_chunk] = DFT_chunk.solve_using_numpy_fft()
        K = int(n*gamma)
        X_chunk_truncated = np.zeros(n,dtype=np.complex)
        X_chunk_truncated[0:K] = X_chunk[0:K]
        iDFT = idft(X_chunk_truncated, fs, n)
        x_idft, Treal = iDFT.solve_using_numpy_ifft()
        x_recon = np.concatenate([x_recon, x_idft])
    Ts= 1/fs
    Treal= np.arange(N)*Ts
        
    # Plot
    fig, axis = plt.subplots(2)
    axis[0].grid()
    axis[1].grid()
    fig.suptitle(f'Q:2.4 Original Signal and Reconstructed Signal  T:{T} fs:{fs}, gamma{gamma}' )
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    axis[0].plot(Treal, x)
    axis[0].set_xlabel('Time (s)')
    axis[0].set_ylabel('Signal')
    axis[1].plot(Treal, x_recon)
    axis[1].set_xlabel('Time (s)')
    axis[1].set_ylabel('Signal')
    plt.show() 
    fig.savefig("MP3_compressor.png", bbox_inches='tight')

########################################## M A I N #########################################################


if __name__ == '__main__':



    #QUESTION 2.2
    T=5
    fs=20000
    gamma_list= [.5,.25,.125,.0625]


    myvoice = microphoneRecordSample(T, fs)
    x = myvoice.createWavFile_vector('mywavefile_q22_1.wav').reshape(T*fs)

    #QUESTION 2.2.1 USING FIRST K DFT COEFFIENCTS
    for gamma in gamma_list:
        voice_compression_q22_1(T, fs, gamma,x)



    #RUN QUESTION 1.3
    """
    1)CALL FUNCTION TO CREATE  CREATE 
        -A SQUARE PULSE AND CALC 
        -THE DFT 
        - COMPUTE THE ENERGY OF THE DIFF SIGNALS x AND xk (THE ENERGY SHOULD DECREASE FOR INCREASING K)
        WITH THE FOLLOWING ATTRIBUTES   
                DURATION            T=  32S
                SAMPLE FREQUENCY OF fs= 8Hz
                NON-0 LENGTH        T0= 4s
            
                K=2, K=4,K=8,K=16,K=32
                
    6) REPEAT THIS PROCESS FOR T0=2      
    """

    T_duration = 32
    fs_samplefrequency = 8
    T0_length_list= [4,2]
    K_energy_list = [2, 3, 8, 16, 32]
    for T0_length in T0_length_list:
        for K_energy in K_energy_list:
            print(f"Running  recog_square_pulse_q13 for T_duration:{T_duration}, fs_samplefrequency:{fs_samplefrequency}, T0_length:{T0_length}, K_energy:{K_energy} ")
            recog_square_pulse_q13(T_duration, fs_samplefrequency, T0_length, K_energy)


    # RUN QUESTION 1.4
    """
    1)CALL FUNCTION TO CREATE  CREATE 
        -A TRIANGLE PULSE AND CALC 
        -THE DFT 
        - COMPUTE THE ENERGY OF THE DIFF SIGNALS x AND xk (THE ENERGY SHOULD DECREASE FOR INCREASING K)
        WITH THE FOLLOWING ATTRIBUTES   
                DURATION            T=  32S
                SAMPLE FREQUENCY OF fs= 8Hz
                NON-0 LENGTH        T0= 4s

                K=2, K=4,K=8,K=16,K=32

    6) REPEAT THIS PROCESS FOR T0=2      
    """
    T_duration = 32
    fs_samplefrequency = 8
    T0_length_list = [4, 2]
    K_energy_list = [2, 3, 8, 16, 32]
    for T0_length in T0_length_list:
        for K_energy in K_energy_list:
            print(f"Running  recog_square_pulse_q13 for T_duration:{T_duration}, fs_samplefrequency:{fs_samplefrequency}, T0_length:{T0_length}, K_energy:{K_energy}")
            recog_square_pulse_q13(T_duration, fs_samplefrequency, T0_length, K_energy)

    '''
   #Generate a square wave of duration T = 32s sampled at a rate fs = 8Hz and frequency 0.25Hz.
   '''
    K_energy_list = [2, 3, 8, 16, 32]
    for K in K_energy_list:
        compression_resconst_square_wave_q19(32,8,.25,K)


    #QUESTION 2.1
    T = 5
    fs = 20000
    record_graph_play_q21(T, fs)


    #QUESTION 2.2.2 USING LARGEST k
    for gamma in gamma_list:
        voice_compression_largest_K_q22_2(T, fs, gamma)





    voice_masking_q23(5, 20000, 0.25)
    #    q21(5, 20000)


