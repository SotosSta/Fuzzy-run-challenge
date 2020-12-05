"""
Great Uni Hack 2020 Fuzzy labs challenge

Written by: Sotirios Stamnas

Challenge: Given 3 files containing data for the three Fuzzy Labs runners
Matt Misha and Tom, we have to calculate who ran the furthest in about 120s.


"""
#--------------------------------------------------------------------------
#Import libraries
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

#--------------------------------------------------------------------------
#Functions that we will be using for this analysis, Credits to fuzzylabs
#For providing these functions in a jupyter notebook.


def get_peaks(data):
    pos_kwargs={
        "distance": 20,
        "height": (35, None)
    }
    peaks, _ = find_peaks(data, **pos_kwargs)
    return np.hstack([[0],peaks,[len(data)-1]])

def get_step_ranges(data):
    peaks = get_peaks(data)
    midpoints = (peaks[1:] + peaks[:-1]) / 2
    return np.vstack([midpoints[:-1], midpoints[1:]]).T.round()




def get_velocity(df, steps, window_size, step_size):
    velocity = []
    for i in np.arange(-window_size+step_size, len(df), step_size):
        # Grab a window of data
        _df = df.loc[i:i+window_size]
        # Get the steps belonging to that window
        _steps = [x for x in steps if (int(x[0]) in _df.index) and (x[1] in _df.index)]
        # Store velocity
        velocity += [{
            "time": _df.time.iloc[-1],
            "velocity": np.mean([get_step_speed(_df.pca0, _df.time, x) for x in _steps])
        }]
    velocity = pd.DataFrame(velocity).fillna(0.0)
    return velocity

def get_step_speed(data, time, step):
    a, b = step.astype(int)
    velocities = sp.integrate.cumtrapz(data[:,0][a:b], x=time[a:b] / 1000)
    v_max = np.max(np.abs(velocities))
    where = np.argmax(np.abs(velocities))  
    time_when_velocity_max = time[a+where] / 1000
    return v_max,time_when_velocity_max
#--------------------------------------------------------------------------

#Load data
Df_matt_data =  pd.read_csv('matt.csv')

Df_misha_data  = pd.read_csv('misha.csv')

Df_tom_data =  pd.read_csv('tom.csv')


plt.figure(0)
#Plotting the accelaration componets for matt's run

plt.title("Tom's run", fontsize = 16)

plt.xlabel('time (s)', fontsize = 14)
plt.ylabel(r'a ($ms^{-2}$) ', fontsize = 14)

plt.plot(Df_matt_data.time/1000,Df_matt_data.aX, label = 'aX')
plt.plot(Df_matt_data.time/1000,Df_matt_data.aY, label =  'aY')
plt.plot(Df_matt_data.time/1000,Df_matt_data.aZ, label = 'aZ')
#We can see from the three plots (zooming in a little bit) that gravity is
# acting on part of the x and z directions. We can see that because the 
#ax and aZ accelarations are a bit up than the aX accelaration 
plt.legend()
plt.show()

plt.figure(1)
#Plotting the accelaration componets for misha's run

plt.title("Misha's run", fontsize = 16)

plt.xlabel('time (s)', fontsize = 14)
plt.ylabel(r'a ($ms^{-2}$) ', fontsize = 14)

plt.plot(Df_misha_data.time/1000,Df_misha_data.aX, label = 'aX')
plt.plot(Df_misha_data.time/1000,Df_misha_data.aY, label =  'aY')
plt.plot(Df_misha_data.time/1000,Df_misha_data.aZ, label = 'aZ')
#In this case, we can see from the three plots (zooming yet again) that
# gravity is acting on part of the x and z directions. 
plt.legend()
plt.show()

plt.figure(2)
#Plotting the accelaration componets for tom's run

plt.title("Tom's run", fontsize = 16)

plt.xlabel('time (s)', fontsize = 14)
plt.ylabel(r'a ($ms^{-2}$) ', fontsize = 14)

plt.plot(Df_tom_data.time/1000,Df_tom_data.aX, label = 'aX')
plt.plot(Df_tom_data.time/1000,Df_tom_data.aY, label =  'aY')
plt.plot(Df_tom_data.time/1000,Df_tom_data.aZ, label = 'aZ')
#Lastly in tom's run (zooming in one last time) gravity is
# acting on part of the y and z directions. 
plt.legend()
plt.show()


#This variation on which accelaration component is affected by gravity,
#is probably due to factors like, different placement of the device on the
#shoe of each runner, or different angle of inclination in the route of
#each runner.


#Convert all accelarations from g to m/s^2 units
#and account for gravity in x and z components

#Also we try doing an approximate removal of the gravity's effect
#on the accelaration components


Df_matt_data.aX=  (Df_matt_data.aX-0.75)*9.8
Df_matt_data.aY =  (Df_matt_data.aY)*9.8
Df_matt_data.aZ =  (Df_matt_data.aZ-0.6)*9.8

Df_misha_data.aX=  (Df_misha_data.aX-0.3)*9.8
Df_misha_data.aY = (Df_misha_data.aY-0.4)*9.8
Df_misha_data.aZ =  (Df_misha_data.aZ-0.67)*9.8

Df_tom_data.aX=  Df_tom_data.aX*9.8 
Df_tom_data.aY =  (Df_tom_data.aY)*9.8
Df_tom_data.aZ =  (Df_tom_data.aZ-0.6)*9.8
#-----------------------------------------------------------------------------

#Try finding velocity of each runner by simply integrating

matt_v = sp.integrate.cumtrapz(Df_matt_data.aX, Df_matt_data.time/1000)

misha_v =  sp.integrate.cumtrapz(Df_misha_data.aX, Df_misha_data.time/1000)

tom_v =  sp.integrate.cumtrapz(Df_tom_data.aX, Df_tom_data.time/1000)

plt.figure(3)
#Plotting the results

plt.xlabel('time (s)', fontsize = 14)
plt.ylabel(r'V ($ms^{-1}$) ', fontsize = 14)

plt.plot(Df_matt_data.time[:-1]/1000,matt_v, label = 'Matt' )
plt.plot(Df_misha_data.time[:-1]/1000,misha_v, label = 'Misha' )
plt.plot(Df_tom_data.time[:-1]/1000,tom_v, label = 'Tom' )

#Results do not seem really helpful velocity should remain relatively
#constant over the duration of the run. We obviously need another way
#to approach this problem
plt.legend()
plt.show()


#Perform principal component analysis by using the PCA function from
#sklearn. The PCA analysis helps us to reduce the dimensionality of our
#problem

#Τhe first axis of PCA is the 'major axis', #in our case that should be
#acceleration projected into the direction of motion for all three runners

pca_matt = PCA().fit_transform(Df_matt_data.loc[:,["aX", "aY", "aZ"]].values)
pca_misha = PCA().fit_transform(Df_misha_data.loc[:,["aX", "aY", "aZ"]].values)
pca_tom = PCA().fit_transform(Df_tom_data.loc[:,["aX", "aY", "aZ"]].values)


#Call get_step_ranges function to locate the time intervals within a
#'step' is taken

steps_matt = get_step_ranges(pca_matt[:,0])
#steps_matt.shape

steps_misha = get_step_ranges(pca_misha[:,0])
#steps_misha.shape

steps_tom = get_step_ranges(pca_tom[:,0])
#steps_tom.shape

#We will plot the 100th step for each runner so we can visualize better
#what we defined as a 'step'.
step_matt = steps_matt[100]
step_misha = steps_misha[100]
step_tom = steps_tom[100]


plt.figure(4)
#Plot a step of Matt's runs
plt.title("Matt's run", fontsize = 16)

plt.xlabel('Time (s)', fontsize = 14)
plt.ylabel(r'a ($ms^{-2}$)', fontsize = 14)

plt.plot(Df_matt_data.time[int(step_matt[0]):int(step_matt[1])]/1000,
         pca_matt[int(step_matt[0]):int(step_matt[1])][:,0])

plt.show()

plt.figure(5)
#Plot a step of Misha's runs
plt.title("Mishas's run", fontsize = 16)

plt.xlabel('Time (s)', fontsize = 14)
plt.ylabel(r'a ($ms^{-2}$)', fontsize = 14)

plt.plot(Df_misha_data.time[int(step_misha[0]):int(step_misha[1])]/1000,
         pca_misha[int(step_misha[0]):int(step_misha[1])][:,0])

plt.show()


plt.figure(6)
#Plot a step of Tom's runs
plt.title("Tom's run", fontsize = 16)

plt.xlabel('Time (s)', fontsize = 14)
plt.ylabel(r'a ($ms^{-2}$)', fontsize = 14)

plt.plot(Df_tom_data.time[int(step_tom[0]):int(step_tom[1])]/1000,
         pca_tom[int(step_tom[0]):int(step_tom[1])][:,0])

plt.show()
#--------------------------------------------------------------------------

#We find the velocity of each runner at each step using the get_step_speed
#function and a loop over each runner's number of steps


#For Matt
v_all_matt = np.array([])
t_all_matt = np.array([])

for i in range(len(steps_matt)):
    v_matt,t_matt = get_step_speed(pca_matt,Df_matt_data.time,steps_matt[i])
    v_all_matt = np.append(v_all_matt,v_matt)  
    t_all_matt = np.append(t_all_matt,t_matt) 
    
#For Misha 
v_all_misha = np.array([])
t_all_misha = np.array([])
    
for i in range(len(steps_misha)):
    v_misha,t_misha = get_step_speed(pca_misha,Df_misha_data.time,steps_misha[i])
    v_all_misha = np.append(v_all_misha,v_misha)  
    t_all_misha = np.append(t_all_misha,t_misha) 
    
  
#For Tom
v_all_tom = np.array([])
t_all_tom = np.array([])
    
for i in range(len(steps_tom)):
    v_tom,t_tom = get_step_speed(pca_tom,Df_tom_data.time,steps_tom[i])
    v_all_tom = np.append(v_all_tom,v_tom)  
    t_all_tom = np.append(t_all_tom,t_tom)
    

plt.figure(7)
#Plot the velocities of each runner in the same graph

plt.xlabel('Time (s)', fontsize = 14)
plt.ylabel(r'v ($ms^{-1}$)', fontsize = 14)

plt.plot(t_all_matt,v_all_matt, label = 'Matt')
plt.plot(t_all_misha,v_all_misha, label = 'Misha')
plt.plot(t_all_tom,v_all_tom, label = 'Tom')

#The results of this approach are much better, we can see that the 
#Velocities of each runner stay relatively constant through their runs
#except from the beggining and end. However, these wont effect our results
#that much.
plt.legend()
plt.show()
#--------------------------------------------------------------------------

#Finally integrate the velocity distributions over time to get the
#distance run over time for our 3 runners

distance_matt = sp.integrate.cumtrapz(v_all_matt, x=t_all_matt)
distance_misha = sp.integrate.cumtrapz(v_all_misha, x=t_all_misha)
distance_tom = sp.integrate.cumtrapz(v_all_tom, x=t_all_tom)

#Lastly we plot our results and carefully correcting the data so that
#all runners start at position x =0 at t=0 and runs about 120s.

plt.figure(8)
#Plot the distance results for all three runners in the same graph
plt.plot(t_all_matt[1:-1]-t_all_matt[0],
         distance_matt[:-1]-distance_matt[0], label = 'Matt')
plt.plot(t_all_misha[1:-1]-t_all_misha[0],
         distance_misha[:-1]-distance_misha[0], label = 'Misha')
plt.plot(t_all_tom[1:-1]-t_all_tom[0],
         distance_tom[:-1]-distance_tom[0], label = 'Tom')

plt.legend()
plt.show()

#Calculate total distance run 
matt_total_distance = distance_matt[-2]-distance_matt[0]
misha_total_distance = distance_misha[-2]-distance_misha[0]
tom_total_distance = distance_tom[-2]-distance_tom[0]

#Calculate total time run 
matt_total_time = t_all_matt[-2]-t_all_matt[0]
misha_total_time = t_all_misha[-2]-t_all_misha[0]
tom_total_time = t_all_tom[-2]-t_all_tom[0]

print('Matt ran a total of {:.2f} m in {:.2f} s. 3rd'.format(matt_total_distance,
                                                           matt_total_time))
print('Misha ran a total of {:.2f} m in {:.2f} s. 2nd'.format(misha_total_distance,
                                                  misha_total_time))
print('Tom ran a total of {:.2f} m in {:.2f} s. 1st '.format(tom_total_distance,
                                                             tom_total_time))

#Its a close call between Misha and Tom because Tom run about 3 seconds
#more than Misha but i think that this win belongs to Tom. Congrats Tom.
#-------------------------------------------------------------------------

#END OF CODE







