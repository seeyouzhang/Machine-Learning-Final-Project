# import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

# import calculation libraries
from scipy.integrate import odeint
from scipy import integrate, optimize

# time libraries
import datetime
from datetime import datetime

def SEIR(y, t, beta, gamma1, gamma2, alpha, d1, r):
    #unpackage paraments
    S, E, I, R, total_death= y
    #S, E, I, R = y

    # calculte curve
    dS_dt = -beta*S*I*r/N
    dE_dt = beta*S*I*r/N - gamma1*E - alpha*E
    dI_dt = alpha*E - gamma2*I - d1*I
    dR_dt = gamma1*E + gamma2*I

    # the data we want to find
    total_death = d1*E
    #total_confirmed = alpha*E

    # return result
    return([dS_dt, dE_dt, dI_dt, dR_dt, total_death])
    
def fit_odeint(t,beta,gamma1,gamma2,alpha,d1,r):
    re = integrate.odeint(SEIR,[S0,E0,I0,R0,total_death],t,args=(beta,gamma1,gamma2,alpha,d1,r))
    return re[:,-1]

countries = ['Italy','Japan','China','United Kingdom','Spain','Iran','France','Germany']
dates = ['4/15/20','4/16/20','4/17/20','4/18/20','4/19/20','4/20/20','4/21/20','4/22/20','4/23/20','4/24/20','4/25/20','4/26/20','4/27/20','4/28/20','4/29/20','4/30/20','5/1/20']

# load data
confirmed_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
deaths_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
country_pop = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')

country_pop = country_pop.rename(columns={'Country (or dependency)':'Country','Population (2020)':'Population'})
deaths_data = deaths_data.rename(columns={'Province/State':'State','Country/Region':'Country'})
confirmed_data = confirmed_data.rename(columns={'Province/State':'State','Country/Region':'Country'})

re = {}
result_data = {}

for country in countries:

    try:
        # get N
        population = country_pop.loc[country_pop['Country']==country,'Population']
        N = population.iloc[0]

        # deaths data
        deaths_data_copy = deaths_data.loc[confirmed_data['Country']==country]
        deaths = deaths_data_copy.iloc[0,4:]

        y_data = deaths.values
        x_data = [i+1 for i in range(len(y_data))]
        y_data = np.array(y_data)
        x_data = np.array(x_data)


        min_diff = 9999999
        for E0 in range(0,2000,40):
            for I0 in range(0,2000,40):
                #E0 = i # inital Exposed          
                S0 = N - I0 - E0 # inital Susceptible

                # other parameters
                R0 = 0 # inital recovered
                total_death = 0
                total_confirmed = I0
            
                y = S0,E0,I0,R0,total_death

                # fitting
                popt, pcov = optimize.curve_fit(fit_odeint, x_data, y_data)
                fitted = fit_odeint(x_data, *popt)

                # calculte diff
                diff = 0
                for i in range(30):
                    diff += abs(fitted[-i]-y_data[-i])
                    #diff += abs(fitted[-i-1]-y_data[-i-1])
                if diff < min_diff:
                    min_diff = diff
                    re[country] = [popt] + [E0] + [I0]

        print(country+' is Complete!')

        t = [i+1 for i in range(len(deaths)+17)]
        t = np.array(t)

            #predict
        popt = re[country][0]
        E0 = re[country][1]
        I0 = re[country][2]

        fitted2 = fit_odeint(t, *popt)
        fitted1 = fit_odeint(x_data, *popt)

        result_data[country] = [[x_data, y_data], # actual data
                            [x_data, fitted1],
                            [t, fitted2]] # prediction data

        path = '../data/' + country + '_prediction'

        with open(path,'wb') as f:
            pickle.dump(result_data,f)

    except:

        print(country+' is not Complete')

