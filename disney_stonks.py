import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt


# In[3]:


# importing Disney stock history, Jan. 4, 2010 to July 13, 2020
dis = pd.read_csv('dis.csv',parse_dates=['Date'])


# In[4]:


# compiling Disney releases, including Marvel Studios
releases = pd.DataFrame()
decades = ['2010','2020']
for d in decades:
    releases_dec = pd.read_csv(f'disney_{d}.csv',names=['Type','Title','US_Release','Production_Companies'],header=0,
                              usecols=['Title','US_Release','Production_Companies'],parse_dates=['US_Release'])
    releases = pd.concat([releases,releases_dec],ignore_index=True)
marvel = pd.read_csv('marvel.csv',names=['Title','US_Release','Production_Companies'],header=0,parse_dates=['US_Release'])
releases = pd.concat([releases,marvel],ignore_index=True)
# print(releases.US_Release[0])
# datetime.datetime.strptime(releases.US_Release,'%d-%b-%y')
# releases['US_Release'] = releases['US_Release'].astype('datetime64[D]')
releases.sort_values('US_Release',inplace=True,ignore_index=True)


# In[5]:


# plotting Disney stock price over the past decade, with vertical lines representing "major" Disney releases
# (Disney, Pixar, Marvel)
# plt.plot(dis.Date,dis.Close,'k-')
# plt.vlines(releases.loc[releases.Production_Companies=='Marvel Studios'].US_Release,20,160,linestyles='dotted',colors='b')
# plt.vlines(releases.loc[releases.Production_Companies=='Pixar Animation Studios'].US_Release,20,160,
#            linestyles='dotted',colors='r')
# plt.vlines(releases.loc[releases.Production_Companies=='Walt Disney Animation Studios'].US_Release,20,160,
#            linestyles='dotted',colors='g')
# plt.axis([dis.Date.min(),dis.Date.max(),20,160])
# plt.show()


# In[6]:


# calculating days until major Disney release
def nearest(items, target):
    return min(items, key = lambda x: abs(x-target))
# dis['Disneyless_Days'] = [releases.loc[releases.US_Release>date].US_Release.iloc[0] - date for date in dis.Date]

dis['days_until'] = np.nan
dis['days_since'] = np.nan
for ind in dis.index:
    try:
        until = releases.loc[releases.US_Release > dis.loc[ind,'Date']].US_Release.iloc[0] - dis.loc[ind,'Date']
        until = until.days
        dis.loc[ind,'days_until'] = until
    except:
        pass
    try:
        since = dis.loc[ind,'Date'] - releases.loc[releases.US_Release < dis.loc[ind,'Date']].US_Release.iloc[-1]
        since = since.days
        dis.loc[ind,'days_since'] = since
    except:
        pass



# In[80]:


# defining dataset independent & dependent variables
try:
    dis['Date'] = dis['Date'].map(dt.datetime.toordinal)
except:
    pass
# dataset = dis.loc[dis.days_until<=7].loc[dis.days_since<=7]
dataset = dis
x = dataset.dropna().drop(['Close','Adj Close'],axis=1)
y = dataset.dropna()['Close']


# In[81]:


# train/test split and converting dates into numerical values
# x_train, x_test, y_train, y_test = train_test_split(dataset,valid,test_size=0.2)
split = int(len(x)*0.8)
x_train = x.iloc[:split]
y_train = y.iloc[:split]
x_test = x.iloc[split:]
y_test = y.iloc[split:]


# In[82]:


# creating and fitting model
lm = LinearRegression()
model = lm.fit(x_train,y_train)


# In[83]:


predictions = lm.predict(x_test)


# In[89]:


plt.plot(dis.Date,dis.Close,'b--')
plt.plot(x_test.Date,predictions,'r.')
plt.show()
print('Score:',model.score(x_test,y_test))


# In[ ]:




