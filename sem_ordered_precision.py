# -*- coding: utf-8 -*-
import os
import pandas as pd
import pymc3 as pm
import patsy as pt
import numpy as np
import theano.tensor as tt
from itertools import product as prod
import scipy as sp
import arviz as az
import matplotlib.pyplot as plt
import graphviz as gv
import seaborn as sns
from tqdm import tqdm


############# Precision analysis for CHS BSEM model analysis ###############

## Target precisions
# Expectation: < 10%
# paths and loadings < 1  (unrealistic goal)


#np.random.seed(18)

os.chdir("bsem_precision_analysis/")


#simulate 200 subjects
data = pd.read_table("items_dims.txt")
data['question_number'] = np.arange(1,45)
data = pd.concat([data for d in range(120)])
data['subject'] = ['sub'+str(s+1) for s in range(120) for i in range(44)]

# score1 = score1 = np.random.randint(0,2, int(len(data)*0.8))#bias responses to lower vaues in scale
# score2 = np.random.randint(2,5, int(len(data)*0.2)) #add the left 20% as highest scores

data['score'] = np.random.randint(0,5, int(len(data)))

data0 = data[data.subject=='sub1']

factors = pt.dmatrix('0 + dimension', data=data0, return_type='dataframe')
factors['g'] = np.zeros(len(factors), dtype=int)
factor_names = factors.columns
factors = np.array(factors)

#factors = np.array([np.ones(47, dtype=int),np.ones(47, dtype=int),np.zeros(47, dtype=int)]).T

item_names = []
for i in data0['question_number']:
    d = data0[data0.question_number==i]['dimension'].values[0]
    item = 'q'+str(i)+'_'+d
    item_names.append(item)


sub_data = []
for s in data.subject.unique():
    sub = data[data.subject==s].score.values
    sub_data.append(sub)
       
items = np.array(sub_data)

itemj = pd.Categorical(data.question_number.unique()).codes

n, p = items.shape
p, m = factors.shape
I = tt.eye(m, m)

paths = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]])


with pm.Model() as mod:
   
    #items intercepts priors
    # nu_m = pm.Normal('num', 0, 0.1)
    # nu_s = pm.HalfNormal('nus', 0.1)
    nu = pm.Normal("nu", mu=0, sigma=0.5, shape=p, testval=items.mean(axis=0))

    #factors intercepts priors
    # alpha_m = pm.Normal('alpham', 0, 0.1)
    # alpha_s = pm.HalfNormal('alphas', 0.1)
    alpha = pm.Normal("alpha", mu=0, sigma=0.5, shape=m, testval=np.zeros(m))
      
    #factors residuals priors
    # zeta_m = pm.Normal('zetam', 0, 0.1)
    # zeta_s = pm.HalfNormal('zetas', 0.1)
    zeta = pm.Normal("zeta", mu=0, sigma=0.5, shape=(n,m), testval=0)

    #factor loadings priors
    #l = np.asarray(factors).sum().astype(int)
    # lam_m = pm.Normal('lamm', 0, 0.1)
    # lam_s = pm.HalfNormal('lams', 0.1)
    lam = pm.Normal("lam", mu=0, sigma=0.5, shape=p, testval=np.ones(p))

    #loading matrix
    Lambda = tt.zeros(factors.shape)
    k = 0
    for i, j in prod(range(p), range(m)):
        if factors[i, j] == 1:
            Lambda = tt.inc_subtensor(Lambda[i, j], lam[k])
            k += 1
    pm.Deterministic("Lambda", var=Lambda)

    #paths priors
    #g = np.asarray(paths).sum()
    # gam_m = pm.Normal('gamm', 0, 0.1)
    # gam_s = pm.HalfNormal('gams', 0.1)
    gam = pm.Normal("gam", mu=0, sigma=0.5, shape=m-1, testval=np.ones(m-1))

    #paths matrix
    Gamma = tt.zeros(paths.shape)
    k = 0
    for i, j in prod(range(m), range(m)):
        if paths[i, j] == 1:
            Gamma = tt.inc_subtensor(Gamma[i, j], gam[k])
            k += 1
    pm.Deterministic("Gamma", var=Gamma)

    #latent variables (dimensions)
    I = np.eye(m)
    M = pm.Deterministic('M', nu +  pm.math.matrix_dot(pm.math.matrix_dot((alpha+zeta),
             pm.math.matrix_inverse(I-Gamma.T)), Lambda.T))

    #item cutpoints, shape = number of levels in Likert scale
    a = pm.Normal('cuts', mu=[-2,-1,1,2], sigma=0.5, transform=pm.distributions.transforms.ordered,
        shape=4, testval=np.arange(4))
   

    #observations/likelihood
    y = pm.OrderedLogistic('Y', cutpoints=a, eta=M, observed=items, shape=items.shape)

######### Prio Predictive Checks ################    
os.chdir("/prior_preds/")

with mod:
    preds = pm.sample_prior_predictive(samples=4000,
        var_names=["nu","alpha","zeta","Lambda","Gamma","cuts","M"])

expit = sp.special.expit

def pordlog(a):
    pa = expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    return p_cum[1:] - p_cum[:-1]

cuts = preds['cuts'].T
for i in tqdm(range(len(items.T))):
    prob = cuts - preds['M'].T.mean(axis=1)[i]
    posts = np.array([pordlog(prob.T[s]) for s in range(len(prob.T))]).T
    num = item_names[i][:item_names[i].index('_')]
    num = num.replace('q','')
    prop = data[data.question_number==int(num)]
    prop = [len(prop[prop.score==s])/len(prop) for s in [0,1,2,3,4]]
    question = data[data.question_number==int(num)].question.values[0]
    prob = cuts - (preds['M'].T.mean(axis=1)[i])
    if 'q'+str(i) == 'q'+str(num):
        sco = data[data.question_number==num]['score'].values
    if 'blocking' in item_names[i]:
        name = 'Item '+str(num)+' (Blocking): '
        color1 = 'firebrick'
        color2 = 'crimson'
    if 'hiding' in item_names[i]:
        name = 'Item '+str(num)+' (Hiding): '
        color1 = 'navy'
        color2 = 'mediumblue'
    if 'inspecting' in item_names[i]:
        name = 'Item '+str(num)+' (Inspecting): '
        color1 = 'goldenrod'
        color2 = 'gold'
    pmeans = [m.mean() for m in posts]
    h5s = [az.hdi(h, hdi_prob=0.9)[0] for h in posts]
    h95s = [az.hdi(h, hdi_prob=0.9)[1] for h in posts]
    plt.plot(pmeans, color=color1, linewidth=2)
    plt.fill_between([0,1,2,3,4],h5s,h95s, color=color2, alpha=0.2)
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':')
    plt.suptitle(name+'Prior Probability')
    plt.title(question.replace('.',''), size=11)
    plt.grid(alpha=0.1)
    plt.xticks(range(0,5))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(item_names[i]+'_prob.png', dpi=300)
    plt.close()


########### Sample Model ################
os.chdir("/bsem_precision_analysis/")

with mod:
    trace = pm.sample(2000, tune=2000, chains=4, cores=16, target_accept=0.95)
    #trace = pm.load_trace(tracedir)
   
#pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)


pm.model_to_graphviz(mod).render('mod_graph')

az.plot_energy(trace)
plt.savefig('energy.png',dpi=100)


summ = az.summary(trace, hdi_prob=0.9)
summ.to_csv('summary.csv')

with mod:
    ppc = pm.sample_posterior_predictive(trace, 4000)

fig, ax = plt.subplots()
samps = np.random.randint(0,1000,100)
for s in samps:
    ax = sns.kdeplot(ppc['Y'].mean(axis=1)[s], gridsize=1000, color=(0.2, 0.6, 0.5, 0.1))
sns.kdeplot(items.mean(axis=0), gridsize=1000, color='darkviolet');
plt.savefig('ppc.png',dpi=300)
plt.close()
plt.close()

###### Save loadings ######
for j in range(3):
    mean = []
    stdev = []
    hdi5 = []
    hdi95 = []
    for i in range(len(item_names)):
        m = trace['Lambda'].T[j][i].mean().round(3)
        sd = trace['Lambda'].T[j][i].std().round(3)
        h5 = az.hdi(trace['Lambda'].T[j][i], hdi_prob=0.9)[0].round(3)
        h95 = az.hdi(trace['Lambda'].T[j][i], hdi_prob=0.9)[1].round(3)
        mean.append(m)
        stdev.append(sd)
        hdi5.append(h5)
        hdi95.append(h95)
        if j == 0:
            Blocking = pd.DataFrame({'Blocking mean':mean, 'Blocking SD':stdev, 
                                     'Blocking HDI 5%':hdi5, 'Blocking HDI 95%':hdi95,
                                     'Precision':np.array(hdi95)-np.array(hdi5)})
        if j == 1:
            Hiding = pd.DataFrame({'Hiding mean':mean, 'Hiding SD':stdev, 
                                   'Hiding HDI 5%':hdi5, 'Hiding HDI 95%':hdi95,
                                   'Precision':np.array(hdi95)-np.array(hdi5)})
        if j == 2:
            Inspecting = pd.DataFrame({'Inspecting mean':mean, 'Inspecting SD':stdev, 
                                   'Inspecting HDI 5%':hdi5, 'Inspecting HDI 95%':hdi95,
                                   'Precision':np.array(hdi95)-np.array(hdi5)})

loadings = pd.concat([Blocking,Hiding,Inspecting],axis=1)
loadings['Item'] = item_names
loadings.to_csv('loadings_precision.csv', index=False)


###### Save paths ######
mblock = trace["Gamma"].T[3][0].mean().round(3) #blocking mean
sdblock = trace["Gamma"].T[3][0].std().round(3)
h5block = az.hdi(trace['Gamma'].T[3][0], hdi_prob=0.9)[0].round(3)
h95block = az.hdi(trace['Gamma'].T[3][0], hdi_prob=0.9)[1].round(3)
mhid = trace["Gamma"].T[3][1].mean().round(3) #hiding mean
sdhid = trace["Gamma"].T[3][1].std().round(3)
h5hid = az.hdi(trace['Gamma'].T[3][1], hdi_prob=0.9)[0].round(3)
h95hid = az.hdi(trace['Gamma'].T[3][1], hdi_prob=0.9)[1].round(3)
mins = trace["Gamma"].T[3][2].mean().round(3) #inspecting mean
sdins = trace["Gamma"].T[3][2].std().round(3)
h5ins = az.hdi(trace['Gamma'].T[3][2], hdi_prob=0.9)[0].round(3)
h95ins = az.hdi(trace['Gamma'].T[3][2], hdi_prob=0.9)[1].round(3)

dims = ['Blocking', 'Hiding', 'Inspecting']
gmean = [mblock, mhid, mins]
gsd = [sdblock, sdhid, sdins]
gh5 = [h5block, h5hid, h5ins]
gh95 = [h95block, h95hid, h95ins]
paths = pd.DataFrame({'Dimension':dims, 'g mean':gmean, 'g SD':gsd, 
                      'g HDI 5%':gh5, 'g HDI 95%':gh95, 'precision':np.array(gh95)-np.array(gh5)})
paths.to_csv('paths_precision.csv', index=False)



###### Plot loadings and path as graph #####
blo_data = data[data.dimension=='blocking']
hid_data = data[data.dimension=='hiding']
ins_data = data[data.dimension=='inspecting']

g = gv.Digraph('Habits', filename='model.gv', format="png")
g.graph_attr['rankdir'] = 'RL'
g.graph_attr['dpi'] = '300'
for index, row in blo_data.iterrows():
    g.edge('blocking', 'q'+str(row['question_number'])+'. '+str(row["question"]), )
    g.node('q'+str(row['question_number'])+'. '+str(row["question"]), shape='box')
for index, row in hid_data.iterrows():
    g.edge('hiding', 'q'+str(row["question_number"])+'. '+str(row["question"]), )
    g.node('q'+str(row['question_number'])+'. '+str(row["question"]), shape='box')
for index, row in ins_data.iterrows():
    g.edge('inspecting', 'q'+str(row["question_number"])+'. '+str(row["question"]), )
    g.node('q'+str(row['question_number'])+'. '+str(row["question"]), shape='box')
g.edge('Habits', 'blocking')
g.edge('Habits', 'hiding')
g.edge('Habits', 'inspecting')
g.render('model')



##### Plot Probs #####
os.chdir("/response_prob/")

expit = sp.special.expit

def pordlog(a):
    pa = expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))
    return p_cum[1:] - p_cum[:-1]

precis = []

cuts = trace['cuts'].T
for i in tqdm(range(len(items.T))):
    prob = cuts - trace['M'].T.mean(axis=1)[i]
    posts = np.array([pordlog(prob.T[s]) for s in range(len(prob.T))]).T
    num = item_names[i][:item_names[i].index('_')]
    num = num.replace('q','')
    prop = data[data.question_number==int(num)]
    prop = [len(prop[prop.score==s])/len(prop) for s in [0,1,2,3,4]]
    question = data[data.question_number==int(num)].question.values[0]
    prob = cuts - (trace['M'].T.mean(axis=1)[i])
    if 'q'+str(i) == 'q'+str(num):
        sco = data[data.question_number==num]['score'].values
    if 'blocking' in item_names[i]:
        name = 'Item '+str(num)+' (Blocking): '
        color1 = 'firebrick'
        color2 = 'crimson'
    if 'hiding' in item_names[i]:
        name = 'Item '+str(num)+' (Hiding): '
        color1 = 'navy'
        color2 = 'mediumblue'
    if 'inspecting' in item_names[i]:
        name = 'Item '+str(num)+' (Inspecting): '
        color1 = 'goldenrod'
        color2 = 'gold'
    pmeans = [m.mean() for m in posts]
    h5s = [az.hdi(h, hdi_prob=0.9)[0] for h in posts]
    h95s = [az.hdi(h, hdi_prob=0.9)[1] for h in posts]
    precis.append([np.array(h95s)-np.array(h5s)])
    plt.plot(pmeans, color=color1, linewidth=2)
    plt.fill_between([0,1,2,3,4],h5s,h95s, color=color2, alpha=0.2)
    plt.plot(prop, color='slategray', linewidth=2, linestyle=':')
    plt.suptitle(name+'Response Probability')
    plt.title(question.replace('.',''), size=11)
    plt.grid(alpha=0.1)
    plt.xticks(range(0,5))
    plt.xlabel('Score')
    plt.ylabel('Probability')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(item_names[i]+'_prob.png', dpi=300)
    plt.close()


os.chdir("/bsem_precision_analysis/")

precision = np.concatenate(precis)
precision = np.concatenate(precision)
maxs = precision[precision > 0.1]
print(maxs)
print(len(maxs))
