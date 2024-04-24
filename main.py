from DGP import *
from models import *
from Clayton import Clayton
from Frank import Frank
from utils import *
from evaluation import *
import argparse
import pickle
from data_loader import RotterdamDataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COPULA = "cl"
THETA = 0.1

torch.manual_seed(0)
np.random.seed(0)

def make_data_dict(data):
    data_dict = dict()
    data_dict['X'] = torch.tensor(data[0], dtype=torch.float)
    data_dict['T'] = torch.tensor(data[1][:,0], dtype=torch.float)
    data_dict['t1'] = torch.tensor(data[1][:,0], dtype=torch.float)
    data_dict['t2'] = torch.tensor(data[1][:,0], dtype=torch.float)
    data_dict['E'] = torch.tensor(data[2][:,0], dtype=torch.float)
    return data_dict

if __name__ == '__main__':
    # Load data
    dl = RotterdamDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    train_data, valid_data, test_data = dl.split_data(train_size=0.7, valid_size=0.5)
    n_events = 2

    train_data[0], valid_data[0], test_data[0] = preprocess_data(train_data[0], valid_data[0], test_data[0],
                                                                 cat_features, num_features,
                                                                 as_array=True)

    train_dict = make_data_dict(train_data)
    val_dict = make_data_dict(valid_data)
    test_dict = make_data_dict(test_data)

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--nf', type=int, required=True)
    #parser.add_argument('--copula', type=str, choices=['cl', 'fr'], required=True)
    #parser.add_argument('--theta', type=float, required=True)
    #parser.add_argument('--tau', type=int, required=True)
    #args = parser.parse_args()
    #nf = args.nf
    #n_train = 2000
    #n_val = 1000
    #n_test = 1000
    #x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    
    nf = train_dict['X'].shape[1]
    dgp1 = Weibull_linear(nf, 14, 4,DEVICE)
    dgp2 = Weibull_linear(nf, 16, 3,DEVICE)
    dgp1.coeff = torch.rand((nf,),device=DEVICE)
    dgp2.coeff = torch.rand((nf,), device=DEVICE)

    if COPULA == 'cl':
        copula_dgp = Clayton(torch.tensor([THETA], device=DEVICE).type(torch.float32), device=DEVICE)
    else:
        copula_dgp = Frank(torch.tensor([THETA], device=DEVICE).type(torch.float32), device=DEVICE)
    #train_dict, val_dict, test_dict = generate_data(x_dict, dgp1, dgp2,DEVICE, copula_dgp)
    
    km_h1, km_p1 = KM(train_dict['T'], 1.0-train_dict['E'])
    km_h2, km_p2 = KM(train_dict['T'], train_dict['E'])
    print(torch.mean(train_dict['E']))
    results_dict = {'dgp_loss':[], 'dep_loss':[], 'indep_loss':[],\
                        'dep1_l1':[], 'dep2_l1':[], 'indep1_l1':[],\
                        'indep2_l1':[], 'ibs_r1':[], 'ibs_r2':[], \
                        'c_r1':[], 'c_r2':[], 'theta':[]}
    for i in range(1): # 10
        
        if COPULA == 'cl':
            copula = Clayton(torch.tensor([4.0], device=DEVICE).type(torch.float32), device=DEVICE)
        else:
            copula = Frank(torch.tensor([18.0], device=DEVICE).type(torch.float32), device=DEVICE)

        dep_model1 = Weibull_log_linear(nf, 2,1, DEVICE)
        dep_model2 = Weibull_log_linear(nf, 1,2,DEVICE)
        while loss_function(dep_model1, dep_model2, train_dict, copula)==0:
                dep_model1 = Weibull_log_linear(nf, 2,2,DEVICE)
                dep_model2 = Weibull_log_linear(nf, 2,2,DEVICE)
        
        dep_model1, dep_model2, copula = dependent_train_loop(dep_model1, dep_model2, train_dict, val_dict, copula, 3000) # 3000
        print(copula.theta)
        indep_model1 = Weibull_log_linear(nf, 2,2, DEVICE)
        indep_model2 = Weibull_log_linear(nf, 2,2,DEVICE)
        while loss_function(dep_model1, dep_model2, train_dict, None)==0:
                indep_model1 = Weibull_log_linear(nf, 2,2,DEVICE)
                indep_model2 = Weibull_log_linear(nf, 2,2,DEVICE)
        indep_model1, indep_model2 = independent_train_loop_linear(indep_model1, indep_model2, train_dict, val_dict, dgp1, dgp2, test_dict, 3000) # 3000
        #likelihood
        results_dict['theta'].append(copula.theta.cpu().detach().clone().item())
        dgp_loss = loss_function(dgp1, dgp2, test_dict, copula_dgp).cpu().detach().clone().numpy().item()
        dep_loss = loss_function(dep_model1, dep_model2, test_dict, copula).cpu().detach().clone().numpy().item()
        indep_loss = loss_function(indep_model1, indep_model2, test_dict, None).cpu().detach().clone().numpy().item()
        results_dict['dgp_loss'].append(dgp_loss)
        results_dict['dep_loss'].append(dep_loss)
        results_dict['indep_loss'].append(indep_loss)
        
        #IBS
        IBS_r1 = evaluate_IBS(dep_model1, indep_model1, dgp1, test_dict,km_h1, km_p1, False)
        IBS_r2 = evaluate_IBS(dep_model2, indep_model2, dgp2, test_dict,km_h2, km_p2, True)
        results_dict['ibs_r1'].append(IBS_r1)
        results_dict['ibs_r2'].append(IBS_r2)
        #c_index
        c_index_r1 = evaluate_c_index(dep_model1, indep_model1, dgp1, test_dict, False)
        c_index_r2 = evaluate_c_index(dep_model2, indep_model2, dgp2, test_dict, True)
        results_dict['c_r1'].append(c_index_r1)
        results_dict['c_r2'].append(c_index_r2)
        #l1
        dep_model1_l1 = surv_diff(dgp1, dep_model1, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        dep_model2_l1 = surv_diff(dgp2, dep_model2, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        indep_model1_l1 = surv_diff(dgp1, indep_model1, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        indep_model2_l1 = surv_diff(dgp2, indep_model2, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        results_dict['dep1_l1'].append(dep_model1_l1)
        results_dict['dep2_l1'].append(dep_model2_l1)
        results_dict['indep1_l1'].append(indep_model1_l1)
        results_dict['indep2_l1'].append(indep_model2_l1)
    print(results_dict)
    #file_name = "linear_"+args.copula+"_" + str(args.tau)+'.pickle'
    #with open(file_name, 'wb') as handle:
    #    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
