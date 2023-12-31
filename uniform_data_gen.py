from algos import uniform_data_gen

a_r_list=[1,2,3,4]
gamma_list=[0.001,0.01,0.1,1.0]
# N=60000
d=2
R=0.95
counts=10
sample_type_list=["H"] # "H"
i_seed=1
for sample_type in sample_type_list:
    for a_r in a_r_list:
        for gamma in gamma_list:
            N=int(20000*a_r)
            print(f"sample_type: {sample_type}, a_r: {a_r}, gamma: {gamma}, N: {N}")
            uniform_data_gen(N=N,d=d,gamma=gamma,R=R,a_r=a_r,counts=counts,seed_v=i_seed,cluster_thres=1000,sample_type=sample_type)
            i_seed += 1