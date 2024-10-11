from scipy.stats import norm
#%%def sigma_calulator
def sigma_calculator(mean = 1,lower_bound = 1,upper_bound = 1,confidence_level = 0.99):
    # Given values
    mean = 1
    lower_bound = 1
    lower_bound = 1
    confidence_level = 0.99
    
    # z-score for 99% confidence level (two-tailed)
    z_score = norm.ppf((1 + confidence_level) / 2)
    
    # Calculate sigma
    sigma = (upper_bound - mean) / z_score
    return sigma
#%% compute
print(str( sigma_calculator(mean = 1, lower_bound = 0.94,upper_bound = 1.06,confidence_level = 0.99)))
#print(f"The standard deviation (sigma) is approximately: {sigma:.6f}")
