import numpy as np
from scipy.stats import beta, norm, binom, multinomial
from scipy.special import logit, expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def dordbeta(x, mu, phi=1, cutpoints=(-1, 1), log=False):
    """
    Vectorized ordered beta density function.
    
    Parameters:
    - x: array-like, observed values in the (0, 1) interval.
    - mu: array-like, mean of the beta distribution, should be between (0, 1).
    - phi: scalar, dispersion parameter, should be positive.
    - cutpoints: tuple of two numeric values for cutpoints.
    - log: boolean, return log density if True.
    
    Returns:
    - densities (or log-densities) of the ordered beta distribution.
    """
    # Ensure inputs are numpy arrays
    #x = np.asarray(x)
    #mu = np.asarray(mu)
    
    # Check that all values of mu are in the valid range
    if not np.all((mu > 0) & (mu < 1)):
        raise ValueError("mu must be between 0 and 1.")
    
    if not phi > 0:
        raise ValueError("phi must be greater than 0.")
    
    if not cutpoints[1] > cutpoints[0]:
        raise ValueError("Second cutpoint must be greater than the first.")
    
    # Convert mu to logit scale
    mu_ql = logit(mu)
    
    # Probabilities for the three categories: 0, beta-distributed, 1
    low = 1 - expit(mu_ql - cutpoints[0])
    middle = expit(mu_ql - cutpoints[0]) - expit(mu_ql - cutpoints[1])
    high = expit(mu_ql - cutpoints[1])
    
    densities = np.zeros_like(x)
    
    # Case when x == 0
    densities = np.where(x == 0, low, densities)
    
    # Case when x == 1
    densities = np.where(x == 1, high, densities)
    
    # Case when 0 < x < 1 (beta-distributed values)
    beta_mask = (x > 0) & (x < 1)
    beta_dens = beta.pdf(x[beta_mask], mu[beta_mask] * phi, (1 - mu[beta_mask]) * phi)
    densities[beta_mask] = middle[beta_mask] * beta_dens
    
    # Return log density if requested
    if log:
        densities = np.log(densities)
    
    return densities

# Random data generator for the ordered beta model
def rordbeta(n=100, mu=0.5, phi=1, cutpoints=(-1, 1)):
    """
    Generate random data from the ordered beta distribution.
    
    Parameters:
    - n: number of samples to generate.
    - mu: mean of the beta distribution (should be between 0 and 1).
    - phi: dispersion parameter.
    - cutpoints: cutpoints for the ordered beta distribution.
    
    Returns:
    - Generated data.
    """
    if not (0 < mu < 1):
        raise ValueError("mu must be between 0 and 1.")
    
    if not phi > 0:
        raise ValueError("phi must be greater than 0.")
    
    if not cutpoints[1] > cutpoints[0]:
        raise ValueError("Second cutpoint must be greater than the first.")
    
    mu_ql = logit(mu)
    low = 1 - expit(mu_ql - cutpoints[0])
    middle = expit(mu_ql - cutpoints[0]) - expit(mu_ql - cutpoints[1])
    high = expit(mu_ql - cutpoints[1])
    
    beta_variate = beta.rvs(mu * phi, (1 - mu) * phi, size=n)
    outcomes = np.random.choice([0, 2, 1], size=n, p=[low, middle, high])
    random_values = np.array([0 if o == 0 else 1 if o == 1 else beta_variate[i] for i, o in enumerate(outcomes)])
    
    return random_values
  
def custom_logit(X):
    """
    Custom logit function that returns:
    - 0 if X == 0
    - 1 if X == 1
    - logit(X) otherwise
    This function accepts arrays as input.
    """
    X = np.asarray(X)  # Ensure X is a numpy array

    # Create an array for the result
    result = np.empty_like(X)

    # Handle the edge cases
    result[X == 0] = 0  # If X == 0, return 0
    result[X == 1] = 1  # If X == 1, return 1

    # Apply the logit function for values strictly between 0 and 1
    mask = (X > 0) & (X < 1)
    result[mask] = logit(X[mask])  # Apply logit only to values between 0 and 1

    return result

class OrderedBetaModel:
    def __init__(self, endog, exog):
        self.endog = endog
        self.exog = np.column_stack((np.ones(exog.shape[0]), exog))  # Add intercept
        self.result = None
        self.log_likelihood_value = None

    def fit(self):
        n_covariates = self.exog.shape[1]

        # Initial parameter guess: [coefficients for covariates, phi, cutpoint1, cutpoint2]
        initial_params = np.ones(n_covariates + 3)
        initial_params[n_covariates] = 1.0  # Initial guess for phi
        initial_params[n_covariates + 1] = -1.0  # Initial guess for cutpoint1
        initial_params[n_covariates + 2] = 1.0  # Initial guess for cutpoint2

        def log_likelihood(params):
            beta_mu = params[:n_covariates]
            phi = params[n_covariates]
            cutpoint1 = params[n_covariates + 1]
            cutpoint2 = params[n_covariates + 2]
            mu_values = expit(np.dot(self.exog, beta_mu))
            log_likelihood = np.sum(dordbeta(self.endog, mu_values, phi, cutpoints=(cutpoint1, cutpoint2), log=True))
            return -log_likelihood  # Return negative log-likelihood for minimization

        # Minimize the negative log-likelihood
        self.result = minimize(
            log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(None, None)] * n_covariates + [(0, None), (None, None), (None, None)]
        )

        # Store the log-likelihood of the optimized parameters
        self.log_likelihood_value = -log_likelihood(self.result.x)
        return self

    def predict(self, exog=None):
        """
        Generate predicted values based on the fitted model. The function calculates the probabilities 
        for each category (0, (0,1), 1), selects the category with the maximum probability, and returns 
        0, the inverse logit of the predicted mu (for (0,1)), or 1.
        """
        if exog is None:
            exog = self.exog
        else:
            exog = np.column_stack((np.ones(exog.shape[0]), exog))  # Add intercept

        # Calculate the linear predictor and logistic (mu) values
        n_covariates = self.exog.shape[1]
        beta_mu = self.result.x[:n_covariates]
        linear_predictor = np.dot(exog, beta_mu)
        mu_values = expit(linear_predictor)  # Logistic transformation to get mu

        # Cutpoint parameters
        cutpoint1 = self.result.x[n_covariates + 1]  # First cutpoint (for P(0))
        cutpoint2 = self.result.x[n_covariates + 2]  # Second cutpoint (for P(1))

        # Calculate the probabilities for each category (0, (0,1), 1)
        P0 = 1 - expit(linear_predictor - cutpoint1)  # Probability of being exactly 0
        P1 = expit(linear_predictor - cutpoint2)  # Probability of being exactly 1
        P_between = 1 - P0 - P1  # Probability of being between 0 and 1

        # Generate predicted responses based on maximum probability
        predicted_responses = []
        for i in range(len(mu_values)):
            # Calculate the probabilities for the current observation
            probabilities = [P0[i], P_between[i], P1[i]]

            # Find the index of the maximum probability
            max_prob_index = np.argmax(probabilities)

            # Assign the response based on the max probability
            if max_prob_index == 0:  # Case where P(0) is the maximum
                predicted_responses.append(0)
            elif max_prob_index == 2:  # Case where P(1) is the maximum
                predicted_responses.append(1)
            else:  # Case where P((0,1)) is the maximum
                predicted_responses.append(mu_values[i])

        return np.array(predicted_responses)


    def summary(self):
        """
        Print a summary of the fitted model, including regression coefficients,
        standard errors, confidence intervals, degrees of freedom, AIC, BIC, log-likelihood, and N.
        """
        # Number of observations
        n_obs = len(self.endog)
        # Number of parameters (including intercept and cutpoints)
        n_params = len(self.result.x)

        # Extract the Hessian inverse (covariance matrix)
        hessian_inv = self.result.hess_inv.todense() if hasattr(self.result.hess_inv, 'todense') else np.linalg.inv(self.result.hess_inv)
        
        # Standard errors
        std_errors = np.sqrt(np.diag(hessian_inv))

        # 95% Confidence intervals
        z_score = norm.ppf(0.975)
        conf_intervals = np.column_stack((self.result.x - z_score * std_errors, self.result.x + z_score * std_errors))

        # Degrees of freedom
        df = n_obs - n_params

        # Calculate AIC and BIC
        log_likelihood = self.log_likelihood_value
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood

        # Printing summary
        print("=================================================")
        print("               Ordered Beta Regression            ")
        print("=================================================")
        print(f"Number of observations: {n_obs}")
        print(f"Degrees of freedom: {df}")
        print(f"Log-likelihood: {log_likelihood:.4f}")
        print(f"AIC: {aic:.4f}")
        print(f"BIC: {bic:.4f}")
        print("-------------------------------------------------")
        print(f"{'Parameter':<15}{'Estimate':<15}{'Std. Error':<15}{'CI (95%)':<30}")
        print("-------------------------------------------------")
        param_names = ['Intercept'] + [f'X{i}' for i in range(1, self.exog.shape[1])] + ['phi', 'cutpoint1', 'cutpoint2']
        for i, param_name in enumerate(param_names):
            estimate = self.result.x[i]
            std_err = std_errors[i]
            ci_low, ci_high = conf_intervals[i]
            print(f"{param_name:<15}{estimate:<15.4f}{std_err:<15.4f}({ci_low:.4f}, {ci_high:.4f})")
        print("=================================================")
        
    def rmse(self):
        """
        Calculate the Root Mean Square Error (RMSE) of the model fit.
        RMSE is calculated as sqrt(mean((y_true - y_pred)^2)).
        """
        predictions = self.predict()  # Predicted values from the model
        residuals = self.endog - predictions  # Residuals (y_true - y_pred)
        mse = np.mean(residuals ** 2)  # Mean squared error
        rmse_value = np.sqrt(mse)  # Root mean squared error
        return rmse_value
      
    def plot_continuous_predictions(self, exog=None, n_bootstrap=1000):
        """
        Plot the continuous model predictions with 95% confidence intervals calculated via bootstrapping.
        The response and predictions are first transformed to the logit scale, then confidence intervals are
        calculated, and finally, the values are converted back to the (0,1) range using the inverse logit.
        """
        if exog is None:
            exog = self.exog

        # Generate original predictions on the logit scale
        linear_predictions = self.predict(exog)
        continuous_mask = (self.endog > 0) & (self.endog < 1)  # Mask for continuous values
        continuous_linear_preds = linear_predictions[continuous_mask]
        continuous_actual = self.endog[continuous_mask]

        # Initialize matrix to store bootstrap predictions
        bootstrap_predictions = np.zeros((n_bootstrap, len(continuous_linear_preds)))

        # Perform bootstrapping
        for i in range(n_bootstrap):
            # Sample indices with replacement
            bootstrap_indices = np.random.choice(len(continuous_actual), size=len(continuous_actual), replace=True)
            bootstrap_sample_exog = exog[bootstrap_indices, :]
            bootstrap_sample_endog = self.endog[bootstrap_indices]

            # Refit the model on the bootstrap sample
            bootstrap_model = OrderedBetaModel(bootstrap_sample_endog, bootstrap_sample_exog)
            bootstrap_model.fit()

            # Generate predictions on the original exog data (logit scale)
            bootstrap_preds = bootstrap_model.predict(exog)[continuous_mask]
            bootstrap_predictions[i, :] = bootstrap_preds

        # Calculate confidence intervals (95%) from the bootstrap predictions
        conf_interval_lower_logit = np.percentile(bootstrap_predictions, 2.5, axis=0)
        conf_interval_upper_logit = np.percentile(bootstrap_predictions, 97.5, axis=0)

        # Transform original predictions and confidence intervals back to the (0, 1) scale using the inverse logit
        continuous_predictions = continuous_linear_preds
        lower_errors = continuous_predictions - conf_interval_lower_logit
        upper_errors = conf_interval_upper_logit - continuous_predictions
        # Plot predicted data with bootstrapped error bars (blue points)
        
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(continuous_actual, continuous_predictions, yerr=[lower_errors, upper_errors], fmt='o', color='blue',
                     ecolor='red', elinewidth=2, capsize=4, label=f'Predictions with {n_bootstrap} Bootstrap CI')

        # Plot the 45-degree dotted line (y = x)
        min_val = min(continuous_actual.min(), continuous_predictions.min())
        max_val = max(continuous_actual.max(), continuous_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='45-degree Line (y = x)', zorder=5)

        plt.xlabel('Original Continuous Values (Actual Data)')
        plt.ylabel('Predicted Continuous Values')
        plt.title(f'Predicted Continuous Values vs Actual Data with {n_bootstrap} Bootstrap CIs')
        plt.legend()
        plt.show()

    def plot_categorical_predictions(self, exog=None):
        """
        Plot a bar plot showing the count of actual responses (0, 1, or (0,1)) 
        with predicted proportions and 95% confidence intervals on top of each bar.
        """
        if exog is None:
            exog = self.exog

        # Generate predictions
        predictions = self.predict(exog)

        # Classify the actual and predicted values
        actual_categorical = np.where(self.endog == 0, '0', np.where(self.endog == 1, '1', '(0, 1)'))
        predicted_categorical = np.where(predictions < 0.01, '0', np.where(predictions > 0.99, '1', '(0, 1)'))

        # Count the number of actual responses in each category
        actual_counts = {
            '0': np.sum(actual_categorical == '0'),
            '1': np.sum(actual_categorical == '1'),
            '(0, 1)': np.sum(actual_categorical == '(0, 1)')
        }

        # Calculate the predicted proportions in each category
        predicted_counts = {
            '0': np.sum(predicted_categorical == '0'),
            '1': np.sum(predicted_categorical == '1'),
            '(0, 1)': np.sum(predicted_categorical == '(0, 1)')
        }
        total_predictions = len(predicted_categorical)
        predicted_proportions = {k: predicted_counts[k] / total_predictions for k in predicted_counts}

        # Binomial confidence intervals for the predicted proportions
        conf_intervals = {}
        for category, count in predicted_counts.items():
            ci_low, ci_high = binom.interval(0.95, total_predictions, predicted_proportions[category])
            conf_intervals[category] = (ci_low / total_predictions, ci_high / total_predictions)

        # Plot actual counts as bars
        categories = ['0', '1', '(0, 1)']
        actual_values = [actual_counts[cat] for cat in categories]
        predicted_values = [predicted_proportions[cat] for cat in categories]
        error_bars = [(predicted_proportions[cat] - conf_intervals[cat][0], conf_intervals[cat][1] - predicted_proportions[cat]) for cat in categories]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bar chart for actual counts
        ax.bar(categories, actual_values, color='lightblue', label='Actual Counts', alpha=0.7)

        # Overlay the predicted proportions with error bars on top of the bars
        ax.errorbar(categories, [v * total_predictions for v in predicted_values], 
                    yerr=[[v[0] * total_predictions for v in error_bars], 
                          [v[1] * total_predictions for v in error_bars]], 
                    fmt='o', color='red', label='Predicted Proportions with CI')

        # Add labels and title
        ax.set_xlabel('Response Category')
        ax.set_ylabel('Count (Actual and Predicted)')
        ax.set_title('Actual Response Counts and Predicted Proportions with 95% CI')
        ax.legend()

        # Show the plot
        plt.show()



