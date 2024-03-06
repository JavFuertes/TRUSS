class BayesianOptimizer(): 
    def __init__(self, target_func, x_init, y_init, n_iter, batch_size, bounds):
        
        self.config = {
            'x_init': x_init,
            'y_init': y_init,
            'target_func': target_func,
            'n_iter': n_iter,
            'batch_size': batch_size,
            'bounds': bounds,
            'boundsmm': [(0,1), (0,1)], # Bounds consistent with Minmax scaling 
        }
        self.config['x_bounds'] = [self.config['boundsmm'][0]] * 5 + [self.config['boundsmm'][1]] * 15
        
        self.best_samples_ = pd.DataFrame(columns=['x', 'y', 'ei'])
        self.metrics = {
            'distances': [],
            'uncertainty': [],
            'y_loss': [],
        }
        
        gp_config = {
            'matern_length_scale_bounds': (1e-3, 1e5),
            'initial_length_scale': np.ones(20),
            'nu': 0.5,
        }
        kernel = Matern(length_scale=gp_config['initial_length_scale'], 
                        length_scale_bounds=gp_config['matern_length_scale_bounds'], 
                        nu=gp_config['nu'])
        self.gauss_pr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    def estimate_uncertainty(self, n=10):
        sampled_points = []
        uncertainties = []
        for _ in range(100):  # Sample 100 times
                x_start = self.next_guess()  # Random starting point for exploration 
                x_start_sc = self.scaler(x_start, 'scale')       

                y, cov = self.gauss_pr.predict(np.array(x_start_sc).reshape(1, -1), return_std=True)
                sampled_points.append(x_start_sc)
                uncertainties.append(cov[0])  # We use the standard deviation (uncertainty)

        uncertainties = np.array(uncertainties)
        sampled_points = np.array(sampled_points)
        sorted_indices = np.argsort(-uncertainties)
        top_n_points = sampled_points[sorted_indices[:n]]

        return top_n_points
        
    def _acquisition_function(self, x_new):
        '''
        Calculates the expected improvement at a given point x_new
        '''
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array(x_new).reshape(1, -1), return_std=True)
        if sigma_y_new == 0.0:
            return 0.0

        min_mean_y = np.min(self.config['y_init'])
        z = (min_mean_y - mean_y_new) / sigma_y_new
        exp_imp = (min_mean_y - mean_y_new) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        return -exp_imp
  
    def _get_next_probable_point(self):
        '''
        We only standardise the data after the minimisation so the LOSS FUNCTION operates with standard scales
        ### FIX , Before We were only returning if better this does not allow the gaussian process to understand the data as well
        '''
        x_optimal = None
        x_next_sc = None
        max_acq_value = float('inf')

        for _ in range(self.config['batch_size']):
            x_start_sc = self.estimate_uncertainty(n=10)
            res = minimize(fun=self._acquisition_function, x0=x_start_sc, bounds=self.config['x_bounds'], method='L-BFGS-B')
            if res.fun < max_acq_value:
                max_acq_value = res.fun
                x_next_sc = res.x
                x_optimal = self.scaler(x_next_sc, 'descale')

        return x_optimal, x_next_sc, -max_acq_value
    
    def _extend_prior_with_posterior_data(self, x, y):
        self.config['x_init'] = np.append(self.config['x_init'], np.array([x]), axis=0)
        self.config['y_init'] = np.append(self.config['y_init'], np.array([y]), axis=0)

  
    def optimize(self):
        """
        Performs optimization to minimize the target function using Gaussian Process Regression.

        Iteratively selects and evaluates new points based on the Expected Improvement (EI) criterion,
        updating the optimal point and minimum value found.

        Returns:
            optimal_x (np.ndarray): Optimal input values minimizing the target function.
            y_min (float): Minimum value of the target function observed.
        """
        y_min_ind = np.argmin(self.config['y_init'])
        y_min = self.config['y_init'][y_min_ind]
        optimal_x = self.config['x_init'][y_min_ind]
        optimal_ei = None

        for i in range(self.config['n_iter']):
            print(f'Iteration: {i} Best loss = {y_min:.2f}\n')
            self.gauss_pr.fit(self.config['x_init'], self.config['y_init'])
            x_next, x_next_sc, ei = self._get_next_probable_point()
            y_next = self.config['target_func'](np.array(x_next))  # Call target function
            self._extend_prior_with_posterior_data(x_next_sc, y_next)
            if y_next < y_min:
                y_min = y_next
                optimal_x = x_next
                optimal_ei = ei

            self.metrics['best_samples_'] = pd.concat([self.metrics['best_samples_'], pd.DataFrame({"x": [optimal_x], "y": [y_min], "ei": [optimal_ei]})], ignore_index=True)
        return optimal_x, y_min

    @staticmethod
    def next_guess(bounds):
        """
        Generate a next guess based on the bounds provided.
        
        Parameters:
        - bounds: A tuple containing two elements, each an (min, max) tuple representing bounds.
        
        Returns:
        - A numpy array containing random values within the specified bounds.
        """
        y_bound = bounds[0]
        A_bound = bounds[1]
        y = np.random.uniform(y_bound[0], y_bound[1], 5)
        Areas = np.random.uniform(A_bound[0], A_bound[1], 15)
        x_list = np.append(y, Areas)
        return x_list

    @staticmethod
    def scaler(x, mode, OBJECT):
        """
        Scales or descales the provided array x using another object's scaler method.
        
        Parameters:
        - x: The input array to scale or descale.
        - mode: A string indicating the scaling mode ('scale' or 'descale').
        - OBJECT: An instance of another class with a 'scaler' method to perform the actual scaling.
        
        Returns:
        - A numpy array that has been scaled or descaled by the OBJECT's scaler method.
        """
        y_coords = x[:5]
        y_coords_sc = OBJECT.scaler(mode, y_coords, 'y')
        areas = x[5:]
        areas_sc = OBJECT.scaler(mode, areas, 'a')
        x_out = np.concatenate([y_coords_sc, areas_sc])
        return x_out 
