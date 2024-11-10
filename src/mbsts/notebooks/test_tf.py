import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
sts = tfp.sts

# Ensure the correct shape of the data
df_hist = np.random.rand(8787, 2)  # Dummy data for illustration
num_timesteps, num_series = df_hist.shape
num_predictors = 1
horizon = 15 * 24  # 15 days ahead, hourly data

# Convert the data to a numpy array and change type to np.float32
observed_time_series = df_hist.astype(np.float32)

# Create some dummy predictors for illustration
predictors = tf.random.normal([num_timesteps, num_predictors])

# Define model components
trend = sts.LocalLinearTrend(observed_time_series=observed_time_series, name='trend')
seasonal = sts.Seasonal(num_seasons=24, observed_time_series=observed_time_series, name='seasonal')
cycle = sts.SmoothSeasonal(period=24 * 7, frequency_multipliers=[1., 2., 3.], observed_time_series=observed_time_series,
                           name='cycle')
regression = sts.LinearRegression(design_matrix=predictors, name='predictors')

# Define the model
sts_model = sts.Sum([trend, seasonal, cycle, regression], observed_time_series=observed_time_series)


# Define the joint distribution
def get_param_prior_values(parameters):
    return [param.prior for param in parameters]


param_priors = get_param_prior_values(sts_model.parameters)


def sts_model_joint_distribution(observed_time_series, sts_model, num_timesteps, param_vals):
    def sts_model_fn():
        param_list = [param for param in param_vals]
        state_space_model = sts_model.make_state_space_model(
            num_timesteps=num_timesteps,
            param_vals=param_list,
            initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=tf.ones([sts_model.latent_size]))
        )
        return state_space_model

    return tfd.JointDistributionCoroutineAutoBatched(sts_model_fn)


# Define the joint distribution and conditioning
joint_dist = sts_model_joint_distribution(observed_time_series, sts_model, num_timesteps, param_priors)

# Pin the parameters to their values (not the observed data itself)
pinned_joint_dist = joint_dist.experimental_pin(observed_time_series=observed_time_series)


# Function to compute log_prob
def compute_log_prob(pinned_joint_dist, observed_time_series):
    def vectorized_model_fn(time_series):
        return pinned_joint_dist.log_prob(tf.expand_dims(time_series, axis=-1))

    log_probs = tf.vectorized_map(vectorized_model_fn, tf.transpose(observed_time_series))
    return tf.reduce_sum(log_probs)


# Calculate the log probability
log_prob = compute_log_prob(pinned_joint_dist, observed_time_series)
