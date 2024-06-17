from mur_wind_prev import get_wind_prev_pivot
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb

ZONES = (
"22Bolo", "24Ama1", "26Fanh", "27Arr1", "29Sob2", "44Milagre", "48Vale_de", "60Maravil", "61MaraviII", "68Jogui")
HIST_NUM_YEARS = 4

# # # Hourly Wind generation data with multiple time series as columns
# df = get_wind_prev_pivot(ZONES, HIST_NUM_YEARS)
# #save df to pkl
# df.to_pickle('wind_generation_data.pkl')

#load df from pkl
df = pd.read_pickle('wind_generation_data.pkl')

#df contains hourly wind generation data including historical data and forecasted for each time series. I want to split the data into historical and forecasted data based now time filtering the datetimeindex. getting df_hist and df_forecast
df_hist = df.loc[df.index < pd.Timestamp.now()]
df_forecast = df.loc[df.index >= pd.Timestamp.now()]

# Model parameters
num_timesteps, num_series = df_hist.shape
num_predictors = 1
horizon = 15 * 24  # 15 days ahead, hourly data

# Convert the data to a numpy array and change type to np.float32
observed_time_series = df_hist.values.astype(np.float32)

# Create some dummy predictors for illustration
predictors = tf.random.normal([num_timesteps, num_predictors])
# Example predictors
# predictors = tf.Variable(tf.random.normal([num_timesteps, num_predictors]), name='predictors')

trend = tfp.sts.LocalLinearTrend(observed_time_series=observed_time_series, name='trend')

# Daily seasonality for hourly data (24 hours)
seasonal = tfp.sts.Seasonal(num_seasons=24, observed_time_series=observed_time_series, name='seasonal')

# Weekly cycle for hourly data (24*7 hours)
cycle = tfp.sts.SmoothSeasonal(period=24 * 7, frequency_multipliers=[1., 2., 3.],
                               observed_time_series=observed_time_series, name='cycle')  # Weekly cycle

regression = tfp.sts.LinearRegression(design_matrix=predictors, name='predictors')

# Define the model
sts_model = tfp.sts.Sum([
    trend,  # 2 latent states
    seasonal,  # 23 latent states
    cycle,  # 6 latent states
    regression  # 0
], observed_time_series=observed_time_series)


# Define the joint distribution
def get_param_prior_values(parameters):
    return [param.prior for param in parameters]


def sts_model_joint_distribution(observed_time_series, sts_model, num_timesteps, num_series):
    param_vals = get_param_prior_values(sts_model.parameters)

    def sts_model_fn():
        param_list = []
        for param in param_vals:
            param_list.append((yield param))

        # Vectorized state space model
        def vectorized_model_fn(time_series):
            state_space_model = sts_model.make_state_space_model(
                num_timesteps=num_timesteps,
                param_vals=param_list,
                initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=tf.ones([sts_model.latent_size]))
            )

            return state_space_model.log_prob(tf.expand_dims(time_series, axis=-1))
            # return state_space_model.log_prob(time_series)

        # log_probs = tf.vectorized_map(vectorized_model_fn, tf.range(num_series))
        log_probs = tf.vectorized_map(vectorized_model_fn, tf.transpose(observed_time_series))
        yield tf.reduce_sum(log_probs)

    return tfd.JointDistributionCoroutineAutoBatched(sts_model_fn)


# Define the joint distribution and conditioning
joint_dist = sts_model_joint_distribution(observed_time_series, sts_model, num_timesteps, num_series)
# Pin the observed time series to the joint distribution

pinned_joint_dist = joint_dist.experimental_pin(observed_time_series=observed_time_series)
# Inference using MCMC
num_results = 500
num_burnin_steps = 300


# Define the HMC transition kernel
@tf.function
def run_chain():
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=pinned_joint_dist.unnormalized_log_prob,
        step_size=0.1,
        num_leapfrog_steps=3
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=int(num_burnin_steps * 0.8),
        target_accept_prob=0.75
    )
    initial_chain_state = [tf.zeros_like(param.initial_value()) for param in sts_model.parameters]
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_chain_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
    )


# Execute the MCMC chain
samples, is_accepted = run_chain()


# Forecast the future
def forecast(model, samples, num_steps_forecast):
    forecast_dist = tfp.sts.forecast(
        model,
        observed_time_series=observed_time_series,
        parameter_samples=samples,
        num_steps_forecast=num_steps_forecast
    )

    return forecast_dist.sample(num_steps_forecast).numpy()


forecast_samples = forecast(sts_model, samples, horizon)
