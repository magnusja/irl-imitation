import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import time

import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *



class DeepIRLFC:


  def __init__(self, n_input, n_actions, lr, T, n_h1=400, n_h2=300, l2=10, deterministic_env=False, deterministic=False, sparse=False, conv=False, name='deep_irl_fc'):
    if len(n_input) > 1:
        self.height, self.width = n_input
        self.n_input = self.height * self.width
    else:
        self.n_input = n_input[0]
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.deterministic_env = deterministic_env
    self.deterministic = deterministic
    self.sparse = sparse
    self.conv = conv

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.input_s, self.reward, self.theta = self._build_network(self.name, conv)

    if self.deterministic_env:
        p_a_shape = (self.n_input, n_actions)
        p_a_dtype = tf.int32
    else:
        p_a_shape = (self.n_input, n_actions, self.n_input)
        p_a_dtype = tf.float32

    # value iteration
    if sparse:
        self.P_a = tf.sparse_placeholder(p_a_dtype, shape=p_a_shape)
        self.reduce_max_sparse = tf.sparse_reduce_max_sparse
        self.reduce_sum_sparse = tf.sparse_reduce_sum_sparse
        self.reduce_max = tf.sparse_reduce_max
        self.reduce_sum = tf.sparse_reduce_sum
        self.sparse_transpose = tf.sparse_transpose
    else:
        self.P_a = tf.placeholder(p_a_dtype, shape=p_a_shape)
        self.reduce_max = tf.reduce_max
        self.reduce_max_sparse = tf.reduce_max
        self.reduce_sum = tf.reduce_sum
        self.reduce_sum_sparse = tf.reduce_sum
        self.sparse_transpose = tf.transpose

    self.gamma = tf.placeholder(tf.float32)
    self.epsilon = tf.placeholder(tf.float32)
    self.values, self.policy = self._vi(self.reward)

    # state visitation frequency
    self.T = T
    self.mu = tf.placeholder(tf.float32, self.n_input, name='mu_placerholder')

    self.svf = self._svf(self.policy)

    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    self.grad_r = tf.placeholder(tf.float32, [self.n_input, 1])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

    self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # apply l2 loss gradients
    self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

    self.grad_norms = tf.global_norm(self.grad_theta)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())


  def _build_network(self, name, conv):
    if conv:
        input_s = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        with tf.variable_scope(name):
          #conv1 = tf_utils.conv2d(input_s, 64, (1, 1), 1)
          #conv2 = tf_utils.conv2d(conv1, 32, (1, 1), 1)
          #conv3 = tf_utils.conv2d(conv2, 32, (1, 1), 1)
          reward = tf_utils.conv2d(input_s, 1, (1, 1), 1)
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, tf.squeeze(tf.reshape(reward, (-1, self.n_input))), theta
    else:
        input_s = tf.placeholder(tf.float32, [None, self.n_input])
        with tf.variable_scope(name):
          fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.elu,
            initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
          fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.elu,
            initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
          reward = tf_utils.fc(fc2, self.n_input, scope="reward")
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, tf.squeeze(reward), theta

  def _vi(self, rewards):

      rewards_expanded = rewards #tf.tile(tf.expand_dims(rewards, 1), [1, self.n_input])

      def vi_step(values):
          if self.deterministic_env:
            new_value = tf.gather(rewards_expanded, self.P_a) + self.gamma * tf.gather(values, self.P_a)
          else:
            new_value = self.reduce_sum_sparse(self.P_a * (rewards_expanded + self.gamma * values), axis=2)

          return new_value

      def body(i, c, t):
          old_values = t.read(i)
          new_values = vi_step(old_values)
          new_values = self.reduce_max(new_values, axis=1)
          t = t.write(i + 1, new_values)

          c = tf.reduce_max(tf.abs(new_values - old_values)) > self.epsilon
          c.set_shape(())

          return i + 1, c, t

      def condition(i, c, t):
          return c

      t = tf.TensorArray(dtype=tf.float32, size=350, clear_after_read=True)
      t = t.write(0, tf.constant(0, dtype=tf.float32, shape=(self.n_input,)))

      i, _, values = tf.while_loop(condition, body, [0, True, t], parallel_iterations=1, back_prop=False,
                                   name='VI_loop')
      values = values.read(i)
      new_values = vi_step(values)

      if self.deterministic:
        policy = tf.argmax(new_values, axis=1)
      else:
        policy = tf.nn.softmax(new_values)

      return values, policy

  def _svf(self, policy):
      if not self.deterministic_env:
          if self.deterministic:
            r = tf.range(self.n_input, dtype=tf.int64)
            expanded = tf.expand_dims(policy, 1)
            tiled = tf.tile(expanded, [1, self.n_input])

            grid = tf.meshgrid(r, r)
            indices = tf.stack([grid[1], grid[0], tiled], axis=2)

            P_a_cur_policy = tf.gather_nd(self.sparse_transpose(self.P_a, (0, 2, 1)), indices)
            P_a_cur_policy = tf.transpose(P_a_cur_policy, (1, 0))
          else:
            P_a_cur_policy = self.P_a * tf.expand_dims(policy, 2)
      else:
          if self.deterministic:
            r = tf.range(self.n_input, dtype=tf.int64)
            indices = tf.stack([r, policy], axis=1)

            P_a_cur_policy = tf.gather_nd(self.P_a, indices)
            P_a_cur_policy = tf.Print(P_a_cur_policy, [P_a_cur_policy], 'P_a_cur_policy', summarize=500)
          else:
            P_a_cur_policy = self.P_a

      mu = list()
      mu.append(self.mu)
      with tf.variable_scope('svf'):
          if self.deterministic:
              for t in range(self.T - 1):
                  if self.deterministic_env:
                      cur_mu = tf.Variable(tf.constant(0, dtype=tf.float32, shape=(self.n_input,)), trainable=False)
                      cur_mu = tf.scatter_add(cur_mu, P_a_cur_policy, mu[t])
                  else:
                    cur_mu = self.reduce_sum(mu[t] * P_a_cur_policy, axis=1)
                  mu.append(cur_mu)
          else:
              for t in range(self.T - 1):
                  cur_mu = self.reduce_sum(self.reduce_sum_sparse(tf.tile(tf.expand_dims(tf.expand_dims(mu[t], 1), 2),
                                                                          [1, tf.shape(policy)[1],
                                                                           self.n_input]) * P_a_cur_policy, axis=1),
                                           axis=0)
                  mu.append(cur_mu)

      mu = tf.stack(mu)
      mu = tf.reduce_sum(mu, axis=0)
      # NOTE: it helps to scale the svf by T to recover the reward properly
      # I noticed that if it is not scaled by T then the recovered reward and the resulting value function
      # have extremely low values (usually < 0.01). With such low values it is hard to actually recover a
      # difference in the value of states (i.e. if only the last few digits after the comma differ).
      # One intuition why scaling by T is useful is to stabilize the gradients and avoid that the gradients
      # are getting too high
      # TODO: maybe gradient clipping and normalizing the svf of demonstrations and the policy might help as well
      # As a side note: This is not mentioned somewhere in the pulications (besides this youtube video:
      # https://youtu.be/d9DlQSJQAoI?t=973), but for me this countermeasure works pretty well
      return mu / self.T


  def get_theta(self):
    return self.sess.run(self.theta)

  def get_rewards(self, states):
    if self.conv:
        states = np.expand_dims(np.expand_dims(states, axis=0), axis=-1)
    else:
        states = np.expand_dims(states, axis=0)
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards

  def get_policy(self, states, P_a, gamma, epsilon=0.01):
    if self.conv:
      states = np.expand_dims(np.expand_dims(states, axis=0), axis=-1)
    else:
      states = np.expand_dims(states, axis=0)
    return self.sess.run([self.reward, self.values, self.policy],
                         feed_dict={self.input_s: states, self.P_a: P_a, self.gamma: gamma, self.epsilon: epsilon})

  def get_policy_svf(self, states, P_a, gamma, p_start_state, epsilon=0.01):

      if self.conv:
        states = np.expand_dims(np.expand_dims(states, axis=0), axis=-1)
      else:
        states = np.expand_dims(states, axis=0)
      return self.sess.run([self.reward, self.values, self.policy, self.svf],
                           feed_dict={self.input_s: states, self.P_a: P_a, self.gamma: gamma, self.mu: p_start_state, self.epsilon: epsilon})

  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, 1])
    if self.conv:
        feat_map = np.expand_dims(np.expand_dims(feat_map, axis=0), axis=-1)
    else:
        feat_map = np.expand_dims(feat_map, axis=0)
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms


def start_state_probs(trajs, n_states):
    p_start_state = np.zeros([n_states])

    for traj in trajs:
        p_start_state[traj[0].cur_state] += 1
    p_start_state = p_start_state[:] / len(trajs)

    return p_start_state

def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T) 
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

  
  returns:
    p       Nx1 vector - state visitation frequencies
  """
  tt = time.time()
  if len(P_a.shape) == 3:
    N_STATES, _, N_ACTIONS = np.shape(P_a)
  else:
    N_STATES, N_ACTIONS = np.shape(P_a)

  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T])

  mu[:, 0] = start_state_probs(trajs, N_STATES)

  num_cpus = multiprocessing.cpu_count()
  chunk_size = N_STATES // num_cpus

  if chunk_size == 0:
    chunk_size = N_STATES


  if len(P_a.shape) == 3:
      if deterministic:
        P_az = P_a[np.arange(0, N_STATES), :, policy]
      else:
        P_a = P_a.transpose(0, 2, 1)
  else:
      if deterministic:
        P_az = P_a[np.arange(N_STATES), policy]

  if len(P_a.shape) == 3:
      def step(t, start, end):
          if deterministic:
            mu[start:end, t + 1] = np.sum(mu[:, t, np.newaxis] * P_az[:, start:end], axis=0)
          else:
            mu[start:end, t + 1] = np.sum(np.sum(mu[:, t, np.newaxis, np.newaxis] * (P_a[:, :, start:end] * policy[:, :, np.newaxis]), axis=1), axis=0)
  else:
      def step(t, start, end):
          if deterministic:
            # The following needs be be done using ufunc
            # https://stackoverflow.com/questions/41990028/add-multiple-values-to-one-numpy-array-index
            # P_az[start:end] sometimes points to same state for multiple values, with the usual fancy indexing only
            # one addition (latest) would be executed!
            # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
            # mu[P_az[start:end], t + 1] += mu[start:end, t]
            np.add.at(mu, [P_az[start:end], t + 1], mu[start:end, t])
          else:
            # mu[P_a[start:end, :], t + 1] += mu[start:end, t, np.newaxis] * policy[start:end, :]
            np.add.at(mu, [P_a[start:end, :], t + 1], mu[start:end, t, np.newaxis] * policy[start:end, :])


  with ThreadPoolExecutor(max_workers=1) as e:
    for t in range(T - 1):
      futures = list()
      for i in range(0, N_STATES, chunk_size):
          futures.append(e.submit(step, t, i, min(N_STATES, i + chunk_size)))

      for f in futures:
          # Force throwing an exception if thrown by step()
          f.result()

  # for t in range(T - 1):
  #   mu[:, t+1] = (mu[:, t]*P_a[np.arange(0, N_STATES), :, policy]).sum(axis=1)

  p = np.sum(mu, 1)

  # NOTE: it helps to scale the svf by T to recover the reward properly
  # I noticed that if it is not scaled by T then the recovered reward and the resulting value function
  # have extremely low values (usually < 0.01). With such low values it is hard to actually recover a
  # difference in the value of states (i.e. if only the last few digits after the comma differ).
  # One intuition why scaling by T is useful is to stabilize the gradients and avoid that the gradients
  # are getting too high
  # TODO: maybe gradient clipping and normalizing the svf of demonstrations and the policy might help as well
  # As a side note: This is not mentioned somewhere in the pulications (besides this youtube video:
  # https://youtu.be/d9DlQSJQAoI?t=973), but for me this countermeasure works pretty well
  p /= T

  print(time.time() - tt)
  return p


def demo_svf(trajs, n_states):
  """
  compute state visitation frequences from demonstrations
  
  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences   
  """

  p = np.zeros(n_states)
  for traj in trajs:
    for step in traj:
      p[step.cur_state] += 1
  p = p/len(trajs)
  return p


def compute_state_visition_freq_old(P_a, gamma, trajs, policy, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T)
    using dynamic programming
    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

    returns:
      p       Nx1 vector - state visitation frequencies
    """
    if len(P_a.shape) == 3:
        N_STATES, _, N_ACTIONS = np.shape(P_a)
    else:
        N_STATES, N_ACTIONS = np.shape(P_a)

    T = len(trajs[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)
    for t in range(T - 1):
      for s in range(N_STATES):
            if deterministic:
                if len(P_a.shape) == 3:
                    mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
                else:
                    mu[P_a[s, int(policy[s])], t + 1] += mu[s, t]
            else:
                if len(P_a.shape) == 3:
                    mu[s, t + 1] = sum(
                        [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
                        range(N_STATES)])
                else:
                    for a1 in range(N_ACTIONS):
                        mu[P_a[s, a1], t + 1] += mu[s, t] * policy[s, a1]


    print(mu)
    p = np.sum(mu, 1)
    print('SUM SVF', p.sum())

    # NOTE: it helps to scale the svf by T to recover the reward properly
    # I noticed that if it is not scaled by T then the recovered reward and the resulting value function
    # have extremely low values (usually < 0.01). With such low values it is hard to actually recover a
    # difference in the value of states (i.e. if only the last few digits after the comma differ).
    # One intuition why scaling by T is useful is to stabilize the gradients and avoid that the gradients
    # are getting too high
    # TODO: maybe gradient clipping and normalizing the svf of demonstrations and the policy might help as well
    # As a side note: This is not mentioned somewhere in the pulications (besides this youtube video:
    # https://youtu.be/d9DlQSJQAoI?t=973), but for me this countermeasure works pretty well
    p /= T
    return p


def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters, conv, sparse):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                       landing at state s1 when taking action
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """

  # tf.set_random_seed(1)

  if len(P_a.shape) == 3:
      N_STATES, _, N_ACTIONS = np.shape(P_a)
  else:
      N_STATES, N_ACTIONS = np.shape(P_a)

  deterministic = True

  # init nn model
  nn_r = DeepIRLFC(feat_map.shape, N_ACTIONS, lr, len(trajs[0]), 3, 3, deterministic_env=len(P_a.shape) == 2,  deterministic=deterministic, conv=conv, sparse=sparse)

  # find state visitation frequencies using demonstrations
  mu_D = demo_svf(trajs, N_STATES)
  p_start_state = start_state_probs(trajs, N_STATES)

  if len(P_a.shape) == 3:
      P_a_t = P_a.transpose(0, 2, 1)
      if sparse:
        mask = P_a_t > 0
        indices = np.argwhere(mask)
        P_a_t = tf.SparseTensorValue(indices, P_a_t[mask], P_a_t.shape)
  else:
      P_a_t = P_a

  grads = list()

  # training 
  for iteration in range(n_iters):
    if iteration % (n_iters/10) == 0:
      print 'iteration: {}'.format(iteration)

    # compute the reward matrix
    # rewards = nn_r.get_rewards(feat_map)

    # compute policy
    #_, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=deterministic)

    # compute rewards and policy at the same time
    #t = time.time()
    rewards, values, policy = nn_r.get_policy(feat_map, P_a_t, gamma, 0.000001)
    #print('tensorflow VI', time.time() - t)
    
    # compute expected svf
    #mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=deterministic)

    rewards, values, policy, mu_exp = nn_r.get_policy_svf(feat_map, P_a_t, gamma, p_start_state, 0.000001)
    #print(rewards)

    assert_all_the_stuff(rewards, policy, values, mu_exp, P_a, N_ACTIONS, N_STATES, trajs, gamma, deterministic)

    # compute gradients on rewards:
    grad_r = mu_D - mu_exp
    grads.append(grad_r)

    # apply gradients to the neural network
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)


  print('grad mean', np.mean(grads, axis=0))
  print('grad std', np.std(grads, axis=0))

  rewards = nn_r.get_rewards(feat_map)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)

def assert_all_the_stuff(rewards, policy, values, mu_exp, P_a, N_ACTIONS, N_STATES, trajs, gamma, deterministic):

    def assert_vi(P_a):
        assert_values, assert_policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.000001,
                                                                       deterministic=deterministic)
        assert_values_old, assert_policy_old = value_iteration.value_iteration_old(P_a, rewards, gamma, error=0.000001,
                                                                                   deterministic=deterministic)

        if len(P_a) == 3:
            assert_values2 = value_iteration.optimal_value(N_STATES, N_ACTIONS, P_a_t, rewards, gamma, threshold=0.000001)

            assert (np.abs(assert_values - assert_values2) < 0.0001).all()

        assert (np.abs(assert_values - assert_values_old) < 0.0001).all()
        assert (np.abs(values - assert_values) < 0.0001).all()
        assert (np.abs(values - assert_values_old) < 0.0001).all()

        # print(assert_policy)
        # print(assert_policy_old)
        # print(policy)
        # print(values)
        # print(assert_values)
        # print(rewards)
        assert (np.abs(assert_policy - assert_policy_old) < 0.0001).all()
        assert (np.abs(policy - assert_policy) < 0.0001).all()
        assert (np.abs(policy - assert_policy_old) < 0.0001).all()

    assert_vi(P_a)
    if len(P_a.shape) == 2:
        print('creating full transistion matrix')
        # construct full sparse transisiton matrix and make sure values are the same
        P_a_t = np.zeros((N_STATES, N_ACTIONS, N_STATES))
        P_a_t[P_a] = 1
        assert_vi(P_a)

    assert (np.abs(mu_exp - compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=deterministic)) < 0.00001).all()
    assert (
    np.abs(mu_exp - compute_state_visition_freq_old(P_a, gamma, trajs, policy, deterministic=deterministic)) < 0.00001).all()
    
    print('tf sum SVF', mu_exp.sum())





